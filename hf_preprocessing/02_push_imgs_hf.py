#!/usr/bin/env python3
"""
Clean version of HF dataset pusher that:
1. Purges ALL files before pushing (including dataset_infos.json)
2. Pushes each config as a separate entity
3. Maintains consistent train/test splits across configs
4. Has test mode with limited tiles
"""
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, Features, Image, Value
from datasets.features import Sequence, Array2D
from huggingface_hub import HfApi
from transformers import AutoModel, AutoImageProcessor

# -------------------------
# Config
# -------------------------
hf_org = 'mpg-ranch'
hf_repo = 'drone-lsr'
RGB_TILE_DIR = Path('data/raster/tiles/rgb')
CANOPY_TILE_DIR = Path('data/raster/tiles/chm')
TARGET_SHARD_SIZE_MB = 500
TIMES = [1000, 1200, 1500]   # t0, t1, t2
BATCH_SIZE = 16
WRITER_BATCH_SIZE = 2000
TILE_SIZE = 1024
CANOPY_SCALE = 100  # meters to centimeters for integer storage
CANOPY_FILL_VALUE = np.iinfo(np.int32).min
TRAIN_TEST_SEED = 42

# Model configurations
MODEL_CONFIGS = {
    'dinov2_base': {
        'id': 'facebook/dinov2-base',
        'description': 'DINOv2 Base (ViT-B/14)'
    },
    'dinov3_sat': {
        'id': 'facebook/dinov3-vitl16-pretrain-sat493m',
        'description': 'DINOv3 Large with SAT pretraining'
    }
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------
# Device setup
# -------------------------
if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# -------------------------
# Model loading and inspection
# -------------------------
def load_and_inspect_model(model_config_name, model_id):
    """Load model and print its configuration details."""
    print(f"\n=== Loading {model_config_name} ===")
    print(f"Model ID: {model_id}")

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    config = model.config
    n_registers = getattr(config, 'num_register_tokens', 0)
    hidden_size = config.hidden_size

    # Calculate patch grid based on actual processing size (224x224)
    processing_size = 224  # We process all images at 224x224
    patch_size = config.patch_size

    h = w = processing_size // patch_size
    n_patches = h * w

    print(f"  Hidden size: {hidden_size}")
    print(f"  Processing size: {processing_size}x{processing_size}")
    print(f"  Patch size: {patch_size}")
    print(f"  Patch grid: {h}×{w} = {n_patches} patches")
    print(f"  Register tokens: {n_registers}")
    print(f"  Total tokens: 1 (CLS) + {n_registers} (registers) + {n_patches} (patches)")

    return processor, model, {
        'n_registers': n_registers,
        'hidden_size': hidden_size,
        'n_patches': n_patches,
        'patch_grid': (h, w)
    }

@torch.no_grad()
def encode_images(image_paths, model_info):
    """Encode images from paths with the specified model."""
    from PIL import Image

    processor = model_info['processor']
    model = model_info['model']
    n_registers = model_info['specs']['n_registers']

    # Load images from paths and ensure RGB mode
    pil_images = []
    for path in image_paths:
        im = Image.open(path)
        if im.mode != "RGB":
            im = im.convert("RGB")
        pil_images.append(im)

    # Use processor with explicit size override to ensure 224x224 for both models
    inputs = processor(images=pil_images, size={'height': 224, 'width': 224}, return_tensors="pt").to(device)
    outputs = model(**inputs)
    tokens = outputs.last_hidden_state              # [B, 1+R+N, D]

    cls = tokens[:, 0]                              # [B, D]
    patches = tokens[:, 1 + n_registers:]           # [B, N, D]

    return cls.cpu().numpy().astype(np.float32), patches.cpu().numpy().astype(np.float32)

# -------------------------
# Tile scanning
# -------------------------
def scan_tiles(max_tiles=None):
    """Scan tiles directory with optional limit for testing."""
    logging.info("Scanning tiles directory...")
    tiles_by_location = defaultdict(dict)
    for tile_path in RGB_TILE_DIR.glob('*.png'):
        parts = tile_path.stem.split('_')  # {ROW}_{COL}_{TIME}.png
        if len(parts) != 3:
            continue
        row, col, time = parts
        try:
            time = int(time)
            if time not in TIMES:
                continue
        except ValueError:
            continue
        location_id = f"{row}_{col}"
        tiles_by_location[location_id][time] = tile_path

    complete_tiles = {}
    for location_id, time_dict in tiles_by_location.items():
        if len(time_dict) == 3 and all(t in time_dict for t in TIMES):
            complete_tiles[location_id] = time_dict

    # Limit tiles for testing if specified
    if max_tiles is not None:
        complete_tiles = dict(list(complete_tiles.items())[:max_tiles])
        logging.info(f"Limited to {max_tiles} tile sets for testing")
    else:
        logging.info(f"Found {len(complete_tiles)} complete tile sets")

    return complete_tiles

def canopy_tile_path(location_id: str) -> Path:
    """Return the expected path for a canopy tile."""
    return CANOPY_TILE_DIR / f"{location_id}.tif"

def load_canopy_tile(location_id: str):
    """Load and quantize the canopy height tile for a location."""
    path = canopy_tile_path(location_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing canopy tile for {location_id} at {path}")

    with rasterio.open(path) as src:
        tile = src.read(1)
        if tile.shape != (TILE_SIZE, TILE_SIZE):
            raise ValueError(f"Unexpected canopy tile shape {tile.shape} for {location_id}")
        nodata = src.nodata

    tile = tile.astype(np.float32, copy=False)
    valid = np.isfinite(tile)
    if nodata is not None and not np.isnan(nodata):
        valid &= tile != nodata

    canopy_tile = np.full(tile.shape, CANOPY_FILL_VALUE, dtype=np.int32)
    if np.any(valid):
        scaled = np.rint(tile[valid] * CANOPY_SCALE).astype(np.int32)
        canopy_tile[valid] = scaled
    return canopy_tile

# -------------------------
# Encoding functions
# -------------------------
def encode_batch_for_model(batch, model_info):
    """Encode batch for a specific model."""
    cls0, patch0 = encode_images(batch['image_t0'], model_info)
    cls1, patch1 = encode_images(batch['image_t1'], model_info)
    cls2, patch2 = encode_images(batch['image_t2'], model_info)
    return {
        'cls_t0': cls0,
        'cls_t1': cls1,
        'cls_t2': cls2,
        'patch_t0': patch0,
        'patch_t1': patch1,
        'patch_t2': patch2,
    }

def purge_all_files(api, repo_id):
    """Purge dataset files (parquet) and metadata (dataset_infos.json) from repository."""
    print(f"\n=== PURGING DATASET FILES FROM {repo_id} ===")
    try:
        all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

        # Filter to only dataset files we want to delete
        files_to_delete = [
            f for f in all_files
            if f.endswith('.parquet') or f == 'dataset_infos.json'
        ]

        if not files_to_delete:
            print("  No dataset files to delete")
            return

        for file_path in files_to_delete:
            try:
                api.delete_file(path_in_repo=file_path, repo_id=repo_id, repo_type="dataset")
                print(f"  Deleted: {file_path}")
            except Exception as e:
                print(f"  Failed to delete {file_path}: {e}")

        # Show what was preserved
        preserved_files = [f for f in all_files if f not in files_to_delete]
        if preserved_files:
            print(f"  Preserved: {', '.join(preserved_files[:3])}{'...' if len(preserved_files) > 3 else ''}")

        print(f"✅ Purged {len(files_to_delete)} dataset files from repository")
    except Exception as e:
        print(f"⚠️ Warning: Could not purge files: {e}")

def push_config_separately(api, repo_id, config_name, train_dataset, test_dataset):
    """Push a single config to HuggingFace with separate train/test pushes."""
    print(f"\n=== Pushing {config_name} config ===")

    try:
        # Push train split
        train_dataset.push_to_hub(
            repo_id,
            config_name=config_name,
            split='train',
            max_shard_size=f"{TARGET_SHARD_SIZE_MB}MB",
            commit_message=f"Push {config_name} train split"
        )
        print(f"✅ Pushed {config_name} train: {train_dataset.num_rows} samples")

        # Push test split
        test_dataset.push_to_hub(
            repo_id,
            config_name=config_name,
            split='test',
            max_shard_size=f"{TARGET_SHARD_SIZE_MB}MB",
            commit_message=f"Push {config_name} test split"
        )
        print(f"✅ Pushed {config_name} test: {test_dataset.num_rows} samples")

    except Exception as e:
        print(f"❌ Failed to push {config_name}: {e}")
        raise

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Create multi-config HuggingFace dataset")
    parser.add_argument("--test", action="store_true",
                       help="Test mode: use only 10 tiles")
    parser.add_argument("--no-purge", action="store_true",
                       help="Skip purging existing files")
    parser.add_argument("--debug", action="store_true",
                       help="Process but don't push to HF")
    args = parser.parse_args()

    repo_id = f"{hf_org}/{hf_repo}"
    api = HfApi() if not args.debug else None

    # Scan tiles (limited if test mode)
    max_tiles = 10 if args.test else None
    tiles_dict = scan_tiles(max_tiles)
    if not tiles_dict:
        logging.error("No complete tile sets found")
        return 1

    if not CANOPY_TILE_DIR.exists():
        logging.error("Missing canopy tile directory at %s", CANOPY_TILE_DIR)
        return 1

    # Load models
    print("\n=== Loading Models ===")
    models = {}
    for config_name, config_info in MODEL_CONFIGS.items():
        processor, model, specs = load_and_inspect_model(config_name, config_info['id'])
        models[config_name] = {
            'processor': processor,
            'model': model,
            'specs': specs,
            'description': config_info['description']
        }

    # Create base records with images and canopy
    print("\n=== Creating Base Records ===")
    records = []
    for location_id in sorted(tiles_dict.keys()):
        time_paths = tiles_dict[location_id]
        try:
            canopy_tile = load_canopy_tile(location_id)
        except (FileNotFoundError, ValueError) as err:
            logging.error(str(err))
            continue  # Skip this tile set

        records.append({
            'image_t0': str(time_paths[1000]),
            'image_t1': str(time_paths[1200]),
            'image_t2': str(time_paths[1500]),
            'idx': location_id,
            'canopy_height': canopy_tile.tolist(),
        })

    if not records:
        logging.error("No valid tiles available for processing")
        return 1

    print(f"Processing {len(records)} tile sets...")

    # Create base DataFrame
    df_base = pd.DataFrame(records)

    # -------------------------
    # Create train/test split ONCE for all configs
    # -------------------------
    print("\n=== Creating Consistent Train/Test Split ===")
    all_indices = sorted(df_base['idx'].unique())
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=TRAIN_TEST_SEED,
        shuffle=True
    )
    print(f"Split: {len(train_indices)} train, {len(test_indices)} test")

    # Create efficient index mappings for splitting
    train_indices_set = set(train_indices)
    test_indices_set = set(test_indices)

    # Pre-compute index positions for efficient dataset splitting
    train_positions = [i for i, idx in enumerate(df_base['idx']) if idx in train_indices_set]
    test_positions = [i for i, idx in enumerate(df_base['idx']) if idx in test_indices_set]

    print(f"Index positions: {len(train_positions)} train, {len(test_positions)} test")

    # -------------------------
    # Purge repository if requested
    # -------------------------
    if not args.debug and not args.no_purge:
        purge_all_files(api, repo_id)

    # -------------------------
    # Process and push each config separately
    # -------------------------

    # 1. Default config (images + canopy)
    print("\n=== Processing Default Config ===")
    default_features = Features({
        'idx': Value('string'),
        'image_t0': Image(),
        'image_t1': Image(),
        'image_t2': Image(),
        'canopy_height': Array2D((TILE_SIZE, TILE_SIZE), dtype='int32'),
    })

    default_dataset = Dataset.from_pandas(df_base, features=default_features, preserve_index=False)

    # Split default dataset efficiently using select()
    default_train = default_dataset.select(train_positions)
    default_test = default_dataset.select(test_positions)

    print(f"Default: {default_train.num_rows} train, {default_test.num_rows} test")

    if not args.debug:
        push_config_separately(api, repo_id, 'default', default_train, default_test)

    # 2. Model embedding configs
    for model_name, model_info in models.items():
        print(f"\n=== Processing {model_name} Config ===")
        specs = model_info['specs']

        # Create features for this model
        embedding_features = Features({
            'idx': Value('string'),
            'cls_t0': Sequence(Value('float32'), length=specs['hidden_size']),
            'cls_t1': Sequence(Value('float32'), length=specs['hidden_size']),
            'cls_t2': Sequence(Value('float32'), length=specs['hidden_size']),
            'patch_t0': Array2D((specs['n_patches'], specs['hidden_size']), dtype='float32'),
            'patch_t1': Array2D((specs['n_patches'], specs['hidden_size']), dtype='float32'),
            'patch_t2': Array2D((specs['n_patches'], specs['hidden_size']), dtype='float32'),
        })

        # Create dataset with idx and image paths
        df_embedding = df_base[['idx', 'image_t0', 'image_t1', 'image_t2']].copy()
        embedding_dataset = Dataset.from_pandas(df_embedding, preserve_index=False)

        # Encode with this model
        print(f"Encoding with {model_info['description']}...")
        embedding_dataset = embedding_dataset.map(
            lambda batch: encode_batch_for_model(batch, model_info),
            batched=True,
            batch_size=BATCH_SIZE,
            features=embedding_features,
            writer_batch_size=WRITER_BATCH_SIZE,
            remove_columns=['image_t0', 'image_t1', 'image_t2']
        )

        # Split embedding dataset efficiently using select()
        embedding_train = embedding_dataset.select(train_positions)
        embedding_test = embedding_dataset.select(test_positions)

        print(f"{model_name}: {embedding_train.num_rows} train, {embedding_test.num_rows} test")

        # Verify consistency
        train_idx_set = set(embedding_train['idx'])
        test_idx_set = set(embedding_test['idx'])
        assert train_idx_set == set(train_indices), f"Train indices mismatch for {model_name}"
        assert test_idx_set == set(test_indices), f"Test indices mismatch for {model_name}"

        if not args.debug:
            push_config_separately(api, repo_id, model_name, embedding_train, embedding_test)

    print(f"\n✅ Successfully processed all configs")

    if args.test:
        print("\n⚠️ TEST MODE: Only processed 10 tiles per config")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())