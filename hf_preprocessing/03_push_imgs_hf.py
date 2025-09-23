#!/usr/bin/env python3
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
hf_repo = 'light-stable-semantics'
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

# Load both models
models = {}
for config_name, config_info in MODEL_CONFIGS.items():
    processor, model, specs = load_and_inspect_model(config_name, config_info['id'])
    models[config_name] = {
        'processor': processor,
        'model': model,
        'specs': specs,
        'description': config_info['description']
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
# Tile scanning (unchanged)
# -------------------------
def scan_tiles():
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
# Encoding functions for each config
# -------------------------
def encode_batch_for_model(batch, model_name):
    """Encode batch for a specific model."""
    model_info = models[model_name]
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

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Create multi-config HuggingFace dataset")
    parser.add_argument("--debug", action="store_true",
                       help="Process and encode images but don't push to HF")
    args = parser.parse_args()

    repo_id = f"{hf_org}/{hf_repo}"

    if not args.debug:
        api = HfApi()

    # Scan and organize tiles
    tiles_dict = scan_tiles()
    if not tiles_dict:
        logging.error("No complete tile sets found")
        return 1

    if not CANOPY_TILE_DIR.exists():
        logging.error("Missing canopy tile directory at %s", CANOPY_TILE_DIR)
        return 1

    # Create base records with images and canopy
    records = []
    for location_id in sorted(tiles_dict.keys()):
        time_paths = tiles_dict[location_id]
        try:
            canopy_tile = load_canopy_tile(location_id)
        except (FileNotFoundError, ValueError) as err:
            logging.error(str(err))
            return 1

        records.append({
            'image_t0': str(time_paths[1000]),
            'image_t1': str(time_paths[1200]),
            'image_t2': str(time_paths[1500]),
            'idx': location_id,
            'canopy_height': canopy_tile.tolist(),
        })

    if not records:
        logging.error("No tiles available for processing")
        return 1

    print(f"\nProcessing {len(records)} tile sets...")

    # Create base DataFrame
    df_base = pd.DataFrame(records)

    # -------------------------
    # Config 1: Default (images + canopy)
    # -------------------------
    print("\n=== Creating default config ===")
    default_features = Features({
        'idx': Value('string'),
        'image_t0': Image(),
        'image_t1': Image(),
        'image_t2': Image(),
        'canopy_height': Array2D((TILE_SIZE, TILE_SIZE), dtype='int32'),
    })

    default_dataset = Dataset.from_pandas(df_base, features=default_features, preserve_index=False)
    print(f"Default config: {default_dataset.num_rows} samples")

    # -------------------------
    # Config 2 & 3: Model embeddings
    # -------------------------
    datasets = {'default': default_dataset}

    for model_name, model_info in models.items():
        print(f"\n=== Creating {model_name} config ===")
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

        # Create dataset with idx and image paths for encoding
        df_embedding = df_base[['idx', 'image_t0', 'image_t1', 'image_t2']].copy()
        embedding_dataset = Dataset.from_pandas(df_embedding, preserve_index=False)

        # Encode with this model
        print(f"Encoding with {model_info['description']}...")
        embedding_dataset = embedding_dataset.map(
            lambda batch: encode_batch_for_model(batch, model_name),
            batched=True,
            batch_size=BATCH_SIZE,
            features=embedding_features,
            writer_batch_size=WRITER_BATCH_SIZE,
            remove_columns=['image_t0', 'image_t1', 'image_t2']  # Remove temp image cols
        )

        print(f"{model_name} config: {embedding_dataset.num_rows} samples")
        datasets[model_name] = embedding_dataset

    # -------------------------
    # Create consistent train/test splits based on idx
    # -------------------------
    print("\n=== Creating consistent train/test splits ===")

    # Split indices first to ensure consistency across all configs
    all_indices = sorted(df_base['idx'].unique())
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=TRAIN_TEST_SEED,
        shuffle=True
    )

    print(f"Split {len(all_indices)} unique indices: {len(train_indices)} train, {len(test_indices)} test")

    # Apply consistent splits to each config
    split_dataset_dict = {}
    for config_name, dataset in datasets.items():
        # Create masks based on idx values
        train_mask = [idx in train_indices for idx in dataset['idx']]
        test_mask = [idx in test_indices for idx in dataset['idx']]

        # Filter datasets
        train_dataset = dataset.filter(lambda example, idx: example['idx'] in train_indices, with_indices=True)
        test_dataset = dataset.filter(lambda example, idx: example['idx'] in test_indices, with_indices=True)

        split_dataset_dict[config_name] = {
            'train': train_dataset,
            'test': test_dataset
        }

        print(f"{config_name}: {train_dataset.num_rows} train, {test_dataset.num_rows} test")

        # Verify consistency (optional check)
        train_idx_set = set(train_dataset['idx'])
        test_idx_set = set(test_dataset['idx'])
        assert train_idx_set == set(train_indices), f"Train indices mismatch for {config_name}"
        assert test_idx_set == set(test_indices), f"Test indices mismatch for {config_name}"

    print("\n✅ Verified: All configs have identical train/test splits based on idx")

    if args.debug:
        print("\n=== DEBUG MODE: Skipping HuggingFace upload ===")
        print("Dataset structure:")
        for config_name, splits in split_dataset_dict.items():
            print(f"  {config_name}:")
            print(f"    train: {splits['train'].num_rows} samples")
            print(f"    test: {splits['test'].num_rows} samples")
            print(f"    Features: {list(splits['train'].features.keys())}")
        return 0

    # -------------------------
    # Upload to HuggingFace
    # -------------------------
    print(f"\n=== Uploading to {repo_id} ===")

    # Remove old parquet files (only for the first config to avoid conflicts)
    try:
        old_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        parquet_files = [f for f in old_files if f.endswith('.parquet')]
        for parquet_file in parquet_files:
            api.delete_file(path_in_repo=parquet_file, repo_id=repo_id, repo_type="dataset")
        print(f"Removed {len(parquet_files)} old parquet files")
    except Exception as e:
        print(f"Warning: Could not clean old files: {e}")

    # Push each config separately with proper config names
    for config_name, splits in split_dataset_dict.items():
        print(f"\nUploading {config_name} config...")
        config_dataset_dict = DatasetDict({
            'train': splits['train'],
            'test': splits['test']
        })
        config_dataset_dict.push_to_hub(
            repo_id,
            config_name=config_name,
            max_shard_size=f"{TARGET_SHARD_SIZE_MB}MB"
        )
        print(f"✅ {config_name}: {splits['train'].num_rows} train, {splits['test'].num_rows} test")

    print(f"\n✅ Successfully uploaded all configs to {repo_id}")
    print("\nDataset structure:")
    for config_name, splits in split_dataset_dict.items():
        print(f"  {config_name}:")
        print(f"    train: {splits['train'].num_rows} samples")
        print(f"    test: {splits['test'].num_rows} samples")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())