#!/usr/bin/env python3
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch
import numpy as np
from datasets import Dataset, Features, Image, Value
from datasets.features import Sequence, Array2D
from huggingface_hub import HfApi
from transformers import AutoModel, AutoImageProcessor

# -------------------------
# Config
# -------------------------
hf_org = 'mpg-ranch'
hf_repo = 'light-stable-semantics'
TILES_DIR = Path('data/raster/tiles')
TARGET_SHARD_SIZE_MB = 500
TIMES = [1000, 1200, 1500]   # t0, t1, t2
MODEL_ID = "facebook/dinov3-vitl16-pretrain-sat493m"
BATCH_SIZE = 16
WRITER_BATCH_SIZE = 2000

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------
# Model (load once)
# -------------------------
if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

N_REG = model.config.num_register_tokens
DIM = model.config.hidden_size

img_size = getattr(model.config, "image_size")
ps = model.config.patch_size

if isinstance(img_size, (tuple, list)):
    h = img_size[0] // ps
    w = img_size[1] // ps
else:
    h = w = img_size // ps
N_PATCH = h * w

@torch.no_grad()
def encode_images(pil_images):
    pil_images = [im.convert("RGB") if hasattr(im, "mode") and im.mode != "RGB" else im for im in pil_images]
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    outputs = model(**inputs)
    tokens = outputs.last_hidden_state              # [B, 1+R+N, D]
    cls = tokens[:, 0]                              # [B, D]
    patches = tokens[:, 1 + N_REG:]                 # [B, N, D]
    return cls.cpu().numpy().astype(np.float32), patches.cpu().numpy().astype(np.float32)

# -------------------------
# Scan tiles
# -------------------------
def scan_tiles():
    logging.info("Scanning tiles directory...")
    tiles_by_location = defaultdict(dict)
    for tile_path in TILES_DIR.glob('*.png'):
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

# -------------------------
# Map fn: add embeddings
# -------------------------
def encode_batch(batch):
    cls0, patch0 = encode_images(batch['image_t0'])
    cls1, patch1 = encode_images(batch['image_t1'])
    cls2, patch2 = encode_images(batch['image_t2'])
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
    repo_id = f"{hf_org}/{hf_repo}"
    api = HfApi()

    # Scan and organize tiles
    tiles_dict = scan_tiles()
    if not tiles_dict:
        logging.error("No complete tile sets found")
        return 1

    # Build records
    records = []
    for location_id, time_paths in tiles_dict.items():
        records.append({
            'image_t0': str(time_paths[1000]),
            'image_t1': str(time_paths[1200]),
            'image_t2': str(time_paths[1500]),
            'idx': location_id
        })
    df = pd.DataFrame(records)

    features = Features({
        'idx': Value('string'),
        'image_t0': Image(),
        'image_t1': Image(),
        'image_t2': Image(),
        'cls_t0': Sequence(Value('float32'), length=DIM),
        'cls_t1': Sequence(Value('float32'), length=DIM),
        'cls_t2': Sequence(Value('float32'), length=DIM),
        'patch_t0': Array2D((N_PATCH, DIM), dtype='float32'),
        'patch_t1': Array2D((N_PATCH, DIM), dtype='float32'),
        'patch_t2': Array2D((N_PATCH, DIM), dtype='float32'),
    })

    dataset = Dataset.from_pandas(df, features=features, preserve_index=False)
    dataset = dataset.map(
        encode_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        features=features,
        writer_batch_size=WRITER_BATCH_SIZE,
    )

    old_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    parquet_files = [f for f in old_files if f.endswith('.parquet')]
    for parquet_file in parquet_files:
        api.delete_file(path_in_repo=parquet_file, repo_id=repo_id, repo_type="dataset")

    # Push to hub
    dataset.push_to_hub(repo_id, max_shard_size=f"{TARGET_SHARD_SIZE_MB}MB")
    logging.info(f"Uploaded dataset with CLS + patch tokens → {repo_id}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())