---
license: cc-by-4.0
task_categories:
- feature-extraction
- image-to-image
language:
- en
tags:
- remote-sensing
- aerial-imagery
- orthomosaic
- lighting-invariance
- semantic-stability
- vision-encoder
- time-series
- dinov3
- embeddings
pretty_name: Light Stable Semantics
size_categories:
- n<1K
---

# Light Stable Semantics Dataset

## Dataset Description

This dataset contains aerial orthomosaic tiles captured at three different times of day (10:00, 12:00, and 15:00). Each tile includes the original RGB images, a co-registered canopy height model (CHM) raster, and pre-computed DINOv3 embeddings extracted using `facebook/dinov3-vitl16-pretrain-sat493m`. The dataset is designed for adapting vision encoders that can maintain consistent feature representations despite changes in illumination, with applications in remote sensing and environmental monitoring.

## Dataset Features

Each record in the dataset contains the following features:

| Feature | Type | Shape | Description |
|---------|------|--------|-------------|
| `idx` | string | - | Tile identifier in format `{ROW}_{COL}` for geographic referencing |
| `image_t0` | Image | 1024×1024×3 | Morning capture at 10:00 AM (time=1000) |
| `image_t1` | Image | 1024×1024×3 | Noon capture at 12:00 PM (time=1200) |
| `image_t2` | Image | 1024×1024×3 | Afternoon capture at 3:00 PM (time=1500) |
| `cls_t0` | float32 | [1024] | DINOv3 CLS token (global features) for morning image |
| `cls_t1` | float32 | [1024] | DINOv3 CLS token (global features) for noon image |
| `cls_t2` | float32 | [1024] | DINOv3 CLS token (global features) for afternoon image |
| `patch_t0` | float32 | [196, 1024] | DINOv3 patch tokens (spatial features) for morning image |
| `patch_t1` | float32 | [196, 1024] | DINOv3 patch tokens (spatial features) for noon image |
| `patch_t2` | float32 | [196, 1024] | DINOv3 patch tokens (spatial features) for afternoon image |
| `canopy_height` | int32 | [1024, 1024] | Canopy height grid in centimetres derived from the canopy height model |

The canopy height layer is reprojected to align with the RGB tiles and multiplied by 100 before casting to `int32`, so each value represents centimetres above ground. Missing data is encoded with `-2147483648` (the minimum 32-bit integer).

## Usage Example

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("mpg-ranch/light-stable-semantics")

# Access a single record
sample = dataset['train'][0]

# Images for the three time points
morning_image = sample['image_t0']
noon_image = sample['image_t1'] 
afternoon_image = sample['image_t2']

# Pre-computed DINOv3 embeddings
morning_cls = sample['cls_t0']     # Global features (1024-dim)
noon_cls = sample['cls_t1']        # Global features (1024-dim)
afternoon_cls = sample['cls_t2']   # Global features (1024-dim)

morning_patches = sample['patch_t0']  # Spatial features (196×1024)
noon_patches = sample['patch_t1']     # Spatial features (196×1024)
afternoon_patches = sample['patch_t2'] # Spatial features (196×1024)

# Tile location identifier
tile_id = sample['idx']  # Format: "{ROW}_{COL} of tiles within the original orthomosaic"

# Co-registered canopy height (centimetres stored as int32)
canopy_cm = sample['canopy_height']
```

## Pre-computed Embeddings

The dataset includes pre-computed embeddings extracted using the **facebook/dinov3-vitl16-pretrain-sat493m** model:

- **CLS Tokens**: 1024-dimensional global feature vectors that capture scene-level semantics
- **Patch Tokens**: 196×1024 arrays encoding spatial relationships and local features
- **Purpose**: Enable efficient training and analysis without requiring on-the-fly feature extraction
- **Model Details**: DINOv3 Vision Transformer Large (16×16 patches) pre-trained on satellite imagery

## Dataset Information

- **Location**: Lower Partridge Alley, MPG Ranch, Montana, USA
- **Survey Date**: November 7, 2024
- **Coverage**: 620 complete tile sets
- **Resolution**: 1024×1024 pixels at 1.2cm ground resolution
- **Total Size**: ~6.4GB of image data plus embeddings
- **Quality Control**: Tiles with transient objects, such as vehicles, were excluded from the dataset. RGB imagery and canopy rasters are removed together to keep modalities aligned.

## Use Cases

This dataset is intended for:
- Developing vision encoders robust to lighting variations
- Semantic stability research in computer vision
- Time-invariant feature learning
- Remote sensing applications requiring lighting robustness
- Comparative analysis of illumination effects on vision model features

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{mpg_ranch_light_stable_semantics_2024,
  title={Light Stable Semantics Dataset},
  author={Kyle Doherty and Erik Samose and Max Gurinas and Brandon Trabucco and Ruslan Salakhutdinov},
  year={2024},
  month={November},
  url={https://huggingface.co/datasets/mpg-ranch/light-stable-semantics},
  publisher={Hugging Face},
  note={Aerial orthomosaic tiles with DINOv3 embeddings for light-stable semantic vision encoder training},
  location={MPG Ranch, Montana, USA},
  survey_date={2024-11-07},
  organization={MPG Ranch}
}
```

## License

This dataset is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

**Attribution Requirements:**
- You must give appropriate credit to MPG Ranch
- Provide a link to the license
- Indicate if changes were made to the dataset

##Updates
Placeholder
