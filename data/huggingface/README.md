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
- representation-stability
- vision-encoder
- time-series
- dinov2
- dinov3
- embeddings
- multi-config
pretty_name: Light Stable Representations
size_categories:
- n<1K
---

# Light Stable Representations Dataset

## Dataset Description

This dataset contains aerial orthomosaic tiles captured at three different times of day (10:00, 12:00, and 15:00). The dataset is organized into three configurations: `default` (raw images + canopy height), `dinov2_base` (DINOv2 embeddings), and `dinov3_sat` (DINOv3 embeddings). All configurations share consistent train/test splits with matching tile identifiers for cross-referencing. The dataset is designed for training vision encoders that maintain consistent feature representations despite changes in illumination, with applications in remote sensing and environmental monitoring.

## Dataset Configurations

The dataset is organized into three configurations, each serving different research needs:

### Configuration: `default`
Raw imagery and environmental data for direct analysis:

| Feature | Type | Shape | Description |
|---------|------|--------|-------------|
| `idx` | string | - | Tile identifier in format `{ROW}_{COL}` for geographic referencing |
| `image_t0` | Image | 1024×1024×3 | Morning capture at 10:00 AM (time=1000) |
| `image_t1` | Image | 1024×1024×3 | Noon capture at 12:00 PM (time=1200) |
| `image_t2` | Image | 1024×1024×3 | Afternoon capture at 3:00 PM (time=1500) |
| `canopy_height` | int32 | [1024, 1024] | Canopy height grid in centimeters from canopy height model |

### Configuration: `dinov2_base`
Pre-computed DINOv2 Base (ViT-B/14) embeddings:

| Feature | Type | Shape | Description |
|---------|------|--------|-------------|
| `idx` | string | - | Tile identifier matching other configurations |
| `cls_t0` | float32 | [768] | DINOv2 CLS token (global features) for morning image |
| `cls_t1` | float32 | [768] | DINOv2 CLS token (global features) for noon image |
| `cls_t2` | float32 | [768] | DINOv2 CLS token (global features) for afternoon image |
| `patch_t0` | float32 | [256, 768] | DINOv2 patch tokens (16×16 spatial grid) for morning image |
| `patch_t1` | float32 | [256, 768] | DINOv2 patch tokens (16×16 spatial grid) for noon image |
| `patch_t2` | float32 | [256, 768] | DINOv2 patch tokens (16×16 spatial grid) for afternoon image |

### Configuration: `dinov3_sat`
Pre-computed DINOv3 Large (ViT-L/16) embeddings with satellite pretraining:

| Feature | Type | Shape | Description |
|---------|------|--------|-------------|
| `idx` | string | - | Tile identifier matching other configurations |
| `cls_t0` | float32 | [1024] | DINOv3 CLS token (global features) for morning image |
| `cls_t1` | float32 | [1024] | DINOv3 CLS token (global features) for noon image |
| `cls_t2` | float32 | [1024] | DINOv3 CLS token (global features) for afternoon image |
| `patch_t0` | float32 | [196, 1024] | DINOv3 patch tokens (14×14 spatial grid) for morning image |
| `patch_t1` | float32 | [196, 1024] | DINOv3 patch tokens (14×14 spatial grid) for noon image |
| `patch_t2` | float32 | [196, 1024] | DINOv3 patch tokens (14×14 spatial grid) for afternoon image |

**Notes:**
- Canopy height values represent centimeters above ground; missing data is encoded as `-2147483648`
- All configurations use consistent 80%/20% train/test splits with matching `idx` values
- Patch tokens represent spatial features in different grid resolutions: 16×16 (DINOv2) vs 14×14 (DINOv3)

## Usage Example

```python
from datasets import load_dataset

# Load specific configurations
dataset_default = load_dataset("mpg-ranch/drone-lsr", "default")
dataset_dinov2 = load_dataset("mpg-ranch/drone-lsr", "dinov2_base")
dataset_dinov3 = load_dataset("mpg-ranch/drone-lsr", "dinov3_sat")

# Access raw imagery and canopy height
sample_default = dataset_default['train'][0]
morning_image = sample_default['image_t0']      # RGB image
noon_image = sample_default['image_t1']         # RGB image
afternoon_image = sample_default['image_t2']    # RGB image
canopy_height = sample_default['canopy_height'] # Height grid in cm
tile_id = sample_default['idx']                 # Geographic identifier

# Access DINOv2 embeddings (same tile via matching idx)
sample_dinov2 = dataset_dinov2['train'][0]
dinov2_cls_morning = sample_dinov2['cls_t0']     # Global features (768-dim)
dinov2_patches_morning = sample_dinov2['patch_t0'] # Spatial features (256×768)

# Access DINOv3 embeddings (same tile via matching idx)
sample_dinov3 = dataset_dinov3['train'][0]
dinov3_cls_morning = sample_dinov3['cls_t0']     # Global features (1024-dim)
dinov3_patches_morning = sample_dinov3['patch_t0'] # Spatial features (196×1024)

# Verify consistent tile identifiers across configurations
assert sample_default['idx'] == sample_dinov2['idx'] == sample_dinov3['idx']

# Access test sets for evaluation
test_default = dataset_default['test'][0]
test_dinov2 = dataset_dinov2['test'][0]
test_dinov3 = dataset_dinov3['test'][0]
```

## Pre-computed Embeddings

The dataset includes pre-computed embeddings from two state-of-the-art vision transformers:

### DINOv2 Base (`facebook/dinov2-base`)
- **Architecture**: Vision Transformer Base with 14×14 patch size
- **CLS Tokens**: 768-dimensional global feature vectors capturing scene-level representations
- **Patch Tokens**: 256×768 arrays (16×16 spatial grid) encoding local features
- **Training**: Self-supervised learning on natural images

### DINOv3 Large (`facebook/dinov3-vitl16-pretrain-sat493m`)
- **Architecture**: Vision Transformer Large with 16×16 patch size
- **CLS Tokens**: 1024-dimensional global feature vectors capturing scene-level representations
- **Patch Tokens**: 196×1024 arrays (14×14 spatial grid) encoding local features
- **Training**: Self-supervised learning with satellite imagery pretraining

**Purpose**: Enable efficient training and analysis without requiring on-the-fly feature extraction, while providing comparison between natural image and satellite-pretrained models.

## Dataset Information

- **Location**: Lower Partridge Alley, MPG Ranch, Montana, USA
- **Survey Date**: November 7, 2024
- **Coverage**: 620 complete tile sets (80% train / 20% test split via seeded random sampling)
- **Resolution**: 1024×1024 pixels at 1.2cm ground resolution
- **Total Size**: ~6.4GB of image data plus embeddings
- **Quality Control**: Tiles with transient objects, such as vehicles, were excluded from the dataset. RGB imagery and canopy rasters are removed together to keep modalities aligned.

## Use Cases

This dataset is intended for:
- Developing vision encoders robust to lighting variations
- Representation stability research in computer vision
- Time-invariant feature learning
- Remote sensing applications requiring lighting robustness
- Comparative analysis of illumination effects on vision model features

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{mpg_ranch_light_stable_semantics_2024,
  title={Light Stable Representations Dataset},
  author={Kyle Doherty and Erik Samose and Max Gurinas and Brandon Trabucco and Ruslan Salakhutdinov},
  year={2024},
  month={November},
  url={https://huggingface.co/datasets/mpg-ranch/drone-lsr},
  publisher={Hugging Face},
  note={Aerial orthomosaic tiles with DINOv2 and DINOv3 embeddings for light-stable representation vision encoder training},
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

