# Light Stable Semantics Dataset

<div align="center">
<img src="figures/example_scene.gif" alt="Example Scene" width="50%">
</div>

## Project Goals

This project aims to develop vision encoders that maintain semantic stability under varying lighting conditions by creating a specialized dataset from drone orthomosaics captured at three different times of day (10:00, 12:00, and 15:00) over the same geographic area at MPG Ranch, Montana, USA. By analyzing how vision model features change across these time points while the underlying scene semantics remain constant, we can identify which features are most sensitive to illumination changes versus those that capture lighting-invariant semantic information. The dataset enables researchers to train and evaluate computer vision models for improved robustness to natural lighting variations, ultimately advancing applications in remote sensing, environmental monitoring, and autonomous navigation where consistent scene understanding across different lighting conditions is critical.

## Dataset Structure

The dataset consists of 1024x1024 pixel tiles extracted from three orthomosaics, with each tile representing the same geographic location captured under different lighting conditions. Each tile includes both the original images and pre-computed DINOv3 embeddings for efficient feature analysis.

**Dataset Schema:**
- **Images**: RGB tiles for three time points (10:00, 12:00, 15:00)
- **Global Features**: DINOv3 CLS tokens (1024-length vectors) capturing scene-level semantics
- **Spatial Features**: DINOv3 patch tokens (196×1024 arrays) encoding spatial relationships
- **Identifiers**: Location-based tile IDs for geographic referencing

The data is available on Hugging Face Hub at [mpg-ranch/light-stable-semantics](https://huggingface.co/datasets/mpg-ranch/light-stable-semantics).

## Processing Pipeline

The dataset creation follows a rigorous 4-step preprocessing pipeline:

1. **00_tile_orthos.py**: Tiles orthomosaics into 1024×1024 pixel tiles with quality control
2. **01_exclude_temporal_anomalies.py**: Removes tiles with shadows, clouds, or other temporal artifacts
3. **02_push_docs_to_hf.py**: Uploads documentation and metadata to Hugging Face Hub
4. **03_push_imgs_hf.py**: Extracts DINOv3 embeddings and uploads enhanced dataset

This pipeline ensures high-quality, semantically consistent tiles suitable for training lighting-robust vision models.

## Repository Contents

- `hf_preprocessing/`: Complete processing pipeline from orthomosaics to Hugging Face dataset
- `experiments/`: Analysis scripts for evaluating lighting stability in vision features  
- `data/`: Local storage for processed tiles, exclusion lists, and intermediate outputs