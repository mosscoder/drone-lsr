---
dataset_info:
  features:
    - name: image_t0
      dtype: image
    - name: image_t1
      dtype: image
    - name: image_t2
      dtype: image
    - name: idx
      dtype: string
configs:
  - config_name: default
license: cc-by-4.0
task_categories:
- feature-extraction
tags:
- remote-sensing
- aerial-imagery
- orthomosaic
- lighting-invariance
- semantic-stability
- vision-encoder
- time-series
size_categories:
- 1K<n<10K
language:
- en
pretty_name: Light Stable Semantics
---

# Light Stable Semantics Dataset

## Dataset Description

This dataset contains aerial orthomosaic tiles captured at three different times of day (10:00, 12:00, and 15:00) to develop vision encoders that are semantically stable under varying lighting conditions. The dataset is designed for training computer vision models that can maintain consistent feature representations despite changes in illumination.

### Dataset Summary

- **Purpose**: Training light-stable semantic vision encoders
- **Data Type**: Aerial orthomosaic tiles (RGBA, 1024x1024 pixels)
- **Time Points**: 3 captures per location (morning, noon, afternoon)
- **Coverage**: Lower Partridge area aerial survey
- **Date**: November 7, 2024 (241107)
- **Location**: MPG Ranch, Montana, USA

### Data Structure

Each record contains:
- `image_t0`: Morning image (10:00, time=1000)
- `image_t1`: Noon image (12:00, time=1200)  
- `image_t2`: Afternoon image (15:00, time=1500)
- `idx`: Tile identifier in format `{ROW}_{COL}`

### Data Collection

The orthomosaics were captured using drone surveys with identical geographic bounds but at different times of day to capture varying lighting conditions. All tiles:
- Are 1024x1024 pixels of 1.2cm resolution
- Maintain spatial alignment across time points
- Use consistent geographic coordinates

### Use Cases

This dataset is intended for:
- Training vision encoders robust to lighting changes
- Semantic stability research in computer vision
- Time-invariant feature learning
- Remote sensing applications requiring lighting robustness

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
  note={Aerial orthomosaic tiles captured at multiple times of day for light-stable semantic vision encoder training},
  location={MPG Ranch, Montana, USA},
  survey_date={2024-11-07},
  organization={MPG Ranch}
}
```

## Licensing

This dataset is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. 

Under the following terms:
- **Attribution** — You must give appropriate credit to MPG Ranch, provide a link to the license, and indicate if changes were made.