#!/usr/bin/env python3
"""
Create a tile grid showing 5 randomly sampled tiles across time and canopy height.

Grid layout (4 rows × 5 columns):
- Row 1: Morning (t0) RGB images
- Row 2: Noon (t1) RGB images
- Row 3: Afternoon (t2) RGB images
- Row 4: Canopy height visualizations
"""

import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datasets import load_dataset

def set_seed(seed=42):
    """Set random seed for reproducible sampling."""
    random.seed(seed)
    np.random.seed(seed)

def get_specific_tiles(dataset, tile_ids=['23_19', '12_26', '10_24', '4_19', '16_16']):
    """Get specific tiles by their IDs."""
    # Create a mapping from idx to data
    idx_to_data = {item['idx']: item for item in dataset}

    # Get the specific tiles
    sampled_tiles = []
    for tile_id in tile_ids:
        if tile_id in idx_to_data:
            sampled_tiles.append(idx_to_data[tile_id])
        else:
            print(f"Warning: Tile ID '{tile_id}' not found in dataset")

    return sampled_tiles

def canopy_height_to_image(canopy_data, colormap='viridis', vmin=None, vmax=None):
    """Convert canopy height data to RGB image using colormap with global normalization."""
    # Convert to numpy array if needed
    if not isinstance(canopy_data, np.ndarray):
        canopy_data = np.array(canopy_data)

    # Handle the case where data might be nested
    if canopy_data.ndim > 2:
        canopy_data = canopy_data.squeeze()

    # Use global vmin/vmax if provided, otherwise local min/max
    if vmin is None:
        vmin = canopy_data.min()
    if vmax is None:
        vmax = canopy_data.max()

    # Normalize to 0-1 for colormap using global range
    if vmax > vmin:
        normalized = np.clip((canopy_data - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(canopy_data)

    # Apply colormap
    cmap = plt.cm.get_cmap(colormap)
    colored = cmap(normalized)

    # Convert to RGB image (remove alpha channel and scale to 0-255)
    rgb_image = (colored[:, :, :3] * 255).astype(np.uint8)

    return Image.fromarray(rgb_image)

def resize_image(image, target_size=(200, 200)):
    """Resize image maintaining aspect ratio and center crop if needed."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image.resize(target_size, Image.LANCZOS)

def create_tile_grid(tiles, cell_size=(200, 200), gap=5):
    """Create a 4×5 grid of tiles showing RGB at 3 times + canopy height."""
    n_tiles = len(tiles)
    n_rows = 4  # t0, t1, t2, canopy_height

    # Calculate global min/max for canopy height normalization
    all_canopy_data = []
    for tile in tiles:
        canopy_data = np.array(tile['canopy_height'])
        if canopy_data.ndim > 2:
            canopy_data = canopy_data.squeeze()
        all_canopy_data.append(canopy_data)

    all_canopy_data = np.concatenate([data.flatten() for data in all_canopy_data])
    global_vmin = all_canopy_data.min()
    global_vmax = all_canopy_data.max()

    print(f"Global canopy height range: {global_vmin:.1f} - {global_vmax:.1f} cm")

    # Calculate grid dimensions
    grid_width = n_tiles * cell_size[0] + (n_tiles - 1) * gap
    grid_height = n_rows * cell_size[1] + (n_rows - 1) * gap

    # Create white background
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')

    # Row labels
    row_labels = ['Morning (10:00)', 'Noon (12:00)', 'Afternoon (15:00)', 'Canopy Height']

    for col, tile in enumerate(tiles):
        # Calculate x position for this column
        x_pos = col * (cell_size[0] + gap)

        # Row 0: Morning (t0)
        morning_img = resize_image(tile['image_t0'], cell_size)
        y_pos = 0
        grid_image.paste(morning_img, (x_pos, y_pos))

        # Row 1: Noon (t1)
        noon_img = resize_image(tile['image_t1'], cell_size)
        y_pos = cell_size[1] + gap
        grid_image.paste(noon_img, (x_pos, y_pos))

        # Row 2: Afternoon (t2)
        afternoon_img = resize_image(tile['image_t2'], cell_size)
        y_pos = 2 * (cell_size[1] + gap)
        grid_image.paste(afternoon_img, (x_pos, y_pos))

        # Row 3: Canopy Height (with global normalization)
        canopy_img = canopy_height_to_image(tile['canopy_height'], colormap='viridis',
                                          vmin=global_vmin, vmax=global_vmax)
        canopy_img = resize_image(canopy_img, cell_size)
        y_pos = 3 * (cell_size[1] + gap)
        grid_image.paste(canopy_img, (x_pos, y_pos))

    return grid_image, row_labels

def add_labels(image, row_labels, cell_size=(200, 200), gap=5):
    """Add row labels to the grid (no column labels)."""
    # Create a larger image with space for row labels only
    label_width = 160  # Increased width to prevent text overlap with tiles

    new_width = image.width + label_width
    new_height = image.height

    labeled_image = Image.new('RGB', (new_width, new_height), color='white')

    # Paste the original grid with offset for labels
    labeled_image.paste(image, (label_width, 0))

    draw = ImageDraw.Draw(labeled_image)

    # Try to load a font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Add row labels with better spacing
    for row, label in enumerate(row_labels):
        y_pos = row * (cell_size[1] + gap) + cell_size[1] // 2
        x_pos = 20  # More padding from left edge
        # Center vertically
        bbox = draw.textbbox((0, 0), label, font=font)
        text_height = bbox[3] - bbox[1]
        draw.text((x_pos, y_pos - text_height // 2), label, fill='black', font=font)

    return labeled_image

def main():
    # Configuration
    cell_size = (200, 200)
    gap = 5

    # Hardcoded tile IDs (4 tiles total)
    tile_ids = ['22_19', '23_19', '10_24', '16_16']

    # Paths
    base_dir = Path(__file__).parent
    processed_dir = base_dir / "processed"
    output_path = processed_dir / "tile_grid.png"

    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("mpg-ranch/light-stable-semantics", "default", split="train")

    print(f"Getting specific tiles: {tile_ids}")
    sampled_tiles = get_specific_tiles(dataset, tile_ids=tile_ids)

    print(f"Found {len(sampled_tiles)} tiles")

    print("Creating tile grid...")
    grid_image, row_labels = create_tile_grid(sampled_tiles, cell_size=cell_size, gap=gap)

    print("Adding row labels...")
    final_image = add_labels(grid_image, row_labels, cell_size=cell_size, gap=gap)

    # Create output directory
    processed_dir.mkdir(exist_ok=True)

    # Save result
    final_image.save(output_path, 'PNG', optimize=True)
    print(f"Saved tile grid: {output_path}")
    print(f"Grid dimensions: {final_image.size}")

if __name__ == "__main__":
    main()