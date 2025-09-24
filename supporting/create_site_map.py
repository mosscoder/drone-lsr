#!/usr/bin/env python3
"""
Create a combined site map from RGB and terrain images.
Center crops both images to remove map UI elements and combines them vertically.
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def crop_for_ui_removal(image, left_frac=0.05, top_frac=0.15, right_frac=0.15, bottom_frac=0.15):
    """Asymmetrically crop image to remove map UI elements.

    Args:
        image: PIL Image
        left_frac: Fraction to crop from left (0.05 = 5%)
        top_frac: Fraction to crop from top (0.15 = 15%)
        right_frac: Fraction to crop from right (0.15 = 15%)
        bottom_frac: Fraction to crop from bottom (0.15 = 15%)
    """
    width, height = image.size

    # Calculate crop box (left, top, right, bottom)
    left = int(width * left_frac)
    top = int(height * top_frac)
    right = width - int(width * right_frac)
    bottom = height - int(height * bottom_frac)

    return image.crop((left, top, right, bottom))

def combine_images_horizontally(image1, image2, gap=5):
    """Combine two images horizontally with a white gap, resizing if necessary to match heights."""
    # Get dimensions
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Make sure heights match (resize to match the smaller height)
    if height1 != height2:
        target_height = min(height1, height2)
        if height1 > height2:
            # Resize image1 to match image2 height
            new_width1 = int(width1 * target_height / height1)
            image1 = image1.resize((new_width1, target_height), Image.LANCZOS)
            width1 = new_width1
        else:
            # Resize image2 to match image1 height
            new_width2 = int(width2 * target_height / height2)
            image2 = image2.resize((new_width2, target_height), Image.LANCZOS)
            width2 = new_width2
        height1 = height2 = target_height

    # Create combined image with gap
    combined_width = width1 + width2 + gap
    combined = Image.new('RGB', (combined_width, height1), color='white')

    # Paste images side by side with gap
    combined.paste(image1, (0, 0))
    combined.paste(image2, (width1 + gap, 0))

    return combined

def add_panel_labels(image, gap=5):
    """Add lowercase panel labels 'a' and 'b' in bold white text."""
    draw = ImageDraw.Draw(image)

    # Try to use a bold font, fallback to default
    try:
        # Try to load a system font
        font_size = 96  # Doubled from 48 to 96
        font = ImageFont.truetype("/System/Library/Fonts/Arial Bold.ttf", font_size)
    except (OSError, IOError):
        try:
            # Fallback to default PIL font
            font = ImageFont.load_default()
        except:
            font = None

    # Text properties
    text_color = 'white'
    padding = 20

    # Calculate panel positions (assuming equal width panels with gap)
    width, height = image.size
    panel1_width = (width - gap) // 2
    panel2_start = panel1_width + gap

    # Add label 'a' to first panel (top-left)
    draw.text((padding, padding), 'a', fill=text_color, font=font)

    # Add label 'b' to second panel (top-left)
    draw.text((panel2_start + padding, padding), 'b', fill=text_color, font=font)

    return image

def main():
    # Paths
    base_dir = Path(__file__).parent
    raw_dir = base_dir / "raw_images"
    processed_dir = base_dir / "processed"

    rgb_path = raw_dir / "rgb_site.png"
    terrain_path = raw_dir / "terrain_site.png"
    output_path = processed_dir / "site_map.png"

    # Check input files exist
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    if not terrain_path.exists():
        raise FileNotFoundError(f"Terrain image not found: {terrain_path}")

    print(f"Loading images...")
    print(f"  RGB: {rgb_path}")
    print(f"  Terrain: {terrain_path}")

    # Load images
    rgb_img = Image.open(rgb_path)
    terrain_img = Image.open(terrain_path)

    print(f"Original sizes: RGB {rgb_img.size}, Terrain {terrain_img.size}")

    # Asymmetrically crop both images to remove UI elements
    # Preserve left side, remove more from top/bottom
    rgb_cropped = crop_for_ui_removal(rgb_img, left_frac=0.05, top_frac=0.15, right_frac=0.15, bottom_frac=0.15)
    terrain_cropped = crop_for_ui_removal(terrain_img, left_frac=0.05, top_frac=0.15, right_frac=0.15, bottom_frac=0.15)

    print(f"Cropped sizes: RGB {rgb_cropped.size}, Terrain {terrain_cropped.size}")

    # Combine horizontally with gap (RGB on left, terrain on right)
    gap = 5
    combined = combine_images_horizontally(rgb_cropped, terrain_cropped, gap=gap)

    print(f"Combined size: {combined.size}")

    # Add panel labels
    combined = add_panel_labels(combined, gap=gap)

    # Create output directory
    processed_dir.mkdir(exist_ok=True)

    # Save result
    combined.save(output_path, 'PNG', optimize=True)
    print(f"Saved combined site map with labels: {output_path}")

if __name__ == "__main__":
    main()