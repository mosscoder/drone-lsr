
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

times = [1000, 1200, 1500]
url = 'https://storage.googleapis.com/mpg-aerial-survey/surveys/241107_lowerpartridge_{TIME}/processing/dronedeploy/241107_lowerpartridge_{TIME}-visible.tif'

OUTPUT_DIR = Path('data/raster/tiles')
TILE_SIZE = 1024
N_JOBS = max(1, os.cpu_count() - 1)


def get_orthomosaic_info(time):
    """Get information about an orthomosaic."""
    ortho_url = url.format(TIME=time)
    try:
        with rasterio.open(ortho_url) as src:
            info = {
                'time': time,
                'url': ortho_url,
                'bounds': src.bounds,
                'crs': src.crs,
                'resolution': (src.res[0], src.res[1]),
                'width': src.width,
                'height': src.height
            }
            return info
    except Exception as e:
        logging.error(f"Error reading orthomosaic at {time}: {e}")
        raise


def find_reference_orthomosaic():
    """Find the highest resolution orthomosaic to use as reference."""
    logging.info("Loading orthomosaic information...")
    
    ortho_infos = []
    for time in times:
        info = get_orthomosaic_info(time)
        ortho_infos.append(info)
        logging.info(f"Time {time}: Resolution = {info['resolution']}")
    
    # Find highest resolution (smallest pixel size)
    reference = min(ortho_infos, key=lambda x: x['resolution'][0])
    logging.info(f"Selected reference orthomosaic: Time {reference['time']} with resolution {reference['resolution']}")
    
    return reference, ortho_infos


def create_tile_grid(reference):
    """Create a grid of tiles based on reference orthomosaic."""
    bounds = reference['bounds']
    res_x, res_y = reference['resolution']
    
    # Calculate tile size in CRS units
    tile_width = TILE_SIZE * res_x
    tile_height = TILE_SIZE * res_y
    
    # Calculate number of tiles
    width = bounds.right - bounds.left
    height = bounds.top - bounds.bottom
    
    n_cols = int(np.ceil(width / tile_width))
    n_rows = int(np.ceil(height / tile_height))
    
    logging.info(f"Creating tile grid: {n_rows} rows x {n_cols} columns")
    
    tiles = []
    for row in range(n_rows):
        for col in range(n_cols):
            # Calculate tile bounds
            left = bounds.left + col * tile_width
            right = min(left + tile_width, bounds.right)
            top = bounds.top - row * tile_height
            bottom = max(top - tile_height, bounds.bottom)
            
            tiles.append({
                'row': row,
                'col': col,
                'bounds': (left, bottom, right, top),
                'crs': reference['crs']
            })
    
    return tiles


def check_tile_has_data(src, window):
    """Check if a tile has complete data coverage by examining the alpha channel."""
    # Read alpha channel (band 4)
    alpha = src.read(4, window=window)
    
    # Check if all pixels have data (alpha > 0) - skip tile if any pixels are no-data
    return np.all(alpha > 0)


def process_tile(args):
    """Process a single tile across all time points."""
    tile, ortho_infos = args
    row, col = tile['row'], tile['col']
    tile_bounds = tile['bounds']
    
    # First, check if all time points have data for this tile
    has_data = {}
    for ortho in ortho_infos:
        ortho_url = ortho['url']
        try:
            with rasterio.open(ortho_url) as src:
                # Calculate window for this tile
                window = rasterio.windows.from_bounds(
                    *tile_bounds, 
                    transform=src.transform
                )
                
                # Check if tile has data
                has_data[ortho['time']] = check_tile_has_data(src, window)
                
        except Exception as e:
            logging.error(f"Error checking tile {row}_{col} at time {ortho['time']}: {e}")
            return f"ERROR: {row}_{col}"
    
    # Skip if any time point has no data
    if not all(has_data.values()):
        missing_times = [t for t, has in has_data.items() if not has]
        logging.debug(f"Skipping tile {row}_{col} - no data at times: {missing_times}")
        return f"SKIP: {row}_{col}"
    
    # Process each time point
    for ortho in ortho_infos:
        ortho_url = ortho['url']
        time = ortho['time']
        output_path = OUTPUT_DIR / f"{row}_{col}_{time}.png"
        
        try:
            with rasterio.open(ortho_url) as src:
                # Calculate window for this tile
                window = rasterio.windows.from_bounds(
                    *tile_bounds, 
                    transform=src.transform
                )
                
                # Read data
                data = src.read(window=window)
                
                # Calculate transform for output tile
                transform = from_bounds(*tile_bounds, TILE_SIZE, TILE_SIZE)
                
                # If data isn't exactly TILE_SIZE x TILE_SIZE, resample
                if data.shape[1] != TILE_SIZE or data.shape[2] != TILE_SIZE:
                    # Create destination array
                    resampled = np.zeros((4, TILE_SIZE, TILE_SIZE), dtype=data.dtype)
                    
                    # Resample each band
                    for band_idx in range(4):
                        reproject(
                            source=data[band_idx],
                            destination=resampled[band_idx],
                            src_transform=src.window_transform(window),
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=src.crs,
                            resampling=Resampling.bilinear
                        )
                    data = resampled
                
                # Convert data for PNG output
                # Rasterio format: (bands, height, width) -> PIL format: (height, width, bands)
                data_pil = np.transpose(data, (1, 2, 0))
                
                # Ensure data is uint8 for PNG (convert if needed)
                if data_pil.dtype != np.uint8:
                    # Assume data is in 0-255 range, convert to uint8
                    data_pil = data_pil.astype(np.uint8)
                
                # Create PIL Image from array (RGBA format)
                img = Image.fromarray(data_pil, mode='RGBA')
                
                # Save as PNG
                img.save(output_path, 'PNG')
                    
        except Exception as e:
            logging.error(f"Error processing tile {row}_{col} at time {time}: {e}")
            return f"ERROR: {row}_{col}_{time}"
    
    return f"SUCCESS: {row}_{col}"


def main():
    """Main function to orchestrate the tiling process."""
    logging.info(f"Starting orthomosaic tiling with {N_JOBS} workers")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find reference orthomosaic
    reference, ortho_infos = find_reference_orthomosaic()
    
    # Create tile grid
    tiles = create_tile_grid(reference)
    logging.info(f"Generated {len(tiles)} tiles")
    
    # Prepare arguments for parallel processing
    tile_args = [(tile, ortho_infos) for tile in tiles]
    
    # Process tiles in parallel
    logging.info("Starting tile processing...")
    results = []
    
    with Pool(N_JOBS) as pool:
        with tqdm(total=len(tiles), desc="Processing tiles") as pbar:
            for result in pool.imap(process_tile, tile_args):
                results.append(result)
                pbar.update(1)
    
    # Summarize results
    successful = [r for r in results if r.startswith("SUCCESS")]
    skipped = [r for r in results if r.startswith("SKIP")]
    errors = [r for r in results if r.startswith("ERROR")]
    
    logging.info(f"\nTiling complete!")
    logging.info(f"  Successful tiles: {len(successful)}")
    logging.info(f"  Skipped tiles (no data): {len(skipped)}")
    logging.info(f"  Error tiles: {len(errors)}")
    
    if errors:
        logging.warning(f"Errors encountered in {len(errors)} tiles:")
        for error in errors[:10]:  # Show first 10 errors
            logging.warning(f"  {error}")
        if len(errors) > 10:
            logging.warning(f"  ... and {len(errors) - 10} more errors")


if __name__ == "__main__":
    main()