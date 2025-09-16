import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EXCLUSION_LIST_PATH = Path('data/tabular/transient_phenomena.txt')
RGB_TILE_DIR = Path('data/raster/tiles/rgb')
EXCLUDED_RGB_DIR = Path('data/raster/excluded_tiles/rgb')
CANOPY_TILE_DIR = Path('data/raster/tiles/chm')
EXCLUDED_CANOPY_DIR = Path('data/raster/excluded_tiles/chm')
TIMES = ['1000', '1200', '1500']


def read_exclusion_list():
    """Read the list of tiles to exclude from the text file."""
    if not EXCLUSION_LIST_PATH.exists():
        raise FileNotFoundError(f"Exclusion list not found at {EXCLUSION_LIST_PATH}")
    
    excluded_tiles = []
    with open(EXCLUSION_LIST_PATH, 'r') as f:
        for line in f:
            tile_id = line.strip()
            if tile_id:  # Skip empty lines
                excluded_tiles.append(tile_id)
    
    logging.info(f"Read {len(excluded_tiles)} tile IDs to exclude")
    return excluded_tiles


def move_tile_set(tile_id, move_canopy):
    """Move all artefacts for a given tile to the excluded directories."""
    result = {
        'tile_id': tile_id,
        'moved_png': [],
        'missing_png': [],
        'moved_canopy': [],
        'missing_canopy': [],
    }

    for time in TIMES:
        filename = f"{tile_id}_{time}.png"
        src_path = RGB_TILE_DIR / filename
        dst_path = EXCLUDED_RGB_DIR / filename

        if src_path.exists():
            shutil.move(str(src_path), str(dst_path))
            result['moved_png'].append(filename)
        else:
            result['missing_png'].append(filename)

    if move_canopy:
        canopy_name = f"{tile_id}.tif"
        canopy_src = CANOPY_TILE_DIR / canopy_name
        canopy_dst = EXCLUDED_CANOPY_DIR / canopy_name

        if canopy_src.exists():
            shutil.move(str(canopy_src), str(canopy_dst))
            result['moved_canopy'].append(canopy_name)
        else:
            result['missing_canopy'].append(canopy_name)

    return result


def main():
    """Main function to exclude tiles with temporal anomalies."""
    logging.info("Starting temporal anomaly exclusion process")
    
    # Create excluded directories if they don't exist
    EXCLUDED_RGB_DIR.mkdir(parents=True, exist_ok=True)
    move_canopy = CANOPY_TILE_DIR.exists()
    if move_canopy:
        EXCLUDED_CANOPY_DIR.mkdir(parents=True, exist_ok=True)
    else:
        logging.info("Canopy tile directory not found; skipping canopy exclusions")
    
    # Read exclusion list
    try:
        excluded_tiles = read_exclusion_list()
    except FileNotFoundError as e:
        logging.error(str(e))
        return 1
    
    # Process each tile
    imagery_moved = 0
    imagery_missing = 0
    canopy_moved = 0
    canopy_missing = 0
    tile_results = []

    for tile_id in excluded_tiles:
        logging.info(f"Processing tile {tile_id}...")
        result = move_tile_set(tile_id, move_canopy)

        imagery_moved += len(result['moved_png'])
        imagery_missing += len(result['missing_png'])
        canopy_moved += len(result['moved_canopy'])
        canopy_missing += len(result['missing_canopy'])

        tile_results.append(result)

        if result['missing_png']:
            logging.warning(f"  Missing imagery for {tile_id}: {result['missing_png']}")
        if result['missing_canopy']:
            logging.warning(f"  Missing canopy data for {tile_id}: {result['missing_canopy']}")
    
    # Generate summary report
    logging.info("\n=== EXCLUSION SUMMARY ===")
    logging.info(f"Total tile sets processed: {len(excluded_tiles)}")
    logging.info(f"Total imagery files moved: {imagery_moved}")
    logging.info(f"Total imagery files missing: {imagery_missing}")
    logging.info(f"Expected imagery files: {len(excluded_tiles) * len(TIMES)}")
    if move_canopy:
        logging.info(f"Total canopy files moved: {canopy_moved}")
        logging.info(f"Total canopy files missing: {canopy_missing}")
        logging.info(f"Expected canopy files: {len(excluded_tiles)}")
    
    # Save detailed report
    report_path = Path('data/analysis')
    report_path.mkdir(parents=True, exist_ok=True)
    
    report_file = report_path / 'exclusion_report.txt'
    with open(report_file, 'w') as f:
        f.write("Temporal Anomaly Exclusion Report\n")
        f.write("=================================\n\n")
        f.write(f"Total tile sets excluded: {len(excluded_tiles)}\n")
        f.write(f"Total imagery files moved: {imagery_moved}\n")
        f.write(f"Total imagery files missing: {imagery_missing}\n")
        if move_canopy:
            f.write(f"Total canopy files moved: {canopy_moved}\n")
            f.write(f"Total canopy files missing: {canopy_missing}\n")
        f.write("\n")
        
        f.write("Excluded Tiles:\n")
        for result in tile_results:
            f.write(f"\n{result['tile_id']}:\n")
            f.write(f"  Moved imagery: {len(result['moved_png'])} files\n")
            if result['missing_png']:
                f.write(f"  Missing imagery: {result['missing_png']}\n")
            if move_canopy:
                f.write(f"  Moved canopy: {len(result['moved_canopy'])} files\n")
                if result['missing_canopy']:
                    f.write(f"  Missing canopy: {result['missing_canopy']}\n")
    
    logging.info(f"\nDetailed report saved to {report_file}")
    logging.info(f"Excluded tiles moved to {EXCLUDED_RGB_DIR} and {EXCLUDED_CANOPY_DIR if move_canopy else 'N/A'}")
    
    # List remaining tiles
    remaining_tiles = set()
    for tile_path in RGB_TILE_DIR.glob("*.png"):
        parts = tile_path.stem.split('_')
        if len(parts) == 3:
            tile_id = f"{parts[0]}_{parts[1]}"
            remaining_tiles.add(tile_id)
    
    logging.info(f"\nRemaining tile sets in dataset: {len(remaining_tiles)}")
    
    return 0


if __name__ == "__main__":
    exit(main())
