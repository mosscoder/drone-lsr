import os
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EXCLUSION_LIST_PATH = Path('data/tabular/transient_phenomena.txt')
TILE_DIR = Path('data/raster/tiles')
EXCLUDED_DIR = Path('data/raster/excluded_tiles')
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


def move_tile_set(tile_id):
    """Move all time points for a given tile to the excluded directory."""
    moved_files = []
    missing_files = []
    
    for time in TIMES:
        filename = f"{tile_id}_{time}.png"
        src_path = TILE_DIR / filename
        dst_path = EXCLUDED_DIR / filename
        
        if src_path.exists():
            shutil.move(str(src_path), str(dst_path))
            moved_files.append(filename)
        else:
            missing_files.append(filename)
    
    return moved_files, missing_files


def main():
    """Main function to exclude tiles with temporal anomalies."""
    logging.info("Starting temporal anomaly exclusion process")
    
    # Create excluded directory if it doesn't exist
    EXCLUDED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Read exclusion list
    try:
        excluded_tiles = read_exclusion_list()
    except FileNotFoundError as e:
        logging.error(str(e))
        return 1
    
    # Process each tile
    total_moved = 0
    total_missing = 0
    tile_results = []
    
    for tile_id in excluded_tiles:
        logging.info(f"Processing tile {tile_id}...")
        moved_files, missing_files = move_tile_set(tile_id)
        
        total_moved += len(moved_files)
        total_missing += len(missing_files)
        
        tile_results.append({
            'tile_id': tile_id,
            'moved_files': moved_files,
            'missing_files': missing_files
        })
        
        if missing_files:
            logging.warning(f"  Missing files for {tile_id}: {missing_files}")
    
    # Generate summary report
    logging.info("\n=== EXCLUSION SUMMARY ===")
    logging.info(f"Total tile sets processed: {len(excluded_tiles)}")
    logging.info(f"Total files moved: {total_moved}")
    logging.info(f"Total files missing: {total_missing}")
    logging.info(f"Expected files: {len(excluded_tiles) * len(TIMES)}")
    
    # Save detailed report
    report_path = Path('data/analysis')
    report_path.mkdir(parents=True, exist_ok=True)
    
    report_file = report_path / 'exclusion_report.txt'
    with open(report_file, 'w') as f:
        f.write("Temporal Anomaly Exclusion Report\n")
        f.write("=================================\n\n")
        f.write(f"Total tile sets excluded: {len(excluded_tiles)}\n")
        f.write(f"Total files moved: {total_moved}\n")
        f.write(f"Total files missing: {total_missing}\n\n")
        
        f.write("Excluded Tiles:\n")
        for result in tile_results:
            f.write(f"\n{result['tile_id']}:\n")
            f.write(f"  Moved: {len(result['moved_files'])} files\n")
            if result['missing_files']:
                f.write(f"  Missing: {result['missing_files']}\n")
    
    logging.info(f"\nDetailed report saved to {report_file}")
    logging.info(f"Excluded tiles moved to {EXCLUDED_DIR}")
    
    # List remaining tiles
    remaining_tiles = set()
    for tile_path in TILE_DIR.glob("*.png"):
        parts = tile_path.stem.split('_')
        if len(parts) == 3:
            tile_id = f"{parts[0]}_{parts[1]}"
            remaining_tiles.add(tile_id)
    
    logging.info(f"\nRemaining tile sets in dataset: {len(remaining_tiles)}")
    
    return 0


if __name__ == "__main__":
    exit(main())