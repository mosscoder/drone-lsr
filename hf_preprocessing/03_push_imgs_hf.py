import logging
from pathlib import Path
from collections import defaultdict
from datasets import Dataset, Features, Image, Value
from huggingface_hub import HfApi
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

hf_org = 'mpg-ranch'
hf_repo = 'light-stable-semantics'
TILES_DIR = Path('data/raster/tiles')
TARGET_SHARD_SIZE_MB = 500
TIMES = [1000, 1200, 1500]

# Dataset features schema
features = Features({
    'image_t0': Image(),  # will automatically convert to PIL.Image from path, time 1000
    'image_t1': Image(),  # will automatically convert to PIL.Image from path, time 1200
    'image_t2': Image(),  # will automatically convert to PIL.Image from path, time 1500
    'idx': Value('string'), # {ROW}_{COL}
})

def scan_tiles():
    """Scan tiles directory and organize by row_col identifier."""
    logging.info("Scanning tiles directory...")
    
    tiles_by_location = defaultdict(dict)
    
    # Scan all tile files
    for tile_path in TILES_DIR.glob('*.png'):
        # Parse filename: {ROW}_{COL}_{TIME}.png
        parts = tile_path.stem.split('_')
        if len(parts) != 3:
            logging.warning(f"Skipping malformed filename: {tile_path.name}")
            continue
            
        row, col, time = parts
        try:
            time = int(time)
            if time not in TIMES:
                logging.warning(f"Unexpected time {time} in {tile_path.name}")
                continue
        except ValueError:
            logging.warning(f"Invalid time format in {tile_path.name}")
            continue
            
        location_id = f"{row}_{col}"
        tiles_by_location[location_id][time] = tile_path
    
    # Filter for complete sets (all 3 time points)
    complete_tiles = {}
    for location_id, time_dict in tiles_by_location.items():
        if len(time_dict) == 3 and all(t in time_dict for t in TIMES):
            complete_tiles[location_id] = time_dict
        else:
            missing_times = [t for t in TIMES if t not in time_dict]
            logging.debug(f"Incomplete tile set {location_id}, missing times: {missing_times}")
    
    logging.info(f"Found {len(complete_tiles)} complete tile sets out of {len(tiles_by_location)} total locations")
    return complete_tiles


def main():
    """Main function to process tiles and upload to Hugging Face."""
    logging.info("Starting Hugging Face upload process")
    
    repo_id = f"{hf_org}/{hf_repo}"
    api = HfApi()
    
    # Delete old parquet files
    logging.info("Checking for old parquet files to delete...")
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        parquet_files = [f for f in files if f.endswith('.parquet')]
        
        if parquet_files:
            logging.info(f"Found {len(parquet_files)} parquet files to delete")
            for parquet_file in parquet_files:
                try:
                    api.delete_file(path_in_repo=parquet_file, repo_id=repo_id, repo_type="dataset")
                    logging.info(f"Deleted: {parquet_file}")
                except Exception as e:
                    logging.warning(f"Error deleting {parquet_file}: {e}")
        else:
            logging.info("No parquet files found to delete")
    except Exception as e:
        logging.warning(f"Error listing/deleting files: {e}")
    
    # Scan and organize tiles
    tiles_dict = scan_tiles()
    if not tiles_dict:
        logging.error("No complete tile sets found")
        return 1
    
    # Create dataset records
    records = []
    for location_id, time_paths in tiles_dict.items():
        record = {
            'image_t0': str(time_paths[1000]),
            'image_t1': str(time_paths[1200]), 
            'image_t2': str(time_paths[1500]),
            'idx': location_id
        }
        records.append(record)
    logging.info(f"Created {len(records)} dataset records")
    
    # Create and upload dataset
    logging.info(f"Creating dataset from {len(records)} records...")
    df = pd.DataFrame(records)
    dataset = Dataset.from_pandas(df, features=features, preserve_index=False)
    
    logging.info(f"Uploading dataset to {repo_id}...")
    dataset.push_to_hub(repo_id, max_shard_size=f"{TARGET_SHARD_SIZE_MB}MB")
    logging.info(f"✓ Successfully uploaded dataset")
    
    logging.info(f"🎉 Successfully uploaded dataset to {repo_id}")
    logging.info(f"📊 Dataset available at: https://huggingface.co/datasets/{repo_id}")
    
    return 0

if __name__ == "__main__":
    exit(main())