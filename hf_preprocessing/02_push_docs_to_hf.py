import logging
from pathlib import Path
from huggingface_hub import HfApi

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

hf_org = 'mpg-ranch'
hf_repo = 'light-stable-semantics'

def main():
    """Main function to upload documentation files to Hugging Face."""
    logging.info("Starting documentation upload process")
    
    repo_id = f"{hf_org}/{hf_repo}"
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False)
        logging.info(f"Created repository: {repo_id}")
    except Exception as e:
        if "already exists" in str(e).lower():
            logging.info(f"Repository {repo_id} already exists")
        else:
            logging.warning(f"Error creating repository: {e}")
    
    # Upload README
    readme_path = Path('data/huggingface/README.md')
    if readme_path.exists():
        logging.info("Uploading README.md...")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset"
        )
        logging.info(" Uploaded README.md")
    else:
        logging.warning(f"README.md not found at {readme_path}")
    
    logging.info(f"<� Documentation upload complete for {repo_id}")
    logging.info(f"=� Dataset available at: https://huggingface.co/datasets/{repo_id}")
    
    return 0

if __name__ == "__main__":
    exit(main())