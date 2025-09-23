import logging
import yaml
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, metadata_update

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

hf_org = 'mpg-ranch'
hf_repo = 'light-stable-semantics'

def extract_yaml_metadata(readme_path):
    """Extract YAML metadata from README.md file."""
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find YAML frontmatter between --- markers
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            yaml_content = parts[1].strip()
            try:
                metadata = yaml.safe_load(yaml_content)
                logging.info(f"Extracted metadata with keys: {list(metadata.keys())}")
                return metadata
            except yaml.YAMLError as e:
                logging.error(f"Error parsing YAML: {e}")
                return None
    
    logging.warning("No YAML frontmatter found in README.md")
    return None

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
    
    readme_path = Path('data/huggingface/README.md')
    if not readme_path.exists():
        logging.error(f"README.md not found at {readme_path}")
        return 1
    
    # Extract and update metadata first
    metadata = extract_yaml_metadata(readme_path)
    if metadata:
        logging.info("Updating dataset metadata...")
        try:
            metadata_update(repo_id, metadata, repo_type="dataset")
            logging.info("✓ Metadata updated successfully")
        except Exception as e:
            logging.error(f"Error updating metadata: {e}")
    
    # Upload README
    logging.info("Uploading README.md...")
    try:
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Update dataset card metadata - {datetime.now().isoformat()}"
        )
        logging.info("✓ Uploaded README.md")
    except Exception as e:
        logging.warning(f"README upload result: {e}")
    
    # Upload demo notebook if it exists
    notebook_path = Path('data/huggingface/demo_features.ipynb')
    if notebook_path.exists():
        logging.info("Uploading demo notebook...")
        try:
            api.upload_file(
                path_or_fileobj=str(notebook_path),
                path_in_repo="demo_features.ipynb",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Update demo notebook - {datetime.now().isoformat()}"
            )
            logging.info("✓ Uploaded demo_features.ipynb")
        except Exception as e:
            logging.warning(f"Notebook upload result: {e}")
    else:
        logging.info("Demo notebook not found, skipping")
    
    logging.info(f"📄 Documentation upload complete for {repo_id}")
    logging.info(f"🔗 Dataset available at: https://huggingface.co/datasets/{repo_id}")
    
    return 0

if __name__ == "__main__":
    exit(main())