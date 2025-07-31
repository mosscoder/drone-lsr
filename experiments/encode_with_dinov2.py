import torch
from transformers import AutoImageProcessor, AutoModel
from datasets import load_dataset
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def encode_scene(scene_data, model, processor, device):
    """
    Encode a single scene (row) containing three time-aligned images.
    
    Args:
        scene_data: Dictionary containing 'image_t0', 'image_t1', 'image_t2', and 'idx'
        model: Loaded DINOv2 model
        processor: DINOv2 image processor
        device: torch device (cuda or cpu)
        
    Returns:
        torch.Tensor: Shape (3, 768) - time points x features
    """
    # Initialize tensor for this scene
    scene_features = torch.zeros(3, 768)
    
    # Process each time point
    time_keys = ['image_t0', 'image_t1', 'image_t2']
    
    with torch.no_grad():
        for t, time_key in enumerate(time_keys):
            # Get image for this time point
            img = scene_data[time_key]
            
            # Preprocess and extract features
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            
            # Extract CLS token embedding and move to CPU
            cls_embedding = outputs.last_hidden_state[0, 0, :].cpu()  # Shape: (768,)
            scene_features[t, :] = cls_embedding
            
    return scene_features


def save_matrix(tensor, output_path):
    """
    Save a tensor to a .pt file.
    
    Args:
        tensor: PyTorch tensor to save
        output_path: Path where to save the tensor
    """
    # Ensure parent directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save tensor
    torch.save(tensor, output_path)
    logging.info(f"Saved tensor with shape {tensor.shape} to {output_path}")


def main():
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load DINOv2 model and processor
    logging.info("Loading DINOv2 model...")
    model_name = "facebook/dinov2-base"  # 768-dim features
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)  # Move model to appropriate device
    model.eval()  # Set to evaluation mode
    
    # Load dataset from HuggingFace Hub
    logging.info("Loading light-stable-semantics dataset...")
    dataset = load_dataset("mpg-ranch/light-stable-semantics", split="train")
    
    # Process single example
    example_idx = 0
    scene_data = dataset[example_idx]
    scene_id = scene_data['idx']
    logging.info(f"Processing scene {scene_id}")
    
    # Encode the scene
    scene_features = encode_scene(scene_data, model, processor, device)
    logging.info(f"Encoded scene shape: {scene_features.shape}")
    
    # Save the single example
    output_path = "data/tabular/single_scene_example_svd_input.pt"
    save_matrix(scene_features, output_path)
    
    print(f"\n✓ Successfully encoded scene {scene_id}")
    print(f"✓ Output saved to {output_path}")
    print(f"✓ Tensor shape: {scene_features.shape}")
    
    # HOW TO PROCESS THE FULL DATASET:
    #
    # To encode all scenes in the dataset, you'll need to adapt the code above. Here's the process:
    #
    # First, you already have the dataset loaded and the model initialized from the example above.
    # You can reuse these same objects for processing all scenes.
    #
    # Create a tensor to store all scene encodings. Since each scene produces a (3, 768) tensor
    # and you have approximately 609 scenes, your final tensor should have shape (609, 3, 768).
    # This represents scenes x time points x features.
    #
    # Loop through all scenes in the dataset. For each scene, call the encode_scene function
    # with the scene data, model, processor, and device. This will return a (3, 768) tensor for that scene.
    #
    # Store each scene's encoding in the appropriate position of your full tensor. The scene index
    # from your loop becomes the first dimension index in the output tensor.
    #
    # After processing all scenes, use the save_matrix function to save the complete tensor.
    #
    # The final tensor will have shape (num_scenes, 3, 768) where:
    # - Axis 0: Different scenes/locations in the dataset
    # - Axis 1: Time points (0=morning, 1=noon, 2=afternoon)
    # - Axis 2: DINOv2 feature dimensions

if __name__ == "__main__":
    main()