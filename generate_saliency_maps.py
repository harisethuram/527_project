"""
Generate saliency maps for CNN and Transformer models trained on CheXpert.
Saliency maps are computed as gradients of model outputs w.r.t. input images.
Can process either a dataset or a single JPEG image.
"""

import torch
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
from datasets import load_from_disk
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import cv2

from src.model.CNN import ChexpertCNN
from src.model.transformer import ChexpertTransformer

def collate_fn(batch):
    images = torch.stack([b["image_u8"] for b in batch])   # uint8 CHW
    labels = torch.stack([b["labels"] for b in batch])
    return images, labels

def compute_saliency_map(model, images, labels, device):
    """
    Compute saliency maps for a batch of images.
    
    Args:
        model: The neural network model
        images: Input images (B, C, H, W) - already normalized
        labels: Ground truth labels (B, num_classes)
        device: Device to use
        
    Returns:
        saliency_maps: (B, H, W) - max absolute gradient across channels
    """
    # Enable gradient computation for input
    images.requires_grad_(True)
    
    # Forward pass
    outputs = model(images)
    
    # Compute loss for each sample and get gradients
    # Use binary cross entropy with logits
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs, labels.clamp(min=0.0), reduction='none'
    )
    
    # Sum loss across classes (to get total loss per sample)
    loss_per_sample = loss.sum(dim=1)
    total_loss = loss_per_sample.mean()
    
    # Backward pass
    total_loss.backward()
    
    # Get gradients with respect to input
    gradients = images.grad.data  # (B, C, H, W)
    
    # Compute saliency map as max absolute gradient across channels
    saliency_maps = torch.max(torch.abs(gradients), dim=1)[0]  # (B, H, W)
    
    return saliency_maps.cpu().numpy()

def normalize_saliency(saliency_map):
    """Normalize saliency map to [0, 1] range."""
    min_val = saliency_map.min()
    max_val = saliency_map.max()
    if max_val > min_val:
        return (saliency_map - min_val) / (max_val - min_val)
    else:
        return saliency_map

def visualize_and_save_saliency(original_image, saliency_map, output_path, colormap='jet'):
    """
    Visualize saliency map and save as image.
    
    Args:
        original_image: Original image (C, H, W) in range [0, 1] or [0, 255]
        saliency_map: Saliency map (H, W) in range [0, 1]
        output_path: Path to save the visualization
        colormap: Colormap to use for visualization
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Normalize saliency_map to [0, 1]
    saliency_normalized = normalize_saliency(saliency_map)
    
    # Create figure with subplots: original + saliency + overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    if original_image.shape[0] == 3:
        # RGB image
        img_display = original_image.transpose(1, 2, 0)
        img_display = np.clip(img_display, 0, 1)
        axes[0].imshow(img_display, cmap='gray')
    else:
        # Grayscale image
        axes[0].imshow(original_image[0], cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot saliency map
    im1 = axes[1].imshow(saliency_normalized, cmap=colormap)
    axes[1].set_title('Saliency Map')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # Plot overlay (original + saliency)
    if original_image.shape[0] == 3:
        img_display = original_image.transpose(1, 2, 0)
        img_display = np.clip(img_display, 0, 1)
        axes[2].imshow(img_display, cmap='gray', alpha=0.6)
    else:
        axes[2].imshow(original_image[0], cmap='gray', alpha=0.6)
    im2 = axes[2].imshow(saliency_normalized, cmap=colormap, alpha=0.6)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

def load_jpeg_image(image_path, target_size=(224, 224)):
    """
    Load a JPEG image and convert to tensor format matching model input expectations.
    
    Args:
        image_path: Path to JPEG file
        target_size: Target size (H, W) for resizing
        
    Returns:
        image_tensor: (1, 3, H, W) tensor in [0, 255] range (uint8 equivalent)
        original_image: (3, H, W) numpy array in [0, 1] range for visualization
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Resize to target size
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_np = np.array(img, dtype=np.uint8)  # (H, W, 3)
    
    # Convert to tensor format: (1, 3, H, W)
    image_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    # Also prepare for visualization: (3, H, W) in [0, 1] range
    original_image = img_np.astype(np.float32).transpose(2, 0, 1) / 255.0
    
    return image_tensor, original_image

def main():
    parser = argparse.ArgumentParser(description="Generate saliency maps for trained models")
    parser.add_argument("--models", type=str, default="cnn,transformer", 
                        choices=["cnn", "transformer", "cnn,transformer"],
                        help="Models to generate saliency maps for (comma-separated)")
    parser.add_argument("--model_dir", type=str, default="models/init", 
                        help="Directory containing trained model weights")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to a single JPEG image to process (if not set, uses dataset)")
    parser.add_argument("--data_dir", type=str, default="data/test", 
                        help="Path to test dataset (ignored if image_path is provided)")
    parser.add_argument("--output_dir", type=str, default="saliency_maps/", 
                        help="Directory to save saliency map visualizations")
    parser.add_argument("--num_samples", type=int, default=None, 
                        help="Number of samples to process (None for all, only for dataset mode)")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for processing (only for dataset mode)")
    parser.add_argument("--colormap", type=str, default="jet", 
                        help="Colormap to use for visualization")
    
    args = parser.parse_args()
    
    # Parse models
    models_to_process = [m.strip() for m in args.models.split(',')]
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Process each model
    for model_name in models_to_process:
        print(f"\n{'='*60}")
        print(f"Processing {model_name.upper()} model")
        print(f"{'='*60}")
        
        # Initialize model
        if model_name == "cnn":
            model = ChexpertCNN()
        elif model_name == "transformer":
            model = ChexpertTransformer()
        else:
            print(f"Skipping unknown model: {model_name}")
            continue
        
        # Load trained weights
        weights_path = os.path.join(args.model_dir, model_name,f"{model_name}_chexpert.pth")
        if not os.path.exists(weights_path):
            print(f"Warning: Model weights not found at {weights_path}")
            continue
        
        print(f"Loading weights from {weights_path}...")
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        # Create output directory for this model
        model_output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Process single image or dataset
        if args.image_path is not None:
            # Single image mode
            print(f"Processing single JPEG image: {args.image_path}")
            if not os.path.exists(args.image_path):
                print(f"Error: Image file not found at {args.image_path}")
                continue
            
            # Load and process the image
            image_tensor, original_image = load_jpeg_image(args.image_path)
            image_tensor = image_tensor.to(DEVICE, non_blocking=True).float().div_(255.0)
            
            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
            image_tensor = (image_tensor - mean) / std
            
            # Create dummy labels (all zeros)
            labels = torch.zeros(1, 5, device=DEVICE)
            
            # Compute saliency map
            with torch.enable_grad():
                saliency_map = compute_saliency_map(model, image_tensor, labels, DEVICE)
            
            # Save visualization
            image_basename = Path(args.image_path).stem
            output_path = os.path.join(model_output_dir, f"saliency_{image_basename}.png")
            visualize_and_save_saliency(original_image, saliency_map[0], output_path, 
                                       colormap=args.colormap)
            print(f"✓ Saved saliency map to {output_path}")
        
        else:
            # Dataset mode
            print(f"Loading validation dataset from {args.data_dir}...")
            validation_dataset = load_from_disk(args.data_dir)
            validation_dataset.set_format(type="torch", columns=["image_u8", "labels"])
            
            # Create dataloader without shuffling to maintain reproducibility
            num_workers = min(8, (os.cpu_count() or 4))
            validation_dataloader = DataLoader(
                validation_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                collate_fn=collate_fn,
            )
            
            # Process batches
            sample_count = 0
            print(f"Generating saliency maps...")
            
            with torch.no_grad():  # No grad for image normalization, but we'll enable it inside compute_saliency
                for batch_idx, (images, labels) in enumerate(tqdm(validation_dataloader)):
                    # Normalize images
                    images_normalized = images.to(DEVICE, non_blocking=True).float().div_(255.0)
                    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
                    images_normalized = (images_normalized - mean) / std
                    
                    labels = labels.to(DEVICE, non_blocking=True)
                    
                    # Compute saliency maps (this enables gradients internally)
                    # We need to exit no_grad context for gradient computation
                    with torch.enable_grad():
                        saliency_maps = compute_saliency_map(model, images_normalized, labels, DEVICE)
                    
                    # Save saliency maps
                    batch_size = images.shape[0]
                    for i in range(batch_size):
                        # Get original image (denormalized for visualization)
                        original_img = images[i].float() / 255.0  # Convert to [0, 1]
                        original_img = original_img.cpu().numpy()
                        
                        saliency_map = saliency_maps[i]
                        
                        # Save visualization
                        output_path = os.path.join(model_output_dir, f"saliency_{sample_count:05d}.png")
                        visualize_and_save_saliency(original_img, saliency_map, output_path, 
                                                   colormap=args.colormap)
                        
                        sample_count += 1
                        
                        # Check if we've reached the desired number of samples
                        if args.num_samples is not None and sample_count >= args.num_samples:
                            break
                    
                    if args.num_samples is not None and sample_count >= args.num_samples:
                        break
            
            print(f"✓ Generated {sample_count} saliency maps for {model_name} model")
            print(f"  Saved to: {model_output_dir}")

if __name__ == "__main__":
    main()
