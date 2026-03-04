"""
Preprocess test set from chexlocalize/CheXpert/test and save in same format as train/validation.
"""
import argparse
import os
import numpy as np
from pathlib import Path
from PIL import Image
from datasets import Dataset
from torchvision import transforms
from tqdm import tqdm

TARGET_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

def preprocess_test_images(test_dir, resolution=224):
    """
    Load and preprocess all images from the test directory.
    
    Args:
        test_dir: Path to test directory containing patient folders
        resolution: Target resolution for resizing
        
    Returns:
        List of dicts with 'image_u8' and 'labels' keys
    """
    resize = transforms.Resize((resolution, resolution))
    
    images_list = []
    labels_list = []
    
    # Walk through patient folders
    test_path = Path(test_dir)
    patient_dirs = sorted([d for d in test_path.iterdir() if d.is_dir() and d.name.startswith("patient")])
    
    print(f"Found {len(patient_dirs)} patient directories")
    
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        # Walk through study directories
        study_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]
        
        for study_dir in study_dirs:
            # Find all JPG files
            jpg_files = list(study_dir.glob("*.jpg")) + list(study_dir.glob("*.JPG"))
            
            for jpg_file in jpg_files:
                try:
                    # Load and preprocess image
                    img = Image.open(jpg_file).convert("RGB")
                    img = resize(img)
                    arr = np.asarray(img, dtype=np.uint8)  # HWC
                    arr = np.transpose(arr, (2, 0, 1))      # CHW
                    
                    images_list.append(arr)
                    
                    # Create label vector - all -1 for unknown (test set has no labels)
                    labels = np.full(len(TARGET_LABELS), -1.0, dtype=np.float32)
                    labels_list.append(labels)
                    
                except Exception as e:
                    print(f"Error processing {jpg_file}: {e}")
                    continue
    
    print(f"Loaded {len(images_list)} images")
    
    return {
        "image_u8": images_list,
        "labels": labels_list,
    }

def main():
    parser = argparse.ArgumentParser(description="Preprocess CheXpert test set")
    parser.add_argument("--test_dir", type=str, default="chexlocalize/chexlocalize/CheXpert/test",
                        help="Path to test directory")
    parser.add_argument("--out_dir", type=str, default="data/test",
                        help="Output directory to save processed test set")
    parser.add_argument("--resolution", type=int, default=224,
                        help="Target resolution for images")
    args = parser.parse_args()
    
    print(args)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load and preprocess images
    print(f"Loading test images from {args.test_dir}...")
    data_dict = preprocess_test_images(args.test_dir, args.resolution)
    
    # Create dataset
    print("Creating dataset...")
    dataset = Dataset.from_dict(data_dict)
    
    # Save to disk
    print(f"Saving to {args.out_dir}...")
    dataset.save_to_disk(args.out_dir)
    
    print(f"✓ Preprocessed {len(dataset)} test images")
    print(f"✓ Saved to {args.out_dir}")
    print(f"\nDataset info:")
    print(f"  - Images: {len(dataset)}")
    print(f"  - Input shape: (3, {args.resolution}, {args.resolution})")
    print(f"  - Labels: {len(TARGET_LABELS)} conditions")

if __name__ == "__main__":
    main()
