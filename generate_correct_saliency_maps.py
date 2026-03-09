"""
Generate saliency maps for correct predictions only.

Processes test dataset, identifies samples/ailments where the model's prediction
is correct, and generates saliency maps showing which image regions influenced
the decision.

Usage:
    python generate_correct_saliency_maps.py --model_path models/init/cnn/cnn_chexpert.pth
    python generate_correct_saliency_maps.py --model_path models/init/transformer/transformer_chexpert.pth
"""

import torch
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
import matplotlib.pyplot as plt
import json
from sklearn.metrics import roc_auc_score, average_precision_score

from src.model.CNN import ChexpertCNN
from src.model.transformer import ChexpertTransformer

# Target ailments the model was trained on
TARGET_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

def collate_fn(batch):
    """Collate function for dataloader."""
    images = torch.stack([b["image_u8"] for b in batch])   # uint8 CHW
    labels = torch.stack([b["labels"] for b in batch])
    return images, labels

def load_model(model_path, device):
    """
    Load model from checkpoint.
    
    Args:
        model_path: Path to .pth file
        device: Device to load model on
        
    Returns:
        model: Loaded model in eval mode
        model_type: 'cnn' or 'transformer'
    """
    # Determine model type from path
    path_lower = model_path.lower()
    if 'cnn' in path_lower:
        model_type = 'cnn'
        model = ChexpertCNN(num_classes=5)
    elif 'transformer' in path_lower:
        model_type = 'transformer'
        model = ChexpertTransformer(num_classes=5)
    else:
        raise ValueError(f"Cannot determine model type from path: {model_path}")
    
    print(f"Loading {model_type.upper()} model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    
    return model, model_type

def compute_saliency_map(model, image, label, ailment_idx):
    """
    Compute saliency map for a single image and specific ailment.
    
    Args:
        model: The neural network model
        image: Input image (1, C, H, W) - already normalized, detached
        label: Ground truth label scalar for this ailment
        ailment_idx: Index of the ailment to compute saliency for
        
    Returns:
        saliency_map: (H, W) numpy array - max absolute gradient across channels
    """
    model.zero_grad()
    img = image.clone().detach().requires_grad_(True)
    
    output = model(img)
    ailment_output = output[0, ailment_idx:ailment_idx+1]
    target = torch.tensor([label], device=img.device).clamp(min=0.0)
    
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        ailment_output.unsqueeze(0), target.unsqueeze(0)
    )
    loss.backward()
    
    gradients = img.grad.data  # (1, C, H, W)
    saliency_map = torch.max(torch.abs(gradients), dim=1)[0]  # (1, H, W)
    
    return saliency_map.cpu().numpy()[0]

def normalize_saliency(saliency_map):
    """Normalize saliency map to [0, 1] range."""
    min_val = saliency_map.min()
    max_val = saliency_map.max()
    if max_val > min_val:
        return (saliency_map - min_val) / (max_val - min_val)
    else:
        return saliency_map

def visualize_saliency(original_image, saliency_map, output_path):
    """
    Save saliency visualization as image file.
    
    Args:
        original_image: Original image (C, H, W) in range [0, 1]
        saliency_map: Saliency map (H, W) in range [0, 1]
        output_path: Path to save the visualization
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
    im1 = axes[1].imshow(saliency_normalized, cmap='jet')
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
    im2 = axes[2].imshow(saliency_normalized, cmap='jet', alpha=0.6)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

def load_test_labels(labels_csv_path):
    """
    Load test labels from CSV file.
    
    Returns:
        df: DataFrame with sample paths and labels
    """
    df = pd.read_csv(labels_csv_path)
    print(f"Loaded {len(df)} test labels from {labels_csv_path}")
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Generate saliency maps for correct predictions only"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to trained model .pth file"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/test",
        help="Path to test dataset directory"
    )
    parser.add_argument(
        "--labels_csv", 
        type=str, 
        default="data/test/test_labels.csv",
        help="Path to test labels CSV file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="saliency_maps_correct",
        help="Directory to save saliency maps"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.3,
        help="Confidence threshold for positive predictions"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Maximum number of samples to process (None for all)"
    )
    
    args = parser.parse_args()
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Load model
    model, model_type = load_model(args.model_path, DEVICE)
    
    # Load test dataset
    print(f"Loading test dataset from {args.data_dir}...")
    test_dataset = load_from_disk(args.data_dir)
    test_dataset.set_format(type="torch", columns=["image_u8", "labels"])
    
    # Load test labels
    test_labels_df = load_test_labels(args.labels_csv)
    # print(test_labels_df)
    
    # Map image paths to their row index in the dataset
    # The test dataset should have sample indices that correspond to the CSV
    print("Creating sample index mapping...")
    sample_index = 0
    path_to_label_row = {}  # Maps dataset index to test_labels row
    
    # Create dataloader
    num_workers = min(8, (os.cpu_count() or 4))
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    
    # Statistics
    total_samples = 0
    total_correct = 0
    saliency_maps_saved = 0
    
    ailment_correct_count = {ailment: 0 for ailment in TARGET_LABELS}
    ailment_tp_count = {ailment: 0 for ailment in TARGET_LABELS}
    ailment_tn_count = {ailment: 0 for ailment in TARGET_LABELS}
    ailment_fp_count = {ailment: 0 for ailment in TARGET_LABELS}
    ailment_fn_count = {ailment: 0 for ailment in TARGET_LABELS}
    # Collect ground truths and predicted probabilities for AUROC/AUPRC
    all_ground_truths = {ailment: [] for ailment in TARGET_LABELS}
    all_pred_probs = {ailment: [] for ailment in TARGET_LABELS}
    
    print(f"Processing test dataset with threshold={args.threshold}...")
    
    for batch_idx, (images, labels) in enumerate(tqdm(test_dataloader)):
        batch_size = images.shape[0]
        
        # Normalize images (without gradients initially)
        images_normalized = images.to(DEVICE, non_blocking=True).float().div_(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
        images_normalized = (images_normalized - mean) / std
        
        labels = labels.to(DEVICE, non_blocking=True)
        
        # Get predictions (without gradients)
        with torch.no_grad():
            outputs = model(images_normalized)
            pred_probs = torch.sigmoid(outputs)  # (B, num_classes)
            predictions = (pred_probs > args.threshold).float()
        
        # Process each sample in batch
        for i in range(batch_size):
            sample_num = batch_idx * args.batch_size + i
            
            if args.limit is not None and sample_num >= args.limit:
                break
            
            if sample_num >= len(test_labels_df):
                break
            
            # Get ground truth labels for this sample
            label_row = test_labels_df.iloc[sample_num]
            
            # Get image for later use
            original_img = images[i].float() / 255.0  # Convert to [0, 1]
            original_img = original_img.cpu().numpy()
            
            # For each ailment, check if prediction is correct
            for ailment_idx, ailment_name in enumerate(TARGET_LABELS):
                # Ground truth from CSV (0.0 = negative, 1.0 = positive)
                ground_truth = label_row[ailment_name]
                
                # Prediction from model (0 = negative, 1 = positive)
                prediction = predictions[i, ailment_idx].item()
                
                total_samples += 1
                
                # Collect for AUROC/AUPRC
                all_ground_truths[ailment_name].append(int(ground_truth))
                all_pred_probs[ailment_name].append(pred_probs[i, ailment_idx].item())
                
                # Check for true positive: both ground truth and prediction are positive
                is_correct = int(ground_truth) == int(prediction)
                is_true_positive = int(ground_truth) == 1 and int(prediction) == 1
                is_true_negative = int(ground_truth) == 0 and int(prediction) == 0
                is_fp = int(ground_truth) == 1 and int(prediction) == 0
                is_fn = int(ground_truth) == 0 and int(prediction) == 1
                
                if is_correct:
                    total_correct += 1
                    ailment_correct_count[ailment_name] += 1
                if is_true_positive:
                    ailment_tp_count[ailment_name] += 1
                    
                if is_true_negative:
                    ailment_tn_count[ailment_name] += 1
                
                if is_fp:
                    ailment_fp_count[ailment_name] += 1
                if is_fn:
                    ailment_fn_count[ailment_name] += 1
                if is_true_positive:
                    # Compute saliency map for this ailment
                    saliency_map = compute_saliency_map(
                        model, images_normalized[i:i+1],
                        labels[i, ailment_idx].item(), ailment_idx
                    )
                    
                    # Save saliency map
                    output_path = os.path.join(
                        args.output_dir,
                        f"sample_{sample_num:05d}",
                        f"{ailment_name}.png"
                    )
                    visualize_saliency(original_img, saliency_map, output_path)
                    saliency_maps_saved += 1
        
        if args.limit is not None and sample_num >= args.limit:
            break
    
    # Compute AUROC and AUPRC per ailment
    results = {"per_ailment": {}, "average": {}}
    auroc_list = []
    auprc_list = []
    
    for ailment_name in TARGET_LABELS:
        gt = np.array(all_ground_truths[ailment_name])
        pp = np.array(all_pred_probs[ailment_name])
        
        ailment_results = {}
        # AUROC/AUPRC require both classes present
        if len(np.unique(gt)) >= 2:
            auroc = roc_auc_score(gt, pp)
            auprc = average_precision_score(gt, pp)
            ailment_results["auroc"] = auroc
            ailment_results["auprc"] = auprc
            auroc_list.append(auroc)
            auprc_list.append(auprc)
        else:
            ailment_results["auroc"] = None
            ailment_results["auprc"] = None
        
        ailment_results["correct"] = ailment_correct_count[ailment_name]
        ailment_results["total"] = len(gt)
        results["per_ailment"][ailment_name] = ailment_results
    
    results["average"]["auroc"] = float(np.mean(auroc_list)) if auroc_list else None
    results["average"]["auprc"] = float(np.mean(auprc_list)) if auprc_list else None
    results["total_samples"] = total_samples
    results["total_correct"] = total_correct
    results["accuracy"] = total_correct / total_samples if total_samples > 0 else 0.0
    results["saliency_maps_saved"] = saliency_maps_saved
    
    # Save results.json
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save scores CSV (rows=samples, columns=ailments, cells=predicted probabilities)
    scores_df = pd.DataFrame({ailment: all_pred_probs[ailment] for ailment in TARGET_LABELS})
    scores_csv_path = os.path.join(args.output_dir, "scores.csv")
    scores_df.to_csv(scores_csv_path, index_label="sample")
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Saliency Map Generation Complete")
    print(f"{'='*60}")
    print(f"Total samples processed: {total_samples}")
    print(f"Total correct predictions: {total_correct}")
    print(f"Accuracy: {100.0 * total_correct / total_samples:.2f}%")
    print(f"Saliency maps saved: {saliency_maps_saved}")
    print(f"\nPer-ailment metrics:")
    for ailment_name in TARGET_LABELS:
        
        r = results["per_ailment"][ailment_name]
        auroc_str = f"{r['auroc']:.4f}" if r["auroc"] is not None else "N/A"
        auprc_str = f"{r['auprc']:.4f}" if r["auprc"] is not None else "N/A"
        print(f"  {ailment_name}: correct={r['correct']}/{r['total']}  AUROC={auroc_str}  AUPRC={auprc_str}")
        print(f"    True Positives: {ailment_tp_count[ailment_name]}")
        print(f"    True Negatives: {ailment_tn_count[ailment_name]}")
        print(f"    False Positives: {ailment_fp_count[ailment_name]}")
        print(f"    False Negatives: {ailment_fn_count[ailment_name]}")
    #     print(f"    precision: {ailment_tp_count[ailment_name]/(ailment_fp_count[ailment_name]+ailment_tp_count[ailment_name])}")
    #     print(f"    recall: {ailment_tp_count[ailment_name]/(ailment_tp_count[ailment_name]+ailment_fn_count[ailment_name])}")
    # avg = results["average"]
    avg_auroc_str = f"{avg['auroc']:.4f}" if avg["auroc"] is not None else "N/A"
    avg_auprc_str = f"{avg['auprc']:.4f}" if avg["auprc"] is not None else "N/A"
    print(f"\nAverage AUROC: {avg_auroc_str}")
    print(f"Average AUPRC: {avg_auprc_str}")
    print(f"\nResults saved to: {results_path}")
    print(f"Saliency maps saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
