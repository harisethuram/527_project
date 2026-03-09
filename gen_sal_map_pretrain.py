"""
Generate saliency maps for true positive predictions using a pretrained TorchXRayVision model.

This script:
- loads a saved test dataset from disk
- loads a CSV of test labels
- aligns CSV label names to the model's output labels
- runs inference
- computes saliency maps only for true positives
- saves per-sample saliency visualizations
- saves metrics and predicted probabilities

Example:
    python generate_correct_saliency_maps.py \
        --data_dir data/test \
        --labels_csv data/test/test_labels.csv \
        --output_dir saliency_maps_correct/pretrain \
        --batch_size 4 \
        --threshold 0.3
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchxrayvision as xrv


# CSV headers:
# Path,No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,
# Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,
# Pleural Other,Fracture,Support Devices

# Map CSV column names -> model label names.
# TorchXRayVision CheXpert weights often use "Effusion" rather than "Pleural Effusion".
CSV_TO_MODEL_CANDIDATES = {
    "Atelectasis": ["Atelectasis"],
    "Cardiomegaly": ["Cardiomegaly"],
    "Consolidation": ["Consolidation"],
    "Edema": ["Edema"],
    "Pleural Effusion": ["Pleural Effusion", "Effusion"],
}


def collate_fn(batch):
    """Collate function for dataloader."""
    images = torch.stack([b["image_u8"] for b in batch])  # uint8 CHW
    return images


def get_model_labels(model):
    """Get model output label names from TorchXRayVision model."""
    if hasattr(model, "targets") and model.targets is not None:
        labels = list(model.targets)
    elif hasattr(model, "pathologies") and model.pathologies is not None:
        labels = list(model.pathologies)
    else:
        raise ValueError("Could not find model label names on model.targets or model.pathologies.")

    return labels


def build_label_alignment(csv_columns, model_labels):
    """
    Build alignment between CSV labels and model output labels.

    Returns:
        active_labels: list of dicts with keys:
            - csv_name
            - model_name
            - model_idx
    """
    active_labels = []
    model_label_to_index = {
        name: idx for idx, name in enumerate(model_labels) if isinstance(name, str) and name.strip() != ""
    }

    for csv_name, candidates in CSV_TO_MODEL_CANDIDATES.items():
        if csv_name not in csv_columns:
            continue

        matched_model_name = None
        for candidate in candidates:
            if candidate in model_label_to_index:
                matched_model_name = candidate
                break

        if matched_model_name is not None:
            active_labels.append(
                {
                    "csv_name": csv_name,
                    "model_name": matched_model_name,
                    "model_idx": model_label_to_index[matched_model_name],
                }
            )

    if not active_labels:
        raise ValueError("No overlapping labels found between CSV headers and model outputs.")

    return active_labels


def load_and_align_test_labels(labels_csv_path, model_labels):
    """
    Load test labels from CSV and align them to model outputs.

    Returns:
        df: DataFrame
        active_labels: list of dicts with csv/model/index alignment
    """
    df = pd.read_csv(labels_csv_path)
    print(f"Loaded {len(df)} test labels from {labels_csv_path}")

    active_labels = build_label_alignment(df.columns, model_labels)

    # Ensure aligned label columns are numeric and binary-ish
    for item in active_labels:
        csv_name = item["csv_name"]
        df[csv_name] = pd.to_numeric(df[csv_name], errors="coerce")

    missing = {item["csv_name"]: int(df[item["csv_name"]].isna().sum()) for item in active_labels}
    missing = {k: v for k, v in missing.items() if v > 0}
    if missing:
        raise ValueError(f"Found missing/non-numeric values in label columns: {missing}")

    print("\nAligned labels:")
    for item in active_labels:
        print(
            f"  CSV '{item['csv_name']}' -> model '{item['model_name']}' "
            f"(index {item['model_idx']})"
        )

    return df, active_labels


def compute_saliency_map(model, image, label, class_idx):
    """
    Compute saliency map for a single image and specific class.

    Args:
        model: torch model
        image: (1, C, H, W), already normalized
        label: scalar ground-truth label (0 or 1)
        class_idx: model output index

    Returns:
        saliency_map: (H, W) numpy array
    """
    model.zero_grad(set_to_none=True)
    img = image.clone().detach().requires_grad_(True)

    output = model(img)
    class_output = output[:, class_idx]  # shape (1,)
    target = torch.tensor([label], device=img.device, dtype=class_output.dtype).clamp(min=0.0)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(class_output, target)
    loss.backward()

    gradients = img.grad.detach()  # (1, C, H, W)
    saliency_map = torch.max(torch.abs(gradients), dim=1)[0]  # (1, H, W)
    return saliency_map.cpu().numpy()[0]

def compute_saliency_map_no_label(model, image, class_idx, use_prob=False, abs_value=True):
    """
    Compute a saliency map for a single image and specific class using the
    model's own output for that class (not the ground-truth label).

    Args:
        model: torch model
        image: (1, C, H, W), already normalized
        class_idx: int, model output index
        use_prob: if True, backprop through sigmoid(logit); otherwise use raw logit
                  Using the raw logit is usually preferred for saliency.
        abs_value: if True, use absolute gradients before channel reduction

    Returns:
        saliency_map: (H, W) numpy array
    """
    model.eval()
    model.zero_grad(set_to_none=True)

    img = image.clone().detach().requires_grad_(True)

    output = model(img)                  # shape: (1, num_classes)
    class_score = output[:, class_idx]   # shape: (1,)

    # Preferred: use the raw class logit
    target_score = torch.sigmoid(class_score) if use_prob else class_score

    # Backprop the model's own score for this class
    target_score.backward(torch.ones_like(target_score))

    gradients = img.grad.detach()        # shape: (1, C, H, W)

    if abs_value:
        gradients = gradients.abs()

    # Collapse channel dimension -> (1, H, W)
    saliency_map = gradients.max(dim=1)[0]

    return saliency_map.cpu().numpy()[0]

def normalize_saliency(saliency_map):
    """Normalize saliency map to [0, 1]."""
    min_val = float(saliency_map.min())
    max_val = float(saliency_map.max())
    if max_val > min_val:
        return (saliency_map - min_val) / (max_val - min_val)
    return saliency_map


def visualize_saliency(original_image, saliency_map, output_path):
    """
    Save a saliency visualization.

    Args:
        original_image: (C, H, W) in [0, 1]
        saliency_map: (H, W)
        output_path: destination path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    saliency_normalized = normalize_saliency(saliency_map)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    if original_image.shape[0] == 3:
        img_display = np.clip(original_image.transpose(1, 2, 0), 0, 1)
        axes[0].imshow(img_display, cmap="gray")
    else:
        axes[0].imshow(original_image[0], cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Saliency
    im1 = axes[1].imshow(saliency_normalized, cmap="jet")
    axes[1].set_title("Saliency Map")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1])

    # Overlay
    if original_image.shape[0] == 3:
        img_display = np.clip(original_image.transpose(1, 2, 0), 0, 1)
        axes[2].imshow(img_display, cmap="gray", alpha=0.6)
    else:
        axes[2].imshow(original_image[0], cmap="gray", alpha=0.6)
    im2 = axes[2].imshow(saliency_normalized, cmap="jet", alpha=0.6)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def preprocess_batch_for_model(images, device):
    """
    Convert uint8 images to TorchXRayVision-style input:
    - single channel
    - pixel range approximately [-1024, 1024]
    """
    images = images.to(device, non_blocking=True).float()

    # If stored as 3 identical channels, keep one
    if images.shape[1] == 3:
        images = images[:, :1, :, :]

    # Convert 8-bit [0,255] to [-1024,1024]
    images = (images / 255.0) * 2048.0 - 1024.0

    return images


def main():
    parser = argparse.ArgumentParser(
        description="Generate saliency maps for true positive predictions"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/test",
        help="Path to saved test dataset directory",
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default="data/test/test_labels.csv",
        help="Path to test labels CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saliency_maps_correct/pretrain_final_no_labels/",
        help="Directory to save saliency maps and metrics",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Probability threshold for positive prediction",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of dataset samples to process",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained model
    model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    model.to(device)
    model.eval()

    model_labels = get_model_labels(model)
    print("\nModel labels:")
    for idx, label in enumerate(model_labels):
        print(f"  {idx:2d}: {repr(label)}")

    # Load test dataset
    print(f"\nLoading test dataset from {args.data_dir}...")
    test_dataset = load_from_disk(args.data_dir)
    test_dataset.set_format(type="torch", columns=["image_u8"])

    # Load and align CSV labels
    test_labels_df, active_labels = load_and_align_test_labels(args.labels_csv, model_labels)
    active_label_names = [item["csv_name"] for item in active_labels]

    # Dataloader
    num_workers = min(8, (os.cpu_count() or 4))
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
    )

    # Stats
    total_evaluations = 0
    total_correct = 0
    saliency_maps_saved = 0
    print("ACTIVE LABELS")
    print(active_labels)
    ailment_correct_count = {item["csv_name"]: 0 for item in active_labels}
    ailment_tp_count = {item["csv_name"]: 0 for item in active_labels}
    ailment_tn_count = {item["csv_name"]: 0 for item in active_labels}
    ailment_fp_count = {item["csv_name"]: 0 for item in active_labels}
    ailment_fn_count = {item["csv_name"]: 0 for item in active_labels}

    all_ground_truths = {item["csv_name"]: [] for item in active_labels}
    all_pred_probs = {item["csv_name"]: [] for item in active_labels}

    print(f"\nProcessing test dataset with threshold={args.threshold}...")

    stop_processing = False

    for batch_idx, images in enumerate(tqdm(test_dataloader)):
        batch_size = images.shape[0]
        images_normalized = preprocess_batch_for_model(images, device)

        with torch.no_grad():
            outputs = model(images_normalized)
            pred_probs = torch.sigmoid(outputs)
            predictions = (pred_probs > args.threshold).float()

        for i in range(batch_size):
            sample_num = batch_idx * args.batch_size + i

            if args.limit is not None and sample_num >= args.limit:
                stop_processing = True
                break

            if sample_num >= len(test_labels_df):
                stop_processing = True
                break

            label_row = test_labels_df.iloc[sample_num]

            # For display only
            original_img = (images[i].float() / 255.0).cpu().numpy()

            for item in active_labels:
                csv_name = item["csv_name"]
                model_idx = item["model_idx"]

                ground_truth = int(label_row[csv_name])
                prediction = int(predictions[i, model_idx].item())
                pred_prob = float(pred_probs[i, model_idx].item())

                total_evaluations += 1
                all_ground_truths[csv_name].append(ground_truth)
                all_pred_probs[csv_name].append(pred_prob)

                is_correct = (ground_truth == prediction)
                is_true_positive = (ground_truth == 1 and prediction == 1)
                is_true_negative = (ground_truth == 0 and prediction == 0)
                is_false_positive = (ground_truth == 0 and prediction == 1)
                is_false_negative = (ground_truth == 1 and prediction == 0)

                if is_correct:
                    total_correct += 1
                    ailment_correct_count[csv_name] += 1
                if is_true_positive:
                    ailment_tp_count[csv_name] += 1
                if is_true_negative:
                    ailment_tn_count[csv_name] += 1
                if is_false_positive:
                    ailment_fp_count[csv_name] += 1
                if is_false_negative:
                    ailment_fn_count[csv_name] += 1

                # if is_true_positive:
                saliency_map = compute_saliency_map_no_label(
                    model=model,
                    image=images_normalized[i:i+1],
                    # label=float(ground_truth),
                    class_idx=model_idx,
                )

                output_path = os.path.join(
                    args.output_dir,
                    f"sample_{sample_num:05d}",
                    f"{csv_name}.png",
                )
                visualize_saliency(original_img, saliency_map, output_path)
                saliency_maps_saved += 1

        if stop_processing:
            break

    # Metrics
    results = {"per_ailment": {}, "average": {}}
    auroc_list = []
    auprc_list = []

    for csv_name in active_label_names:
        gt = np.array(all_ground_truths[csv_name], dtype=np.int64)
        pp = np.array(all_pred_probs[csv_name], dtype=np.float32)

        ailment_results = {
            "correct": int(ailment_correct_count[csv_name]),
            "total": int(len(gt)),
            "true_positives": int(ailment_tp_count[csv_name]),
            "true_negatives": int(ailment_tn_count[csv_name]),
            "false_positives": int(ailment_fp_count[csv_name]),
            "false_negatives": int(ailment_fn_count[csv_name]),
        }

        if len(np.unique(gt)) >= 2:
            auroc = float(roc_auc_score(gt, pp))
            auprc = float(average_precision_score(gt, pp))
            ailment_results["auroc"] = auroc
            ailment_results["auprc"] = auprc
            auroc_list.append(auroc)
            auprc_list.append(auprc)
        else:
            ailment_results["auroc"] = None
            ailment_results["auprc"] = None

        results["per_ailment"][csv_name] = ailment_results

    results["average"]["auroc"] = float(np.mean(auroc_list)) if auroc_list else None
    results["average"]["auprc"] = float(np.mean(auprc_list)) if auprc_list else None
    results["total_evaluations"] = int(total_evaluations)
    results["total_correct"] = int(total_correct)
    results["accuracy"] = float(total_correct / total_evaluations) if total_evaluations > 0 else 0.0
    results["saliency_maps_saved"] = int(saliency_maps_saved)

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    scores_df = pd.DataFrame({csv_name: all_pred_probs[csv_name] for csv_name in active_label_names})
    scores_csv_path = os.path.join(args.output_dir, "scores.csv")
    scores_df.to_csv(scores_csv_path, index_label="sample")

    # Print summary
    print(f"\n{'=' * 60}")
    print("Saliency Map Generation Complete")
    print(f"{'=' * 60}")
    print(f"Total label evaluations: {total_evaluations}")
    print(f"Total correct predictions: {total_correct}")
    print(f"Accuracy: {100.0 * total_correct / total_evaluations:.2f}%")
    print(f"Saliency maps saved: {saliency_maps_saved}")

    print("\nPer-ailment metrics:")
    for csv_name in active_label_names:
        r = results["per_ailment"][csv_name]
        auroc_str = f"{r['auroc']:.4f}" if r["auroc"] is not None else "N/A"
        auprc_str = f"{r['auprc']:.4f}" if r["auprc"] is not None else "N/A"
        print(f"  {csv_name}: correct={r['correct']}/{r['total']}  AUROC={auroc_str}  AUPRC={auprc_str}")
        print(f"    True Positives: {r['true_positives']}")
        print(f"    True Negatives: {r['true_negatives']}")
        print(f"    False Positives: {r['false_positives']}")
        print(f"    False Negatives: {r['false_negatives']}")

    avg = results["average"]
    avg_auroc_str = f"{avg['auroc']:.4f}" if avg["auroc"] is not None else "N/A"
    avg_auprc_str = f"{avg['auprc']:.4f}" if avg["auprc"] is not None else "N/A"
    print(f"\nAverage AUROC: {avg_auroc_str}")
    print(f"Average AUPRC: {avg_auprc_str}")
    print(f"\nResults saved to: {results_path}")
    print(f"Scores saved to: {scores_csv_path}")
    print(f"Saliency maps saved to: {args.output_dir}")


if __name__ == "__main__":
    main()