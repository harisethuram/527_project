"""Evaluate CNN/CLIP/Transformer on validation and test splits with AUROC/AUPRC."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.CLIP import ChexpertCLIP
from src.model.CNN import ChexpertCNN
from src.model.transformer import ChexpertTransformer

TARGET_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]


def collate_fn(batch):
    images = torch.stack([b["image_u8"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return images, labels


def init_model(model_name):
    if model_name == "cnn":
        return ChexpertCNN()
    if model_name == "clip":
        return ChexpertCLIP()
    if model_name == "transformer":
        return ChexpertTransformer()
    raise ValueError(f"Unknown model '{model_name}'")


def normalize_images(images, device):
    images = images.to(device, non_blocking=True).float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (images - mean) / std


def evaluate_dataset(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = normalize_images(images, device)
            outputs = model(images)
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    ground_truth = np.concatenate(all_labels, axis=0)
    return predictions, ground_truth


def compute_metrics(predictions, ground_truth):
    probabilities = 1.0 / (1.0 + np.exp(-predictions))
    metrics = {}

    for index, condition in enumerate(TARGET_LABELS):
        pred = probabilities[:, index]
        gt = ground_truth[:, index]

        valid_mask = gt >= 0
        pred_valid = pred[valid_mask]
        gt_valid = gt[valid_mask]

        if len(gt_valid) == 0 or len(np.unique(gt_valid)) < 2:
            metrics[condition] = {
                "AUROC": np.nan,
                "AUPRC": np.nan,
                "n_samples": int(len(gt_valid)),
                "n_positive": int(np.sum(gt_valid == 1.0)),
            }
            continue

        auroc = roc_auc_score(gt_valid, pred_valid)
        precision, recall, _ = precision_recall_curve(gt_valid, pred_valid)
        auprc = auc(recall, precision)

        metrics[condition] = {
            "AUROC": float(auroc),
            "AUPRC": float(auprc),
            "n_samples": int(len(gt_valid)),
            "n_positive": int(np.sum(gt_valid == 1.0)),
        }

    return metrics


def print_metrics_table(metrics, title):
    print(f"\n{title}")
    print(f"{'Condition':<20} {'AUROC':<10} {'AUPRC':<10} {'Positives':<12} {'Total':<10}")
    print("-" * 68)
    for condition in TARGET_LABELS:
        row = metrics[condition]
        if np.isnan(row["AUROC"]):
            print(f"{condition:<20} {'N/A':<10} {'N/A':<10} {row['n_positive']:<12} {row['n_samples']:<10}")
        else:
            print(f"{condition:<20} {row['AUROC']:<10.4f} {row['AUPRC']:<10.4f} {row['n_positive']:<12} {row['n_samples']:<10}")


def load_test_label_dict(csv_path):
    df = pd.read_csv(csv_path)
    label_dict = {}
    for _, row in df.iterrows():
        rel_path = row["Path"]
        if rel_path.startswith("test/"):
            rel_path = rel_path[5:]
        label_dict[rel_path] = np.array(
            [
                row["Atelectasis"],
                row["Cardiomegaly"],
                row["Consolidation"],
                row["Edema"],
                row["Pleural Effusion"],
            ],
            dtype=np.float32,
        )
    return label_dict


def build_test_ground_truth(test_root, label_dict):
    test_root = Path(test_root)
    patient_dirs = sorted([path for path in test_root.iterdir() if path.is_dir() and path.name.startswith("patient")])

    ordered_labels = []
    for patient_dir in patient_dirs:
        study_dirs = sorted([path for path in patient_dir.iterdir() if path.is_dir()])
        for study_dir in study_dirs:
            image_paths = sorted(list(study_dir.glob("*.jpg")) + list(study_dir.glob("*.JPG")))
            for image_path in image_paths:
                rel_path = str(image_path.relative_to(test_root))
                if rel_path in label_dict:
                    ordered_labels.append(label_dict[rel_path])

    if not ordered_labels:
        raise RuntimeError("No test labels matched the discovered test images.")
    return np.stack(ordered_labels, axis=0)


def save_metrics_csv(results_summary, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for model_name, splits in results_summary.items():
        for split_name, split_metrics in splits.items():
            for condition in TARGET_LABELS:
                m = split_metrics[condition]
                rows.append(
                    {
                        "Model": model_name,
                        "Split": split_name,
                        "Condition": condition,
                        "AUROC": m["AUROC"],
                        "AUPRC": m["AUPRC"],
                        "n_samples": m["n_samples"],
                        "n_positive": m["n_positive"],
                    }
                )

    output_path = os.path.join(output_dir, "metrics_all.csv")
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved metrics to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on validation and test splits")
    parser.add_argument("--models", type=str, default="cnn,clip,transformer")
    parser.add_argument("--model_dir", type=str, default="models/init")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--test_root", type=str, default="chexlocalize/chexlocalize/CheXpert/test")
    parser.add_argument("--test_labels", type=str, default="chexlocalize/chexlocalize/CheXpert/test_labels.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_path = os.path.join(args.data_dir, "validation")
    test_path = os.path.join(args.data_dir, "test")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing validation split at {val_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test split at {test_path}")
    if not os.path.exists(args.test_labels):
        raise FileNotFoundError(f"Missing test_labels csv at {args.test_labels}")

    val_dataset = load_from_disk(val_path)
    val_dataset.set_format(type="torch", columns=["image_u8", "labels"])
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )

    test_dataset = load_from_disk(test_path)
    test_dataset.set_format(type="torch", columns=["image_u8", "labels"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )

    test_label_dict = load_test_label_dict(args.test_labels)
    test_ground_truth = build_test_ground_truth(args.test_root, test_label_dict)

    if len(test_dataset) != len(test_ground_truth):
        raise RuntimeError(
            f"Test dataset length ({len(test_dataset)}) does not match labels ({len(test_ground_truth)})."
        )

    requested_models = [name.strip().lower() for name in args.models.split(",") if name.strip()]
    valid_models = {"cnn", "clip", "transformer"}
    unknown = [name for name in requested_models if name not in valid_models]
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}. Expected subset of {sorted(valid_models)}")

    results_summary = {}
    for model_name in requested_models:
        print(f"\n{'=' * 80}\nEvaluating {model_name.upper()}\n{'=' * 80}")
        model = init_model(model_name)

        weights_path = os.path.join(args.model_dir, model_name, f"{model_name}_chexpert.pth")
        if not os.path.exists(weights_path):
            alt_path = os.path.join(args.model_dir, f"{model_name}_chexpert.pth")
            if os.path.exists(alt_path):
                weights_path = alt_path
            else:
                print(f"Skipping {model_name}: weights not found at {weights_path} or {alt_path}")
                continue

        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        val_predictions, val_labels = evaluate_dataset(model, val_loader, device)
        val_metrics = compute_metrics(val_predictions, val_labels)
        print_metrics_table(val_metrics, title="Validation metrics")

        test_predictions, _ = evaluate_dataset(model, test_loader, device)
        test_metrics = compute_metrics(test_predictions, test_ground_truth)
        print_metrics_table(test_metrics, title="Test metrics")

        results_summary[model_name] = {
            "validation": val_metrics,
            "test": test_metrics,
        }

    if not results_summary:
        raise RuntimeError("No models were evaluated successfully.")

    save_metrics_csv(results_summary, args.output_dir)


if __name__ == "__main__":
    main()
