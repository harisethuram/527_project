"""
Plot prediction score histograms for CheXpert evaluation.

For each of the five CheXpert competition labels, this script plots
two overlapping histograms:

    red   = scores for ground-truth positives
    green = scores for ground-truth negatives

Example:
    python plot_score_histograms.py \
        --data_dir data/test \
        --labels_csv data/test/test_labels.csv \
        --output_path score_histograms.png
"""

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchxrayvision as xrv


# CheXpert competition labels
EVAL_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


def collate_fn(batch):
    images = torch.stack([b["image_u8"] for b in batch])
    return images


def preprocess_batch_for_model(images, device):
    images = images.to(device).float()

    if images.shape[1] == 3:
        images = images[:, :1]

    # TorchXRayVision normalization
    images = (images / 255.0) * 2048.0 - 1024.0

    return images


def align_labels(df, model):
    model_labels = model.targets if hasattr(model, "targets") else model.pathologies

    label_to_index = {}
    for label in EVAL_LABELS:

        if label == "Pleural Effusion":
            if "Effusion" in model_labels:
                label_to_index[label] = model_labels.index("Effusion")
            else:
                label_to_index[label] = model_labels.index("Pleural Effusion")

        else:
            label_to_index[label] = model_labels.index(label)

    return label_to_index


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/test")
    parser.add_argument("--labels_csv", type=str, default="data/test/test_labels.csv")
    parser.add_argument("--output_path", type=str, default="score_histograms.png")
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    model.to(device)
    model.eval()

    dataset = load_from_disk(args.data_dir)
    dataset.set_format(type="torch", columns=["image_u8"])

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    labels_df = pd.read_csv(args.labels_csv)

    label_to_index = align_labels(labels_df, model)

    pos_scores = {k: [] for k in EVAL_LABELS}
    neg_scores = {k: [] for k in EVAL_LABELS}

    sample_index = 0

    print("Running inference...")

    for images in tqdm(dataloader):

        batch_size = images.shape[0]

        images = preprocess_batch_for_model(images, device)

        with torch.no_grad():
            outputs = model(images)
            probs = torch.sigmoid(outputs)

        for i in range(batch_size):

            row = labels_df.iloc[sample_index]

            for label in EVAL_LABELS:

                idx = label_to_index[label]
                score = probs[i, idx].item()
                gt = int(row[label])

                if gt == 1:
                    pos_scores[label].append(score)
                else:
                    neg_scores[label].append(score)

            sample_index += 1

    print("Plotting histograms...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, label in enumerate(EVAL_LABELS):

        ax = axes[i]

        ax.hist(
            neg_scores[label],
            bins=50,
            alpha=0.6,
            color="green",
            label="Negative"
        )

        ax.hist(
            pos_scores[label],
            bins=50,
            alpha=0.6,
            color="red",
            label="Positive"
        )

        ax.set_title(label)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Count")
        ax.legend()

    axes[-1].axis("off")

    plt.tight_layout()
    plt.savefig(args.output_path, dpi=200)
    print("Saved figure to:", args.output_path)


if __name__ == "__main__":
    main()