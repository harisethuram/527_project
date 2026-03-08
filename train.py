import torch
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
from datasets import load_from_disk
import numpy as np
import csv
import json
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

# from src.data.chexpert import ChexpertDataset
from src.model.CNN import ChexpertCNN
from src.model.CLIP import ChexpertCLIP
from src.model.transformer import ChexpertTransformer

TARGET_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

def collate_fn(batch):
    images = torch.stack([b["image_u8"] for b in batch])   # uint8 CHW
    labels = torch.stack([b["labels"] for b in batch])
    return images, labels

def normalize_images(images, device):
    images = images.to(device, non_blocking=True).float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (images - mean) / std

def load_test_labels_array(csv_path):
    labels = []
    with open(csv_path, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            labels.append(
                [
                    float(row["Atelectasis"]),
                    float(row["Cardiomegaly"]),
                    float(row["Consolidation"]),
                    float(row["Edema"]),
                    float(row["Pleural Effusion"]),
                ]
            )
    if not labels:
        raise RuntimeError("No rows found in test labels CSV.")
    return np.asarray(labels, dtype=np.float32)

def compute_metrics(logits, labels):
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    split_metrics = {}
    auroc_values = []
    auprc_values = []

    for index, condition in enumerate(TARGET_LABELS):
        pred = probabilities[:, index]
        gt = labels[:, index]

        valid_mask = gt >= 0
        pred_valid = pred[valid_mask]
        gt_valid = gt[valid_mask]

        if len(gt_valid) == 0 or len(np.unique(gt_valid)) < 2:
            split_metrics[condition] = {
                "auroc": None,
                "auprc": None,
                "n_samples": int(len(gt_valid)),
                "n_positive": int(np.sum(gt_valid == 1.0)),
            }
            continue

        auroc = float(roc_auc_score(gt_valid, pred_valid))
        precision, recall, _ = precision_recall_curve(gt_valid, pred_valid)
        auprc = float(auc(recall, precision))

        split_metrics[condition] = {
            "auroc": auroc,
            "auprc": auprc,
            "n_samples": int(len(gt_valid)),
            "n_positive": int(np.sum(gt_valid == 1.0)),
        }
        auroc_values.append(auroc)
        auprc_values.append(auprc)

    split_metrics["average"] = {
        "auroc": float(np.mean(auroc_values)) if auroc_values else None,
        "auprc": float(np.mean(auprc_values)) if auprc_values else None,
    }
    return split_metrics

def evaluate_split(model, dataloader, device, labels_override=None):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = normalize_images(images, device)
            outputs = model(images)
            all_logits.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    if labels_override is not None:
        if len(labels_override) != len(logits):
            raise RuntimeError(
                f"labels_override length {len(labels_override)} does not match predictions length {len(logits)}"
            )
        labels = labels_override
    return compute_metrics(logits, labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the Chexpert dataset")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "clip", "transformer"], help="Model architecture to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--output_dir", type=str, default="models/h_param/", help="Directory to save the trained model")
    parser.add_argument("--data_dir", type=str, default="data", help="Base data directory containing train/validation/test")
    parser.add_argument("--test_labels_csv", type=str, default="chexlocalize/chexlocalize/CheXpert/test_labels.csv", help="CSV path with test labels")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Batch size for split evaluation (defaults to --batch_size)")
    args = parser.parse_args()
    print(args)
    # Load the dataset
    print("Loading the dataset...")
    train_dataset = load_from_disk(os.path.join(args.data_dir, "train"))
    validation_dataset = load_from_disk(os.path.join(args.data_dir, "validation"))
    test_dataset = load_from_disk(os.path.join(args.data_dir, "test"))

    train_dataset.set_format(type="torch", columns=["image_u8", "labels"])
    validation_dataset.set_format(type="torch", columns=["image_u8", "labels"])
    test_dataset.set_format(type="torch", columns=["image_u8", "labels"])

    num_workers = min(16, (os.cpu_count() or 8))  # tune: 8/16/32
    print(f"Using {num_workers} workers for data loading.")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )

    eval_batch_size = args.eval_batch_size or args.batch_size
    train_eval_dataloader = DataLoader(
        train_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )
    validation_eval_dataloader = DataLoader(
        validation_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )
    test_eval_dataloader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )

    
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    # Initialize the model
    print("Initializing the model...")
    if args.model == "cnn":
        model = ChexpertCNN()
    elif args.model == "clip":
        model = ChexpertCLIP()
    elif args.model == "transformer":
        model = ChexpertTransformer()
    
    model.to(DEVICE)
    # Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)    
    
    def masked_bce_loss(outputs, labels):
        valid_mask = (labels != -1.0).float()
        # use reduction='none' to apply mask
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels.clamp(min=0.0), reduction='none')
        # mean over valid entries (avoid division by zero if all missing)
        return (loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
    # import time
    # it = iter(train_dataloader)
    # for step in range(50):
    #     t0 = time.perf_counter()
    #     batch = next(it)
    #     t1 = time.perf_counter()
    #     print(step, "fetch", t1-t0)
        
    train_losses = []
    validation_losses = []
    print("Starting training...")
    for epoch in tqdm(range(args.epochs), desc="Training epochs"):
        model.train()
        train_loss = 0.0
        num_train_updates = 0
        
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for images, labels in train_pbar:
            images = normalize_images(images, DEVICE)
            
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = masked_bce_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if num_train_updates % 50 == 0:
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': train_loss / (train_pbar.n + 1)})
            num_train_updates += 1
            # break
            
        
        # Validate the model
        model.eval()
        validation_loss = 0.0
        num_val_updates = 0
        val_pbar = tqdm(validation_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for images, labels in val_pbar:
                images = normalize_images(images, DEVICE)
                
                labels = labels.to(DEVICE, non_blocking=True)
                outputs = model(images)
                loss = masked_bce_loss(outputs, labels)
                if num_val_updates % 20 == 0:
                    val_pbar.set_postfix({'loss': loss.item()})
                    validation_loss += loss.item()
                num_val_updates += 1
                # break
        
        print(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {validation_loss/num_val_updates}")
        train_losses.append(train_loss/num_train_updates)
        validation_losses.append(validation_loss/num_val_updates)
        
        
        
    # save the model
    print(f"Saving the model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.model}_chexpert.pth"))

    # Build test labels from CSV (test split labels source)
    print("Loading test labels for test split metrics...")
    test_labels_override = load_test_labels_array(args.test_labels_csv)
    if len(test_labels_override) != len(test_dataset):
        raise RuntimeError(
            f"Test labels count ({len(test_labels_override)}) does not match processed test dataset size ({len(test_dataset)})."
        )

    print("Evaluating AUROC/AUPRC on train split...")
    train_metrics = evaluate_split(model, train_eval_dataloader, DEVICE)
    print("Evaluating AUROC/AUPRC on validation split...")
    validation_metrics = evaluate_split(model, validation_eval_dataloader, DEVICE)
    print("Evaluating AUROC/AUPRC on test split...")
    test_metrics = evaluate_split(model, test_eval_dataloader, DEVICE, labels_override=test_labels_override)
    
    # save the model architecture and training details and metrics in a json file
    model_info = {
        "model": args.model,
        "batch_size": args.batch_size,
        "eval_batch_size": eval_batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "validation_loss": validation_loss/len(validation_dataloader),
        "metrics": {
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
        }
    }
    with open(os.path.join(args.output_dir, f'{args.model}_chexpert_info.json'), "w") as f:
        json.dump(model_info, f, indent=4)
        
    # plot and save the training and validation loss curves
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(train_losses, label="Train Loss")
    # plt.plot(validation_losses, label="Validation Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title(f"{args.model} Training and Validation Loss")
    # plt.legend()
    # plt.savefig(os.path.join(args.output_dir, f"{args.model}_chexpert_loss.png"))
    # plt.close()
              
