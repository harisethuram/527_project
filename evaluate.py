import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.data.chexpert import ChexpertDataset
from src.model.CNN import ChexpertCNN
from src.model.CLIP import ChexpertCLIP
from src.model.transformer import ChexpertTransformer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the Chexpert dataset")
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "clip", "transformer"], help="Model architecture used")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model .pth file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    args = parser.parse_args()

    # Load the validation dataset (pseudo-test set)
    test_dataset = ChexpertDataset(split="validation")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model
    if args.model == "cnn":
        model = ChexpertCNN()
    elif args.model == "clip":
        model = ChexpertCLIP()
    elif args.model == "transformer":
        model = ChexpertTransformer()
        
    # Load weights
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path {args.model_path} does not exist.")
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # We will collect all predictions and labels
    all_preds = []
    all_labels = []
    
    print(f"Evaluating model {args.model_path} on {len(test_dataset)} samples...")
    eval_pbar = tqdm(test_dataloader, desc="Evaluating")
    with torch.no_grad():
        for images, labels in eval_pbar:
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    ailments = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion"
    ]
    
    # Calculate AUROC for each ailment
    print("\n" + "="*40)
    print(f"{'Pathology':<20} | {'AUROC':<10}")
    print("="*40)
    
    aucs = []
    for i, ailment in enumerate(ailments):
        # Extract labels and predictions for this specific ailment
        labels_i = all_labels[:, i]
        preds_i = all_preds[:, i]
        
        # Mask out missing labels (-1.0)
        valid_mask = labels_i != -1.0
        valid_labels = labels_i[valid_mask]
        valid_preds = preds_i[valid_mask]
        
        if len(valid_labels) == 0 or len(np.unique(valid_labels)) < 2:
            print(f"{ailment:<20} | N/A (Not enough classes)")
            continue
            
        auc = roc_auc_score(valid_labels, valid_preds)
        aucs.append(auc)
        print(f"{ailment:<20} | {auc:.4f}")
        
    print("-" * 40)
    if aucs:
        mean_auc = np.mean(aucs)
        print(f"{'Mean AUROC':<20} | {mean_auc:.4f}")
    print("="*40)
