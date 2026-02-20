import torch
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm

from src.data.chexpert import ChexpertDataset
from src.model.CNN import ChexpertCNN
from src.model.CLIP import ChexpertCLIP
from src.model.transformer import ChexpertTransformer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the Chexpert dataset")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "clip", "transformer"], help="Model architecture to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--output_dir", type=str, default="models/", help="Directory to save the trained model")
    args = parser.parse_args()

    # Load the dataset
    train_dataset = ChexpertDataset(split="train")
    validation_dataset = ChexpertDataset(split="validation")
    test_dataset = ChexpertDataset(split="validation") # CheXpert HuggingFace dataset only has train/validation splits
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model
    if args.model == "cnn":
        model = ChexpertCNN()
    elif args.model == "clip":
        model = ChexpertCLIP()
    elif args.model == "transformer":
        model = ChexpertTransformer()
        
    # Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)    
    
    def masked_bce_loss(outputs, labels):
        valid_mask = (labels != -1.0).float()
        # use reduction='none' to apply mask
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels.clamp(min=0.0), reduction='none')
        # mean over valid entries (avoid division by zero if all missing)
        return (loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, labels in train_pbar:
            optimizer.zero_grad()
            outputs = model(images)
            loss = masked_bce_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': train_loss / (train_pbar.n + 1)})
        
        # Validate the model
        model.eval()
        validation_loss = 0.0
        val_pbar = tqdm(validation_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for images, labels in val_pbar:
                outputs = model(images)
                loss = masked_bce_loss(outputs, labels)
                validation_loss += loss.item()
                val_pbar.set_postfix({'loss': validation_loss / (val_pbar.n + 1)})
        
        print(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {validation_loss/len(validation_dataloader)}")
        
        
    # Test the model
    model.eval()
    test_loss = 0.0
    test_pbar = tqdm(test_dataloader, desc="Testing")
    with torch.no_grad():
        for images, labels in test_pbar:
            outputs = model(images)
            loss = masked_bce_loss(outputs, labels)
            test_loss += loss.item()
            test_pbar.set_postfix({'loss': test_loss / (test_pbar.n + 1)})
    print(f"Test Loss: {test_loss/len(test_dataloader)}")
    
    # save the model
    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"{args.model}_epochs{args.epochs}_bs{args.batch_size}_lr{args.lr}_chexpert.pth"
    torch.save(model.state_dict(), os.path.join(args.output_dir, filename))
    print(f"Model saved to {os.path.join(args.output_dir, filename)}")