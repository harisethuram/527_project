import torch
from torch.utils.data import DataLoader
import argparse
import os

from src.data.chexpert import ChexpertDataset
from src.models.CNN import ChexpertCNN
from src.models.CLIP import ChexpertCLIP
from src.models.transformer import ChexpertTransformer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the Chexpert dataset")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "clip", "transformer"], help="Model architecture to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--output_dir", type=str, default="models/", help="Directory to save the trained model")
    args = parser.parse_args()

    # Load the dataset
    train_dataset = ChexpertDataset(split="train")
    validation_dataset = ChexpertDataset(split="validation")
    test_dataset = ChexpertDataset(split="test")
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validate the model
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for images, labels in validation_dataloader:
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                validation_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {validation_loss/len(validation_dataloader)}")
        
    # Test the model
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_dataloader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
    print(f"Test Loss: {test_loss/len(test_dataloader)}")
    
    # save the model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{args.output_dir}{args.model}_chexpert.pth")