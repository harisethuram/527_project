import torch
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
from datasets import load_from_disk

# from src.data.chexpert import ChexpertDataset
from src.model.CNN import ChexpertCNN
from src.model.CLIP import ChexpertCLIP
from src.model.transformer import ChexpertTransformer

def collate_fn(batch):
    images = torch.stack([b["image_u8"] for b in batch])   # uint8 CHW
    labels = torch.stack([b["labels"] for b in batch])
    return images, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the Chexpert dataset")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "clip", "transformer"], help="Model architecture to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--output_dir", type=str, default="models/", help="Directory to save the trained model")
    args = parser.parse_args()
    print(args)
    # Load the dataset
    print("Loading the dataset...")
    train_dataset = load_from_disk("data/train")
    validation_dataset = load_from_disk("data/validation")

    train_dataset.set_format(type="torch", columns=["image_u8", "labels"])
    validation_dataset.set_format(type="torch", columns=["image_u8", "labels"])

    num_workers = min(16, (os.cpu_count() or 8))  # tune: 8/16/32
    print(f"Using {num_workers} workers for data loading.")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
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
            images = images.to(DEVICE, non_blocking=True).float().div_(255.0)
            mean = torch.tensor([0.485,0.456,0.406], device=DEVICE).view(1,3,1,1)
            std  = torch.tensor([0.229,0.224,0.225], device=DEVICE).view(1,3,1,1)
            images = (images - mean) / std
            
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
                images = images.to(DEVICE, non_blocking=True).float().div_(255.0)
                mean = torch.tensor([0.485,0.456,0.406], device=DEVICE).view(1,3,1,1)
                std  = torch.tensor([0.229,0.224,0.225], device=DEVICE).view(1,3,1,1)
                images = (images - mean) / std
                
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
        
        
        
    # Test the model
    # model.eval()
    # test_loss = 0.0
    # print("Testing the model...")
    # with torch.no_grad():
    #     test_pbar = tqdm(test_dataloader, desc="Testing", leave=False)
    #     for images, labels in test_pbar:
    #         images = images.to(DEVICE, non_blocking=True)
    #         labels = labels.to(DEVICE, non_blocking=True)
    #         outputs = model(images)
    #         loss = masked_bce_loss(outputs, labels)
    #         test_loss += loss.item()
    #         test_pbar.set_postfix({'loss': test_loss / (test_pbar.n + 1)})
    # print(f"Test Loss: {test_loss/len(test_dataloader)}")
    
    # save the model
    print(f"Saving the model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.model}_chexpert.pth"))
    
    # save the model architecture and training details and loss in a json file
    import json
    model_info = {
        "model": args.model,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "validation_loss": validation_loss/len(validation_dataloader),
        # "test_loss": test_loss/len(test_dataloader)
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
              
