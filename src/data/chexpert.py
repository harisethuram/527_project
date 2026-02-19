from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms

class ChexpertDataset(Dataset):
    def __init__(self, dataset=None, split="train", transform=None):
        if dataset is None:
            self.dataset = load_dataset("danjacobellis/chexpert")["train"]
        else:
            self.dataset = dataset
            
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]  # PIL Image
        label = self.dataset[idx]["label"]

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.float32)
        return image, label
