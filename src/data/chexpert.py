from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms

class ChexpertDataset(Dataset):
    def __init__(self, dataset=None, split="train", transform=None):
        if dataset is None:
            self.dataset = load_dataset("danjacobellis/chexpert")[split]
        else:
            self.dataset = dataset
            
        self.transform = transform
        if self.transform is None:
            # Default transforms generally used for ViT, ResNet, and CLIP inputs
            self.transform = transforms.Compose([
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # We only predict these 5 targets as per evaluation requirements
        self.target_labels = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Pleural Effusion"
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]  # PIL Image

        if self.transform:
            image = self.transform(image)

        # Labels are {0: negative, 1: positive, 2: uncertain, 3: missing}
        # Map 1 -> 1.0, 0|2 -> 0.0, 3 -> -1.0 (mask)
        extracted_labels = []
        for ailment in self.target_labels:
            val = item.get(ailment, 3) # default to missing if not present
            if val == 1:
                mapped_val = 1.0
            elif val == 3:
                mapped_val = -1.0
            else: # 0 or 2
                mapped_val = 0.0
            extracted_labels.append(mapped_val)

        labels = torch.tensor(extracted_labels, dtype=torch.float32)
        return image, labels
