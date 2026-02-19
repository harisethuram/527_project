from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("danjacobellis/chexpert")

from PIL import Image
import numpy as np

# save all images as a torch tensor locally
import torch
images = []
for i in tqdm(range(len(dataset["train"])), total=len(dataset["train"])):
    image = np.array(dataset["train"][i]["image"])
    images.append(image)
images = torch.tensor(images)
torch.save(images, "data/images.pt")

# create a dataloader for the dataset