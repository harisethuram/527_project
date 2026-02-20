"""
Train a CNN with a classification head on the Chexpert dataset. 
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ChexpertCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ChexpertCNN, self).__init__()
        # Load a pretrained ResNet50
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Replace the final fully connected layer
        # ResNet50's final layer is named 'fc'
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)