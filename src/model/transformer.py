"""
Obtain embeddings from a pretrain ViT, and use those as features for a simple classifier.
"""
import torch
import torch.nn as nn
import timm

class ChexpertTransformer(nn.Module):
    def __init__(self, num_classes=5):
        super(ChexpertTransformer, self).__init__()
        # Load a pretrained Vision Transformer
        # vit_base_patch16_224 is a common default
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        
        # Replace the final classification head
        # timm's ViT generally has the head named 'head'
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)