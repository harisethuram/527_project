"""
Get image embeddings from CLIP, and use those as features for a simple classifier.
"""
import torch
import torch.nn as nn
from transformers import CLIPVisionModel

class ChexpertCLIP(nn.Module):
    def __init__(self, num_classes=5):
        super(ChexpertCLIP, self).__init__()
        # Load a pretrained CLIP Vision model. 'openai/clip-vit-base-patch32' is a good default.
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Add a classification head on top of the pooled embedding
        in_features = self.model.config.hidden_size
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # x shape should be (batch_size, 3, 224, 224)
        outputs = self.model(pixel_values=x)
        pooled_output = outputs.pooler_output # shape (batch_size, hidden_size)
        return self.classifier(pooled_output)