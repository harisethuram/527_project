import torch
import torch.nn as nn
from transformers import CLIPModel

class ChexpertCLIP(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        # freeze CLIP
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        in_features = self.clip.config.projection_dim  # 512
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        with torch.no_grad():
            vision_out = self.clip.vision_model(pixel_values=x)   # BaseModelOutputWithPooling
            pooled = vision_out.pooler_output                     # (B, 768)
            feats = self.clip.visual_projection(pooled)           # (B, 512)  <- CLIP image embedding

            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        return self.classifier(feats)