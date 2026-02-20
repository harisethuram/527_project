import torch
from src.model.CNN import ChexpertCNN
from src.model.CLIP import ChexpertCLIP
from src.model.transformer import ChexpertTransformer

batch_size = 2
dummy_input = torch.randn(batch_size, 3, 224, 224)

print("Testing CNN...")
cnn_model = ChexpertCNN()
cnn_out = cnn_model(dummy_input)
print(f"CNN output shape: {cnn_out.shape} (Expected: {batch_size}, 5)")

print("Testing Transformer...")
vit_model = ChexpertTransformer()
vit_out = vit_model(dummy_input)
print(f"Transformer output shape: {vit_out.shape} (Expected: {batch_size}, 5)")

print("Testing CLIP...")
clip_model = ChexpertCLIP()
clip_out = clip_model(dummy_input)
print(f"CLIP output shape: {clip_out.shape} (Expected: {batch_size}, 5)")
