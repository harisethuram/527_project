#!/bin/bash

# Test script for saliency map generation
# This script tests the saliency map generation on both a single image and a dataset

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Saliency Map Generation Test Script"
echo "========================================"
echo ""

# Check if model weights exist
if [ ! -f "models/cnn_chexpert.pth" ]; then
    echo "❌ Error: CNN model weights not found at models/cnn_chexpert.pth"
    echo "   Please train the model first using: python train.py --model cnn"
    exit 1
fi

if [ ! -f "models/transformer_chexpert.pth" ]; then
    echo "❌ Error: Transformer model weights not found at models/transformer_chexpert.pth"
    echo "   Please train the model first using: python train.py --model transformer"
    exit 1
fi

# Check if validation dataset exists
if [ ! -d "data/validation" ]; then
    echo "❌ Error: Validation dataset not found at data/validation"
    echo "   Please prepare the dataset first using: python preprocess_and_save.py"
    exit 1
fi

# Test 1: Dataset Mode
echo "Test 1: Processing validation dataset (first 5 samples)"
echo "=========================================================="
python generate_saliency_maps.py \
    --models "cnn,transformer" \
    --model_dir "models/" \
    --data_dir "data/validation" \
    --output_dir "saliency_maps/" \
    --num_samples 5 \
    --batch_size 4 \
    --colormap "jet"

if [ $? -eq 0 ]; then
    echo "✓ Test 1 passed: Generated saliency maps from dataset"
    echo "  Output saved to: saliency_maps/"
    echo ""
else
    echo "❌ Test 1 failed"
    exit 1
fi

# Test 2: Single Image Mode
# Try to find a test image or create a dummy one
TEST_IMAGE="test_image.jpg"

# If no test image exists, create a simple dummy one for testing
if [ ! -f "$TEST_IMAGE" ]; then
    echo ""
    echo "Test 2: Creating dummy test image..."
    python3 << 'EOF'
import numpy as np
from PIL import Image

# Create a simple test image (224x224 RGB)
img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
Image.fromarray(img).save("test_image.jpg")
print("✓ Created dummy test image: test_image.jpg")
EOF
fi

if [ -f "$TEST_IMAGE" ]; then
    echo ""
    echo "Test 3: Processing single JPEG image"
    echo "======================================"
    python generate_saliency_maps.py \
        --models "cnn,transformer" \
        --model_dir "models/" \
        --image_path "$TEST_IMAGE" \
        --output_dir "saliency_maps_single/" \
        --colormap "jet"
    
    if [ $? -eq 0 ]; then
        echo "✓ Test 2 passed: Generated saliency maps from single image"
        echo "  Output saved to: saliency_maps_single/"
        echo ""
    else
        echo "❌ Test 2 failed"
        exit 1
    fi
else
    echo "⚠ Warning: Could not create test image"
fi

echo ""
echo "========================================"
echo "✓ All tests completed successfully!"
echo "========================================"
echo ""
echo "Output locations:"
echo "  - Dataset results: saliency_maps/"
echo "  - Single image results: saliency_maps_single/"
echo ""
echo "To visualize results, open the PNG files in an image viewer."
