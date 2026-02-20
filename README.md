# CheXpert Deep Learning Models

We train and evaluate the following architectures on the CheXpert dataset:
* ResNet50 (CNN)
* Vision Transformer (ViT)
* CLIP (Vision Encoder)

The pipeline evaluates models based on the **AUROC** performance over 5 target pathologies: Atelectasis, Cardiomegaly, Consolidation, Edema, and Pleural Effusion.

## Installation

Ensure you have a Conda environment with Python 3.9+ and install the dependencies:
```bash
pip install torch torchvision transformers timm datasets scikit-learn
```

## Training

Train a model using `train.py`. The script will automatically download the dataset from HuggingFace on the first run.
Available models: `cnn`, `transformer`, `clip`.

```bash
python3 train.py --model cnn --batch_size 32 --epochs 10 --lr 1e-4
```

Models and their checkpoints are saved to the `models/` directory using the hyperparameters in the filename (e.g., `models/cnn_epochs10_bs32_lr0.0001_chexpert.pth`).

## Evaluation

Evaluate a saved `.pth` file on the CheXpert validation set using `evaluate.py`. This calculates AuROC for the 5 target ailments, filtering out any missing `Unlabeled` targets as per standard metrics.

```bash
python evaluate.py --model cnn --model_path models/cnn_epochs10_bs32_lr0.0001_chexpert.pth --batch_size 32
```