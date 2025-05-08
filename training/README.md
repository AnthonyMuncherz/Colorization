# Image Colorization Model Training

This directory contains scripts and modules for training custom image colorization models using deep learning.

## Directory Structure

- `model.py` - Custom colorization model architecture
- `dataset.py` - Data loading and preprocessing utilities
- `train.py` - Main training script
- `evaluate.py` - Evaluation script for trained models
- `colorizer.py` - Integration with the main application
- `download_dataset.py` - Utility to download sample datasets

## Getting Started

### Prerequisites

Ensure you have all required packages installed:

```bash
pip install torch torchvision scikit-image matplotlib tensorboard tqdm
```

### Download a Sample Dataset

To quickly get started, you can download a sample dataset:

```bash
python download_dataset.py --dataset coco_small
```

This will download a small subset of the COCO dataset to the `training/data` directory.

Available datasets:
- `coco_small`: Small sample of COCO dataset
- `landscape`: Sample landscape images
- `portrait`: Sample portrait images

### Training a Model

To train a model using default parameters:

```bash
python train.py --data_dir training/data
```

#### Custom Training Options

You can customize the training process with various options:

```bash
python train.py --data_dir training/data --batch_size 16 --epochs 30 --lr 2e-4
```

Key options:
- `--batch_size`: Batch size for training (default: 16)
- `--epochs`: Number of training epochs (default: 30)
- `--lr`: Learning rate (default: 2e-4)
- `--device`: Device to use (default: 'cuda' if available, else 'cpu')
- `--pretrained`: Path to pretrained model weights (optional)

### Evaluating the Model

After training, you can evaluate your model on test images:

```bash
python evaluate.py --data_dir path/to/test/images --model_path training/checkpoints/best_model.pth
```

This will generate:
- Comparison images showing original grayscale, colorized, and ground truth
- SSIM and PSNR metrics for each image
- A summary of evaluation results

### Integrating with the Main Application

The trained model can be integrated with the main colorization application:

```bash
python colorizer.py
```

This will update the main application's imports to include your custom colorizer model.

## Using Your Trained Model

Once training is complete, the best model will be automatically copied to:
`../model/custom_colorization_model.pth`

This allows the main application to access your custom model alongside the 
pre-trained ECCV16 and SIGGRAPH17 models.

## Examples

### Full Training Workflow

```bash
# 1. Download training data
python download_dataset.py --dataset landscape

# 2. Train model
python train.py --data_dir training/data --epochs 50 --batch_size 32

# 3. Evaluate model
python evaluate.py --data_dir training/data --model_path training/checkpoints/best_model.pth

# 4. Integrate with main app
python colorizer.py
```

## Model Architecture

Our custom colorization model uses an encoder-decoder architecture with:
- Skip connections to preserve spatial details
- Attention mechanism to focus on important features
- L1 loss for color prediction
- Adam optimizer with learning rate scheduling

The model is trained to predict the 'ab' color channels of the LAB color space from the 'L' (lightness) channel. 