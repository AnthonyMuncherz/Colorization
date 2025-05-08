import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from skimage.color import lab2rgb
from skimage import img_as_ubyte

# Import custom modules
from model import CustomColorizationModel, save_model, load_model
from dataset import get_data_loaders

def lab_to_rgb(L, ab):
    """Convert LAB tensor to RGB numpy array"""
    L = L.detach().cpu().numpy()[0, 0, :, :]  # [0, 100]
    ab = ab.detach().cpu().numpy()[0, :, :, :]  # [-110, 110]
    
    # Transpose to HxWxC format
    ab = np.transpose(ab, (1, 2, 0))
    
    # Combine channels
    lab = np.concatenate([L[:, :, np.newaxis], ab], axis=2)
    
    # Convert to RGB
    rgb = lab2rgb(lab)
    
    return img_as_ubyte(rgb)

def train_model(args):
    # Set device
    device = torch.device(args.device)
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up tensorboard writer
    log_dir = os.path.join(args.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create model
    model = CustomColorizationModel().to(device)
    
    # Load pretrained weights if specified
    if args.pretrained:
        if load_model(model, args.pretrained):
            print(f"Loaded pretrained model from {args.pretrained}")
        else:
            print(f"Failed to load pretrained model from {args.pretrained}")
    
    # Loss function 
    criterion = nn.L1Loss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        size=args.image_size
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        
        train_loss = 0.0
        train_samples = 0
        
        for i, batch in enumerate(train_loader):
            # Get data
            L = batch["L"].to(device)
            ab = batch["ab"].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            output = model(L)
            loss = criterion(output, ab)
            loss.backward()
            optimizer.step()
            
            # Statistics
            batch_size = L.size(0)
            train_loss += loss.item() * batch_size
            train_samples += batch_size
            
            # Print status
            if i % args.print_freq == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # Calculate average training loss
        train_loss = train_loss / train_samples
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                L = batch["L"].to(device)
                ab = batch["ab"].to(device)
                
                output = model(L)
                loss = criterion(output, ab)
                
                batch_size = L.size(0)
                val_loss += loss.item() * batch_size
                val_samples += batch_size
                
        val_loss = val_loss / val_samples
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, os.path.join(args.checkpoint_dir, f"best_model.pth"))
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_model(model, os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pth"))
        
        # Log losses
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Generate and log sample images
        if (epoch + 1) % args.sample_freq == 0:
            with torch.no_grad():
                # Get a sample batch
                sample_batch = next(iter(val_loader))
                sample_L = sample_batch["L"].to(device)
                sample_ab_true = sample_batch["ab"].to(device)
                sample_ab_pred = model(sample_L)
                
                # Convert to RGB and save
                for j in range(min(3, sample_L.size(0))):
                    # True RGB
                    true_rgb = lab_to_rgb(sample_L[j:j+1], sample_ab_true[j:j+1])
                    
                    # Predicted RGB
                    pred_rgb = lab_to_rgb(sample_L[j:j+1], sample_ab_pred[j:j+1])
                    
                    # Gray (L channel only)
                    gray = np.repeat(sample_L[j, 0].cpu().numpy()[:, :, np.newaxis] / 100.0, 3, axis=2)
                    gray = img_as_ubyte(gray)
                    
                    # Create comparison image
                    comparison = np.concatenate([gray, pred_rgb, true_rgb], axis=1)
                    
                    # Save image
                    sample_path = os.path.join(
                        args.output_dir, 
                        f"epoch_{epoch+1}_sample_{j+1}.png"
                    )
                    plt.imsave(sample_path, comparison)
                    
                    # Add to tensorboard
                    writer.add_image(
                        f'Sample {j+1}',
                        comparison.transpose(2, 0, 1),
                        epoch,
                        dataformats='CHW'
                    )
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Save final model
    save_model(model, os.path.join(args.checkpoint_dir, "final_model.pth"))
    
    # Copy best model to the main model directory
    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    final_model_path = os.path.join("..", "model", "custom_colorization_model.pth")
    
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy2(best_model_path, final_model_path)
        print(f"Copied best model to {final_model_path}")
    
    print("Training completed!")
    return model

def parse_args():
    parser = argparse.ArgumentParser(description='Train a colorization model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='training/data', 
                        help='Directory containing training images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size to resize images to')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of data to use for validation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Output parameters
    parser.add_argument('--checkpoint_dir', type=str, default='training/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='training/logs',
                        help='Directory to save logs')
    parser.add_argument('--output_dir', type=str, default='training/samples',
                        help='Directory to save sample outputs')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save frequency in epochs')
    parser.add_argument('--sample_freq', type=int, default=1,
                        help='Sample generation frequency in epochs')
    
    # Model parameters
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model weights')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    train_model(args) 