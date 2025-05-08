import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.color import lab2rgb
from skimage import img_as_ubyte
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import custom modules
from model import CustomColorizationModel, load_model
from dataset import ColorizationDataset
from torch.utils.data import DataLoader

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

def evaluate_model(args):
    # Set device
    device = torch.device(args.device)
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = CustomColorizationModel().to(device)
    if not load_model(model, args.model_path):
        print(f"Failed to load model from {args.model_path}")
        return
    
    model.eval()
    
    # Create dataset
    dataset = ColorizationDataset(args.data_dir, size=args.image_size, augment=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    # Metrics
    ssim_scores = []
    psnr_scores = []
    
    # Evaluate
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if i >= args.num_samples and args.num_samples > 0:
                break
                
            # Get data
            L = batch["L"].to(device)
            ab_true = batch["ab"].to(device)
            img_path = batch["path"][0]
            
            # Generate colorization
            ab_pred = model(L)
            
            # Convert to RGB
            true_rgb = lab_to_rgb(L, ab_true)
            pred_rgb = lab_to_rgb(L, ab_pred)
            gray = np.repeat(L[0, 0].cpu().numpy()[:, :, np.newaxis] / 100.0, 3, axis=2)
            gray = img_as_ubyte(gray)
            
            # Calculate metrics
            ssim_score = ssim(true_rgb, pred_rgb, channel_axis=2, data_range=255)
            psnr_score = psnr(true_rgb, pred_rgb, data_range=255)
            
            ssim_scores.append(ssim_score)
            psnr_scores.append(psnr_score)
            
            # Create comparison image
            comparison = np.concatenate([gray, pred_rgb, true_rgb], axis=1)
            
            # Save image
            img_name = os.path.basename(img_path)
            filename = f"{i+1}_{os.path.splitext(img_name)[0]}_comparison.png"
            output_path = os.path.join(args.output_dir, filename)
            plt.imsave(output_path, comparison)
            
            # Print metrics
            print(f"Image {i+1}: SSIM = {ssim_score:.4f}, PSNR = {psnr_score:.2f} dB")
    
    # Calculate average metrics
    avg_ssim = np.mean(ssim_scores)
    avg_psnr = np.mean(psnr_scores)
    
    print("\nEvaluation Results:")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    
    # Save metrics to file
    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Image,SSIM,PSNR\n")
        for i, (s, p) in enumerate(zip(ssim_scores, psnr_scores)):
            f.write(f"{i+1},{s:.4f},{p:.2f}\n")
        f.write(f"\nAverage,{avg_ssim:.4f},{avg_psnr:.2f}\n")
    
    print(f"Evaluation completed. Results saved to {args.output_dir}")
    
    # Create a visualization of the results
    create_results_summary(args.output_dir, ssim_scores, psnr_scores)

def create_results_summary(output_dir, ssim_scores, psnr_scores):
    """Create a summary visualization of evaluation metrics"""
    plt.figure(figsize=(12, 6))
    
    # SSIM plot
    plt.subplot(1, 2, 1)
    plt.hist(ssim_scores, bins=10, alpha=0.7)
    plt.axvline(np.mean(ssim_scores), color='r', linestyle='--', label=f'Mean: {np.mean(ssim_scores):.4f}')
    plt.xlabel('SSIM Score')
    plt.ylabel('Frequency')
    plt.title('SSIM Distribution')
    plt.legend()
    
    # PSNR plot
    plt.subplot(1, 2, 2)
    plt.hist(psnr_scores, bins=10, alpha=0.7)
    plt.axvline(np.mean(psnr_scores), color='r', linestyle='--', label=f'Mean: {np.mean(psnr_scores):.2f} dB')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Frequency')
    plt.title('PSNR Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_summary.png"))

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a colorization model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size to resize images to')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model weights')
    
    # Evaluation parameters
    parser.add_argument('--num_samples', type=int, default=0,
                        help='Number of samples to evaluate (0 for all)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for evaluation (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='training/evaluation',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args) 