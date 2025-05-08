import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import skimage.color as sc
from skimage import img_as_float
from skimage.transform import resize
import random

class ColorizationDataset(Dataset):
    def __init__(self, data_dir, transform=None, size=256, augment=True):
        """
        ColorationDataset for training colorization models
        
        Args:
            data_dir (str): Directory containing image files
            transform (callable, optional): Optional transform to be applied on images
            size (int): Size to resize images to
            augment (bool): Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.transform = transform
        self.size = size
        self.augment = augment
        
        # Get list of image files
        self.image_paths = []
        valid_extensions = ['.jpg', '.jpeg', '.png']
        
        for root, _, files in os.walk(data_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    self.image_paths.append(os.path.join(root, file))
                    
        print(f"Found {len(self.image_paths)} images in {data_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Apply transform if provided
            if self.transform:
                img = self.transform(img)
            else:
                # Resize
                img = img.resize((self.size, self.size), Image.LANCZOS)
                # Convert to numpy
                img = np.array(img)
            
            # Apply augmentation if enabled
            if self.augment:
                img = self._apply_augmentation(img)
            
            # Convert RGB to LAB
            img_lab = sc.rgb2lab(img_as_float(img))
            
            # Split into L and AB channels
            img_l = img_lab[:, :, 0:1]
            img_ab = img_lab[:, :, 1:3]
            
            # Convert to tensors
            img_l = torch.from_numpy(img_l.transpose((2, 0, 1))).float()
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
            
            return {"L": img_l, "ab": img_ab, "path": img_path}
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder in case of error
            placeholder = torch.zeros((1, self.size, self.size))
            return {"L": placeholder, "ab": torch.zeros((2, self.size, self.size)), "path": img_path}
    
    def _apply_augmentation(self, img):
        """Apply data augmentation to the image"""
        # Convert PIL to numpy if needed
        if isinstance(img, Image.Image):
            img = np.array(img)
            
        # Random horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            
        # Random crop and resize
        if random.random() > 0.5:
            h, w, _ = img.shape
            min_dim = min(h, w)
            crop_size = random.uniform(0.8, 1.0) * min_dim
            top = random.randint(0, h - crop_size) if h > crop_size else 0
            left = random.randint(0, w - crop_size) if w > crop_size else 0
            
            img = img[int(top):int(top+crop_size), int(left):int(left+crop_size)]
            img = resize(img, (self.size, self.size), anti_aliasing=True, preserve_range=True).astype(np.uint8)
        
        return img


def get_data_loaders(data_dir, batch_size=16, val_split=0.1, num_workers=4, size=256):
    """
    Create and return training and validation data loaders
    
    Args:
        data_dir (str): Directory containing image data
        batch_size (int): Batch size for training
        val_split (float): Fraction of data to use for validation
        num_workers (int): Number of worker threads for data loading
        size (int): Size to resize images to
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    
    # Create dataset
    dataset = ColorizationDataset(data_dir, transform=None, size=size)
    
    # Split into train and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 