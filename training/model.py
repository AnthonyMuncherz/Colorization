import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import os

class CustomColorizationModel(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(CustomColorizationModel, self).__init__()
        
        # Normalize input and output values
        self.l_cent = 50.0
        self.l_norm = 100.0
        self.ab_norm = 110.0
        
        # Encoder
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(64)
        )
        
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128)
        )
        
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256)
        )
        
        self.encoder_conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512)
        )
        
        # Middle blocks
        self.middle_block = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512)
        )
        
        # Decoder with skip connections
        self.decoder_conv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256)
        )
        
        self.decoder_conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128)
        )
        
        self.decoder_conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
        )
        
        # Final output - 2 channels for ab color space
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Tanh()
        )
        
        # Attention module for improved colorization
        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 1, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )
    
    def normalize_l(self, in_l):
        return (in_l - self.l_cent) / self.l_norm

    def unnormalize_l(self, in_l):
        return in_l * self.l_norm + self.l_cent

    def normalize_ab(self, in_ab):
        return in_ab / self.ab_norm

    def unnormalize_ab(self, in_ab):
        return in_ab * self.ab_norm
    
    def forward(self, input_l):
        # Normalize input
        normalized_l = self.normalize_l(input_l)
        
        # Encoder
        enc1 = self.encoder_conv1(normalized_l)
        enc2 = self.encoder_conv2(enc1)
        enc3 = self.encoder_conv3(enc2)
        enc4 = self.encoder_conv4(enc3)
        
        # Middle block
        middle = self.middle_block(enc4)
        
        # Apply attention
        attention_map = self.attention(middle)
        attended_features = middle * attention_map
        
        # Decoder with skip connections
        dec1 = self.decoder_conv1(attended_features)
        dec2 = self.decoder_conv2(dec1 + enc3)
        dec3 = self.decoder_conv3(dec2 + enc2)
        
        # Output
        out_ab = self.output_conv(dec3 + enc1)
        
        # Unnormalize output
        return self.unnormalize_ab(out_ab)

def save_model(model, path):
    """Save model weights"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """Load model weights"""
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
        return True
    return False 