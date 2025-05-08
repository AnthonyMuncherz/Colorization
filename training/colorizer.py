import os
import torch
import torch.nn as nn
import numpy as np

# Check if we're in the training directory or the main directory
if os.path.exists('model.py'):
    from model import CustomColorizationModel, load_model
else:
    from training.model import CustomColorizationModel, load_model

class CustomColorizer:
    def __init__(self, pretrained=True):
        """
        Custom colorizer model wrapper
        
        Args:
            pretrained (bool): Whether to load pretrained weights
        """
        self.model = CustomColorizationModel()
        
        if pretrained:
            # Try to find model weights
            model_paths = [
                # Try main model directory first
                os.path.join('model', 'custom_colorization_model.pth'),
                # Try relative to current file
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'custom_colorization_model.pth')
            ]
            
            loaded = False
            for path in model_paths:
                if os.path.exists(path):
                    try:
                        self.model.load_state_dict(torch.load(path, map_location='cpu'))
                        print(f"Loaded model weights from {path}")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"Error loading model weights from {path}: {e}")
            
            if not loaded:
                print("Could not load pretrained weights. Using random initialization.")
        
        self.model.eval()
    
    def to(self, device):
        """Move model to device"""
        self.model = self.model.to(device)
        return self
    
    def cuda(self):
        """Move model to CUDA"""
        self.model = self.model.cuda()
        return self
    
    def eval(self):
        """Set model to evaluation mode"""
        self.model = self.model.eval()
        return self
    
    def __call__(self, input_l):
        """Forward pass"""
        with torch.no_grad():
            return self.model(input_l)

def customcolorizer(pretrained=True):
    """Factory function for custom colorizer model"""
    model = CustomColorizer(pretrained=pretrained)
    return model

# Add an integration function for easy use
def integrate_with_app():
    """
    This function updates the app's __init__.py to include our custom colorizer
    """
    colorizers_init_path = os.path.join('..', 'colorization', 'colorizers', '__init__.py')
    
    if not os.path.exists(colorizers_init_path):
        print(f"Could not find {colorizers_init_path}")
        return False
    
    # Read current __init__.py content
    with open(colorizers_init_path, 'r') as f:
        content = f.read()
    
    # Define our custom import
    custom_import = "\nfrom training.colorizer import customcolorizer\n"
    
    # Check if already integrated
    if 'customcolorizer' in content:
        print("Custom colorizer already integrated with app")
        return True
    
    # Update content
    updated_content = content + custom_import
    
    # Write back to file
    with open(colorizers_init_path, 'w') as f:
        f.write(updated_content)
    
    print("Successfully integrated custom colorizer with app")
    return True

if __name__ == "__main__":
    # When run directly, try to integrate with app
    integrate_with_app() 