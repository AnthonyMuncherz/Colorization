import torch
from colorizers import *

def main():
    # Initialize the models
    colorizer_eccv16 = eccv16(pretrained=True).eval()
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    
    print("Models loaded successfully!")
    return colorizer_eccv16, colorizer_siggraph17

if __name__ == '__main__':
    main() 