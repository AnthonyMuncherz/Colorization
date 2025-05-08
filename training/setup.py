import os
import argparse
import subprocess
import sys

def setup_environment(args):
    """Set up the training environment"""
    print("Setting up the training environment...")
    
    # Create necessary directories
    dirs = [
        'data',
        'checkpoints',
        'logs',
        'samples',
        'evaluation',
        'pretrained'
    ]
    
    for d in dirs:
        os.makedirs(os.path.join(os.path.dirname(__file__), d), exist_ok=True)
        print(f"Created directory: {d}")
    
    # Install requirements
    if args.install_deps:
        print("\nInstalling required dependencies...")
        requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
        
        if not os.path.exists(requirements_path):
            print(f"Error: {requirements_path} not found")
            return False
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
            print("Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
            return False
    
    # Download sample data
    if args.download_data:
        print("\nDownloading sample dataset...")
        try:
            download_script = os.path.join(os.path.dirname(__file__), 'download_dataset.py')
            dataset_arg = f'--dataset {args.dataset}' if args.dataset else ''
            
            subprocess.check_call(f'{sys.executable} "{download_script}" {dataset_arg}', shell=True)
            print("Sample dataset downloaded successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading sample dataset: {e}")
            return False
    
    print("\nEnvironment setup completed successfully!")
    print("\nTo train a model run:")
    print("python train.py --data_dir training/data\n")
    
    return True

def parse_args():
    parser = argparse.ArgumentParser(description='Set up the training environment')
    
    parser.add_argument('--install_deps', action='store_true',
                        help='Install dependencies from requirements.txt')
    parser.add_argument('--download_data', action='store_true',
                        help='Download sample training data')
    parser.add_argument('--dataset', type=str, default='coco_small',
                        help='Dataset to download (coco_small, landscape, portrait)')
    parser.add_argument('--all', action='store_true',
                        help='Perform all setup steps')
    
    args = parser.parse_args()
    
    # If --all is specified, enable all options
    if args.all:
        args.install_deps = True
        args.download_data = True
    
    return args

if __name__ == "__main__":
    args = parse_args()
    setup_environment(args) 