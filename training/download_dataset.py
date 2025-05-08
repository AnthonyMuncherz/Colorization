import os
import argparse
import urllib.request
import zipfile
import shutil
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def main(args):
    # Create data directory
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    
    # Available datasets
    datasets = {
        'coco_small': {
            'url': 'https://imagecolorization.s3.amazonaws.com/coco_small_sample.zip',
            'size': '~25MB'
        },
        'landscape': {
            'url': 'https://imagecolorization.s3.amazonaws.com/landscape_sample.zip',
            'size': '~20MB'
        },
        'portrait': {
            'url': 'https://imagecolorization.s3.amazonaws.com/portrait_sample.zip',
            'size': '~15MB'
        }
    }
    
    # Check if dataset is valid
    if args.dataset not in datasets:
        print(f"Error: Dataset '{args.dataset}' not found. Available options:")
        for name, info in datasets.items():
            print(f"  - {name} ({info['size']})")
        return

    dataset_info = datasets[args.dataset]
    url = dataset_info['url']
    
    # Download dataset
    print(f"Downloading {args.dataset} dataset from {url}")
    zip_path = os.path.join(data_dir, f"{args.dataset}.zip")
    
    try:
        download_url(url, zip_path)
        
        # Extract
        print(f"Extracting {zip_path} to {data_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Clean up
        if not args.keep_zip:
            os.remove(zip_path)
            print(f"Removed {zip_path}")
            
        print(f"Dataset downloaded and extracted successfully to {data_dir}")
        
    except Exception as e:
        print(f"Error downloading or extracting dataset: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description='Download a dataset for image colorization training')
    
    parser.add_argument('--dataset', type=str, default='coco_small',
                        help='Dataset to download (coco_small, landscape, portrait)')
    parser.add_argument('--data_dir', type=str, default='training/data',
                        help='Directory to save the dataset')
    parser.add_argument('--keep_zip', action='store_true',
                        help='Keep the zip file after extraction')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args) 