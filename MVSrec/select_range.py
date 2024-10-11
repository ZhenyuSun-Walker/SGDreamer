import argparse
import os
import shutil
import sys
from datetime import datetime

# 获取当前脚本所在目录的父级目录
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# 将父级目录添加到系统路径
sys.path.append(parent_dir)


def filter_images_by_number(folder_path, target_numbers, output_folder):
    """
    Filters images in a folder based on specified target numbers and saves them to an output folder.
    
    Args:
        folder_path (str): Path to the image folder.
        target_numbers (list): List of target image numbers (e.g., [90, 95, 100, ...]).
        output_folder (str): Path to the output folder where filtered images will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist
    for filename in os.listdir(folder_path):
        try:
            image_number = int(filename.split('.')[0])  # Extract the image number from the filename
            if image_number in target_numbers:
                shutil.copy(os.path.join(folder_path, filename), os.path.join(output_folder, filename))
        except ValueError:
            pass  # Skip files that don't follow the expected naming convention

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source', 
                    type=str, default=None, help='source pano path')
    parser.add_argument('--target_numbers', 
                    type=list, help='target images', default=[i for i in range(90, 190, 5)])
    parser.add_argument('--output',
                    type=str, default='scene', help='output directiory')
    return parser.parse_args()

args = parse_args()
# Example usage:
length = len(args.target_numbers)
output_folder_path = os.path.join(f"dataset/{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.output}_{length}/images")
filter_images_by_number(args.source, args.target_numbers, output_folder_path)
