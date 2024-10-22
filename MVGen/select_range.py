import os
import shutil
from datetime import datetime

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

def process_all_scenes(source_dir, target_numbers, mv_images_dir):
    """
    Processes all scene folders in the output directory, filters images, and saves them to mv_images_dir.

    Args:
        output_dir (str): Path to the outputs directory containing results--<timestamp> folders.
        target_numbers (list): List of target image numbers (e.g., [0, 36, 72, ...]).
        mv_images_dir (str): Path to the generate_mvimages folder where the filtered images will be saved.
    """

    for scene_folder in os.listdir(source_dir):
        scene_path = os.path.join(source_dir, scene_folder, 'images')
        if os.path.exists(scene_path):
                # Construct the corresponding output folder in generate_mvimages
                length = len(target_numbers)
                scene_output_folder = os.path.join(mv_images_dir, f"{source_dir.split('/')[2]}",f"{scene_folder}_{length}", 'images')
                filter_images_by_number(scene_path, target_numbers, scene_output_folder)

# Example usage:
target_image_numbers = [i for i in range(0, 719, 36)]  # Assuming 720 images, filtering every 36th image
source_dir = './outputs/results--20241018-224153'  # Path to the source path directory
mv_images_dir = '../generate_mvimages'  # Path to the generate_mvimages folder
process_all_scenes(source_dir, target_image_numbers, mv_images_dir)
