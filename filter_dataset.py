"""
    Author: Taylor J. Brown
    Date: 23APR24
    Orginization: Intelligent Systems Lab (ISL) at the University of Fayetteville
    Project: SAR Image Segmentation for IMPACT 1
"""

import os
from PIL import Image
from tqdm import tqdm
import shutil
from data_point_collector import count_pixels

def filter_dataset(thresholds, iteration):
    """
    Filter the dataset based on the threshold value.
    Creates directories and filters images based on urban and peatland percentages.
    """
    make_filtered_directories(iteration)  # Create directories for filtered images
    base_path = os.getcwd()  # Get the current working directory
    # Determine the path to the images to be filtered
    if iteration == 1:
        split_images_path = os.path.join(base_path, "split_images")
    else:
        split_images_path = os.path.join(base_path, f"filtered_images{iteration-1}")
        
    filtered_images_path = os.path.join(base_path, f"filtered_images{iteration}")

    # Process each folder in the split images directory
    for folder in os.listdir(split_images_path):
        if folder.endswith("Mask"):
            folder_path = os.path.join(split_images_path, folder)
            with tqdm(total=len(os.listdir(folder_path)), desc=f"Filtering {folder:<10}", unit='img', bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}', dynamic_ncols=True) as pbar:
                for image in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image)
                    class_counts = count_pixels(image_path)  # Get the pixel counts for classes
                    total_pixels = sum(class_counts.values())
                    urban_percentage = class_counts["urban"] / total_pixels  # Calculate urban coverage
                    peatland_percentage = class_counts["peatland"] / total_pixels  # Calculate peatland coverage

                    # Filter based on threshold values for urban and peatland percentages
                    if urban_percentage >= thresholds[iteration-1] or peatland_percentage >= thresholds[iteration-1]:
                        move_corrisponding_sar(image, folder.replace("Mask","SAR"), iteration)  # Move corresponding SAR image
                        shutil.copy(image_path, os.path.join(filtered_images_path, folder, image))  # Copy the image to the new location
                                
                    pbar.update(1)  # Update progress bar for each image processed

    
def make_filtered_directories(iteration):
    """
    Create directories for filtered images for each iteration.
    """
    cwd = os.getcwd()  # Get the current working directory
    # Create necessary directories for storing filtered images
    os.makedirs(os.path.join(cwd, f'filtered_images{iteration}', 'train_SAR'), exist_ok=True)
    os.makedirs(os.path.join(cwd, f'filtered_images{iteration}', 'val_SAR'), exist_ok=True)
    os.makedirs(os.path.join(cwd, f'filtered_images{iteration}', 'test_SAR'), exist_ok=True)
    os.makedirs(os.path.join(cwd, f'filtered_images{iteration}', 'train_Mask'), exist_ok=True)
    os.makedirs(os.path.join(cwd, f'filtered_images{iteration}', 'val_Mask'), exist_ok=True)
    os.makedirs(os.path.join(cwd, f'filtered_images{iteration}', 'test_Mask'), exist_ok=True)


def move_corrisponding_sar(mask_img_name, category, iteration):
    """
    Move corresponding SAR images to the filtered directory based on the mask image name and category.
    """
    base_path = os.getcwd()  # Get the current working directory
    split_images_path = os.path.join(base_path, "split_images", category)
    # Copy the SAR image to the corresponding filtered directory
    shutil.copy(os.path.join(split_images_path, mask_img_name), os.path.join(base_path, f"filtered_images{iteration}", category, mask_img_name))

    
def initiate_filter(thresholds=[0.1, 0.12, 0.14, 0.16]):
    """
    Initiate the filtering of the dataset based on the threshold value.
    Iterates through four threshold levels, filtering the dataset each time.
    """
    # Get the current working directory
    cwd = os.getcwd()

    # Filter the dataset based on the threshold value
    for i in range(0, 4):
        filter_dataset(thresholds, i+1)
        print(f"Dataset filtered for threshold {thresholds[i]}.\n")
    

def main():
    initiate_filter()  # Start the filtering process with default thresholds
    print("Dataset filtered successfully.")


if __name__ == "__main__":
    main()
