"""
    Author: Taylor J. Brown
    Date: 23APR24
    Orginization: Intelligent Systems Lab (ISL) at the University of Fayetteville
    Project: SAR Image Segmentation for IMPACT 1
"""

import os
import shutil


def split_files(path, train_ratio, val_ratio, test_ratio):
    """Split files into train, validation, and test sets based on specified ratios."""
    # Check that the input ratios add up to 1.0
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
    
    # List all files in the specified directory
    files = os.listdir(path)
    n_files = len(files)
    train_end = int(n_files * train_ratio)
    val_end = int(n_files * (train_ratio + val_ratio))
    
    # Slice the file list into training, validation, and testing segments
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    
    return train_files, val_files, test_files

def make_directories():
    """Make directories for training, validation, and testing sets."""
    # Get the current working directory
    cwd = os.getcwd()
    # Create directories for training, validation, and testing sets
    os.makedirs(os.path.join(cwd, 'original_images', 'train_SAR'), exist_ok=True)
    os.makedirs(os.path.join(cwd, 'original_images', 'val_SAR'), exist_ok=True)
    os.makedirs(os.path.join(cwd, 'original_images', 'test_SAR'), exist_ok=True)
    os.makedirs(os.path.join(cwd, 'original_images', 'train_mask'), exist_ok=True)
    os.makedirs(os.path.join(cwd, 'original_images', 'val_mask'), exist_ok=True)
    os.makedirs(os.path.join(cwd, 'original_images', 'test_mask'), exist_ok=True)

    # Create directories for the split images
    os.makedirs(os.path.join(cwd, 'split_images', 'train_SAR'), exist_ok=True)
    os.makedirs(os.path.join(cwd, 'split_images', 'val_SAR'), exist_ok=True)
    os.makedirs(os.path.join(cwd, 'split_images', 'test_SAR'), exist_ok=True)
    os.makedirs(os.path.join(cwd, 'split_images', 'train_mask'), exist_ok=True)
    os.makedirs(os.path.join(cwd, 'split_images', 'val_mask'), exist_ok=True)
    os.makedirs(os.path.join(cwd, 'split_images', 'test_mask'), exist_ok=True)


def initiate_split(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """Initiate the splitting of files into training, validation, and testing directories."""
    # Make directories for the training, validation, and testing sets
    make_directories()
    # Get the current working directory
    cwd = os.getcwd()
    # Construct the path to the directory containing initial dumped files
    path = os.path.join(cwd, 'dump_sar_here')
    
    # Split the files into respective categories
    train_files, val_files, test_files = split_files(path, train_ratio, val_ratio, test_ratio)

    # Move each file to its corresponding new directory based on its category
    for file in train_files:
        shutil.move(os.path.join(path, file), 
                    os.path.join(cwd, 'original_images', 'train_SAR', file))
    for file in val_files:
        shutil.move(os.path.join(path, file), 
                    os.path.join(cwd, 'original_images', 'val_SAR', file))
    for file in test_files:
        shutil.move(os.path.join(path, file), 
                    os.path.join(cwd, 'original_images', 'test_SAR', file))


def move_corresponding_masks():
    """Move corresponding mask files to match the SAR files in their respective directories."""
    # Get the current working directory
    cwd = os.getcwd()
    # Construct the path to the directory containing initial dumped masks
    masks_path = os.path.join(cwd, 'dump_masks_here')
    # List all mask files
    masks_files = os.listdir(masks_path)

    # Retrieve lists of SAR files in each category directory
    train_files = os.listdir(os.path.join(cwd, 'original_images', 'train_SAR'))
    val_files = os.listdir(os.path.join(cwd, 'original_images', 'val_SAR'))
    test_files = os.listdir(os.path.join(cwd, 'original_images', 'test_SAR'))

    # Move each mask to match its corresponding SAR file in each category
    for sar in train_files:
        for mask in masks_files:
            if sar.split('.')[0] == mask.split('.')[0]:
                shutil.move(os.path.join(masks_path, mask), 
                            os.path.join(cwd, 'original_images', 'train_mask', mask))
    for sar in val_files:
        for mask in masks_files:
            if sar.split('.')[0] == mask.split('.')[0]:
                shutil.move(os.path.join(masks_path, mask), 
                            os.path.join(cwd, 'original_images', 'val_mask', mask))
    for sar in test_files:
        for mask in masks_files:
            if sar.split('.')[0] == mask.split('.')[0]:
                shutil.move(os.path.join(masks_path, mask), 
                            os.path.join(cwd, 'original_images', 'test_mask', mask))
