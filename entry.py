"""
    Author: Taylor J. Brown
    Date: 23APR24
    Orginization: Intelligent Systems Lab (ISL) at the University of Fayetteville
    Project: SAR Image Segmentation for IMPACT 1
"""

import os
from tqdm import tqdm
from dataset_spliter import *
from dataset_expander import *
from data_point_collector import *
from filter_dataset import *


def split_images(cwd=None):
    # Set the current working directory if not provided
    if cwd is None:
        cwd = os.getcwd()

    # Define the paths for the original and split images
    original_folder_path = os.path.join(cwd, "original_images")
    split_folder_path = os.path.join(cwd, "split_images")

    try:
        # List subdirectories in the original and split image folders
        original_sub_folders = os.listdir(original_folder_path)
        split_sub_folders = os.listdir(split_folder_path)
    except FileNotFoundError as e:
        # Handle the case where a directory does not exist
        print(f"Error finding directory: {str(e)}")
        return
    except PermissionError as e:
        # Handle the case where permission is denied
        print(f"Permission denied: {str(e)}")
        return

    # Check if the number of subfolders in original and split match
    if len(original_sub_folders) != len(split_sub_folders):
        print("Mismatch in the number of subfolders between original and split images.")
        return
    
    # Create a mapping of original folders to split folders
    folder_map = dict(zip(original_sub_folders, split_sub_folders))

    # Process each folder pair
    for original_folder, split_folder in folder_map.items():
        original_images_path = os.path.join(original_folder_path, original_folder)
        split_images_path = os.path.join(split_folder_path, split_folder)

        try:
            # List all images in the original folder
            images = os.listdir(original_images_path)

            # Initialize progress bar for processing images
            with tqdm(total=len(images), 
                      desc=f"Processing {original_folder:<10} images", 
                      unit='img', 
                      bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}', 
                      dynamic_ncols=True) as pbar:
                for image in images:
                    # Process each image, split into four, and save each quadrant
                    image_path = os.path.join(original_images_path, image)
                    images_split = split_image_into_four(image_path)
                    for key, quadrant in images_split.items():
                        save_image(quadrant, split_images_path, os.path.splitext(image)[0] + "_" + key, os.path.splitext(image)[1][1:])
                    
                    # Update the progress bar after each image is processed
                    pbar.update(1)
        except Exception as e:
            # Handle any other exceptions during image processing
            print(f"Failed to process folder {original_folder}: {str(e)}", flush=True)
            continue

    # Notify that processing of all images is complete
    print("Processing complete.\n")
    return


def get_occurrence_percentage(dictionary):
    # Calculate the percentage for each class
    return {class_name: (count / sum(dictionary.values()) * 100) for class_name, count in dictionary.items()}


def display_dataset_details(path, dataset_type, pixel_count_function, percentage_function):
    # Get the list of files in the directory
    files = os.listdir(path)
    
    # Calculate the total number of classes and get the class occurrences
    total_pixels = 512 * 512 * len(files)
    class_counts = pixel_count_function(path, dataset_type, files)

    # Display the results
    print(f"\nTotal number of classes in {dataset_type}: {total_pixels:,}")
    print(f"\n{dataset_type.capitalize()} class occurrences: ")
    for key, value in class_counts.items():
        print(f"{key}: {value:,}")
    
    # Calculate and display percentages
    class_percentages = percentage_function(class_counts)
    print(f"\n{dataset_type.capitalize()} class percentages: ")
    for key, value in class_percentages.items():
        print(f"{key}: {value:.3f}%")
    print("")


def process_dataset(dataset_path, dataset_type="dataset"):
    display_dataset_details(dataset_path, dataset_type, count_pixels_for_split_images, get_occurrence_percentage)


def main():
    # Continuously prompt until the user decides to stop
    while True:
        # Prompt user for initial choice about cleaning directories
        response = input("Do you want to split and clear the dump directories? (y/n): ")
        if response.lower() == "y":
            # Remove the readme files from the dump directories
            os.remove(os.path.join(os.getcwd(), "dump_sar_here", "readme.txt"))
            os.remove(os.path.join(os.getcwd(), "dump_masks_here", "readme.txt"))
            # Check if the dump directories have matching numbers of SAR and mask files
            sar = os.listdir(os.path.join(os.getcwd(), "dump_sar_here"))
            masks = os.listdir(os.path.join(os.getcwd(), "dump_masks_here"))
            #if len(sar) == len(masks) > 0:
            # If files are present and counts match, proceed to split files into sets
            initiate_split()  # Default ratios for splitting are 0.6, 0.2, 0.2
            move_corresponding_masks()  # Ensure masks match the split SAR files
            print("Files have been split ad moved to the appropriate directories.\n")
            break
            else:
                # Notify user of mismatch or empty directories
                print("No files to split. Please add files to the dump directories.")
                # Inner loop to manage file addition without exiting the program
                while True:
                    response1 = input("Do you want to add files to the dump directories? (y/n): ")
                    if response1.lower() == "y":
                        continue  # Continue prompting within the nested loop
                    elif response1.lower() == "n":
                        response = "n"  # Set outer loop to break condition
                        break
                    else:
                        print("Invalid response. Please enter 'y' or 'n'.")
        elif response.lower() == "n":
            break  # Exit the loop if user declines to clear directories
        else:
            print("Invalid response. Please enter 'y' or 'n'.")

    # Handle dataset expansion based on user input
    while True:
        response = input("Do you want to expand the dataset? (y/n): ")
        if response.lower() == "y":
            split_images()  # Splits each image into 4 quadrants to increase dataset size
            break
        elif response.lower() == "n":
            break
        else:
            print("Invalid response. Please enter 'y' or 'n'.")

    # Process class distribution in the dataset after splitting
    while True:
        response = input("Do you want to process the class distribution post split? (y/n): ")
        if response.lower() == "y":
            base_path = os.path.join(os.getcwd(), "split_images")
            # Set paths for each dataset section
            val_mask_path = os.path.join(base_path, "val_Mask")
            test_mask_path = os.path.join(base_path, "test_Mask")
            train_mask_path = os.path.join(base_path, "train_Mask")
            # Process and analyze each section
            process_dataset(val_mask_path, "val")
            process_dataset(test_mask_path, "test")
            process_dataset(train_mask_path, "train")
            break
        elif response.lower() == "n":
            break
        else:
            print("Invalid response. Please enter 'y' or 'n'.")

    # Filter dataset based on user-defined thresholds
    while True:
        response = input("Do you want to filter the dataset based on a threshold? (y/n): ")
        if response.lower() == "y":
            response1 = input("Do you want to use the default threshold values of 10%, 12%, 14%, 16%? (y/n): ")
            if response1.lower() == "y":
                initiate_filter()  # Apply predefined threshold levels
                break
            elif response1.lower() == "n":
                # Allow user to specify custom thresholds
                while True:
                    try:
                        thresholds = [float(x) for x in input("Enter the threshold values separated by a space (.12 = 12%): ").split()]
                        # Sort from lowest to highest threshold
                        thresholds.sort()
                        initiate_filter(thresholds)
                        break
                    except ValueError:
                        print("Invalid input. Please enter numerical values.")
            break
        elif response.lower() == "n":
            break
        else:
            print("Invalid response. Please enter 'y' or 'n'.")

    # Post-filter class distribution processing
    while True:
        response = input("Do you want to process the class distribution post filter? (y/n): ")
        if response.lower() == "y":
            # Process datasets at each filtration level
            for i in range(1,5):
                filtered_images_path = os.path.join(os.getcwd(), f"filtered_images{i}")
                val_mask_path = os.path.join(filtered_images_path, "val_mask")
                test_mask_path = os.path.join(filtered_images_path, "test_mask")
                train_mask_path = os.path.join(filtered_images_path, "train_mask")
                print(f"\nFiltered Dataset {i}")
                process_dataset(val_mask_path, "val")
                process_dataset(test_mask_path, "test")
                process_dataset(train_mask_path, "train")
            break
        elif response.lower() == "n":
            break
        else:
            print("Invalid response. Please enter 'y' or 'n'.")

    
if __name__ == "__main__":
    main()
