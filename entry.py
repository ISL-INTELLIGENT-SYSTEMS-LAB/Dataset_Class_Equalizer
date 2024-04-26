"""
    Author: Taylor J. Brown
    Date: 23APR24
    Orginization: Intelligent Systems Lab (ISL) at the University of Fayetteville
    Project: SAR Image Segmentation for IMPACT 1
"""

import os
import re
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


def display_dataset_details(path, dataset_type, pixel_count_function, percentage_function, auto):
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

    if auto:
        # Check if path is filtered_images or split_images
        if "split" in path:
            with open(f"{dataset_type}_class_distribution.txt", "w") as f:
                f.write(f"Total number of classes in {dataset_type}: {total_pixels:,}\n\n")
                f.write(f"{dataset_type.capitalize()} class occurrences: \n")
                for key, value in class_counts.items():
                    f.write(f"{key}: {value:,}\n")
                f.write("\n")
                f.write(f"{dataset_type.capitalize()} class percentages: \n")
                for key, value in class_percentages.items():
                    f.write(f"{key}: {value:.3f}%\n")
                f.write("\n")
        elif "filtered" in path:
            # Split the number off the end of the filtered_images# folder
            iteration = re.search(r"filtered_images(\d+)", path).group(1)
            with open(f"filtered_{iteration}_class_distribution.txt", "w") as f:
                f.write(f"Total number of classes in {dataset_type}: {total_pixels:,}\n\n")
                f.write(f"{dataset_type.capitalize()} class occurrences: \n")
                for key, value in class_counts.items():
                    f.write(f"{key}: {value:,}\n")
                f.write("\n")
                f.write(f"{dataset_type.capitalize()} class percentages: \n")
                for key, value in class_percentages.items():
                    f.write(f"{key}: {value:.3f}%\n")
                f.write("\n")


def process_dataset(dataset_path, dataset_type="dataset", auto=False):
    display_dataset_details(dataset_path, dataset_type, count_pixels_for_split_images, get_occurrence_percentage, auto)


def check_image_count(dir_path, catgory, auto=False):
    if len(os.listdir(dir_path)) == 0:
        print(f"No images to process in {catgory}.")
        return
    else:
        process_dataset(dir_path, catgory, auto=auto)


def validate_all_images_have_pairs(sar_path, mask_path):
    # Check if all SAR images have corresponding masks
    sar_images = os.listdir(sar_path)
    mask_images = os.listdir(mask_path)
    for sar_image in sar_images:
        if sar_image not in mask_images:
            return False
    return True


def create_txt_file(iteration):
    img_folders = os.listdir(os.path.join(os.getcwd(), f"filtered_images{iteration}"))
    with open(f"filter_{iteration}.txt", "w") as f:
        for folder in img_folders:
            if folder.endswith("mask"):
                continue
            f.write(f"{folder}:\n")
            for img in os.listdir(os.path.join(os.getcwd(), f"filtered_images{iteration}", folder)):
                f.write(f"{img}\n")
            f.write("\n")
        

def automated_main():
    """
    1. Clear the dump directories
    2. Split the files into training, validation, and testing sets (Default: 0.6, 0.2, 0.2)
    3. Move corresponding mask files to match the SAR files in their respective directories
    4. Split the images into quadrants
    5. Check the class distribution post split
    6. Filter the dataset based on a threshold (Default: 10%, 12%, 14%, 16%)
    7. Process the class distribution post filtration
    8. Write the iteration file names that passed the threshold to a text file
    """
    try:
        sar = os.path.join(os.getcwd(), "dump_sar_here")
        masks = os.path.join(os.getcwd(), "dump_masks_here")
        # Check if all SAR images have corresponding masks
        if not validate_all_images_have_pairs(sar, masks):
            print("Error: Not all SAR images have corresponding masks. Please ensure all images have pairs.")
            return

        # Remove the readme files from the dump directories
        os.remove(os.path.join(os.getcwd(), "dump_sar_here", "readme.txt"))
        os.remove(os.path.join(os.getcwd(), "dump_masks_here", "readme.txt"))

        # Split the files at the default ratios of 0.6, 0.2, 0.2
        initiate_split()
        move_corresponding_masks()
        split_images()
        base_path = os.path.join(os.getcwd(), "split_images")
        val_mask_path = os.path.join(base_path, "val_mask")
        test_mask_path = os.path.join(base_path, "test_mask")
        train_mask_path = os.path.join(base_path, "train_mask")
        check_image_count(val_mask_path, "val", True)
        check_image_count(test_mask_path, "test", True)
        check_image_count(train_mask_path, "train", True)
        initiate_filter(auto=True)
        folders = os.listdir(os.getcwd())
        count = 0
        for folder in folders:
            if folder.startswith("filtered"):
                count += 1
        for i in range(1, count+1):
            filtered_images_path = os.path.join(os.getcwd(), f"filtered_images{i}")
            val_mask_path = os.path.join(filtered_images_path, "val_mask")
            test_mask_path = os.path.join(filtered_images_path, "test_mask")
            train_mask_path = os.path.join(filtered_images_path, "train_mask")
            print(f"\nFiltered Dataset {i}")
            check_image_count(val_mask_path, "val", True)
            check_image_count(test_mask_path, "test", True)
            check_image_count(train_mask_path, "train", True)
            create_txt_file(i)
    except Exception as e:
        print(f"Error: {str(e)}")
        return
    

def main():
    while True:
        print("""
Welcome to the SAR Image Dataset Expander!

This program will help you augment your dataset by:
- Splitting images into quadrants.
- Filtering based on a threshold.

➤ Ensure your SAR images are in 'dump_sar_here' and masks in 'dump_masks_here'.
➤ The program will process the images and augment the dataset automatically.

Required Directories:
1. dump_sar_here
2. dump_masks_here

➤ Automated mode uses all defaults for dataset augmentation.

Steps to follow:
1. Clear the dump directories.
2. Split files into training, validation, and testing sets (Defaults: 60%, 20%, 20%).
3. Move corresponding mask files.
4. Split images into quadrants.
5. Check class distribution after split.
6. Filter dataset by thresholds (Defaults: 10%, 12%, 14%, 16%).
7. Process class distribution after filtration.
8. Log the iteration file names that passed the threshold.

Thank you for using the SAR Image Dataset Expander!
""")

        response = input("Do you want to run the program and have it automatically use all defaults to augment the dataset? (y/n): ")
        if response.lower() == "y":
            automated_main()
            return
        elif response.lower() == "n":
            break
        else:
            print("Invalid response. Please enter 'y' or 'n'.")
    # Continuously prompt until the user decides to stop
    while True:
        # Prompt user for initial choice about cleaning directories
        response = input("Do you want to split and clear the dump directories? (y/n): ")
        if response.lower() == "y":
            # Check if all SAR images have corresponding masks
            if not validate_all_images_have_pairs(sar, masks):
                print("Error: Not all SAR images have corresponding masks. Please ensure all images have pairs.")
                break

            # Remove the readme files from the dump directories
            os.remove(os.path.join(os.getcwd(), "dump_sar_here", "readme.txt"))
            os.remove(os.path.join(os.getcwd(), "dump_masks_here", "readme.txt"))

            # Check if the dump directories have matching numbers of SAR and mask files
            sar = os.listdir(os.path.join(os.getcwd(), "dump_sar_here"))
            masks = os.listdir(os.path.join(os.getcwd(), "dump_masks_here"))
            if len(sar) == len(masks) > 0:
                # If files are present and counts match, proceed to split files into sets
                # Ask user if they want to split the files at the default ratios of 0.6, 0.2, 0.2 or a custom ratio
                response1 = input("Do you want to use the default split ratios of 0.6, 0.2, 0.2? (y/n): ")
                if response1.lower() == "y":
                    initiate_split()  # Default ratios for splitting are 0.6, 0.2, 0.2
                    move_corresponding_masks()  # Ensure masks match the split SAR files
                    print("Files have been split and moved to the appropriate directories.\n")
                    break
                elif response1.lower() == "n":
                    # Allow user to specify custom split ratios
                    while True:
                        try:
                            ratios = [float(x) for x in input("Enter the split ratios separated by a space (.6 = 60% | Order: Train, Val, Test): ").split()]
                            # Ensure the sum of the ratios is equal to 1
                            if sum(ratios) != 1:
                                print("Sum of ratios must equal 1. Please try again.")
                                continue
                            initiate_split(ratios)  # Split files based on custom ratios
                            move_corresponding_masks()  # Ensure masks match the split SAR files
                            print("Files have been split and moved to the appropriate directories.\n")
                            break
                        except ValueError:
                            print("Invalid input. Please enter numerical values.")
                    break
            else:
                # Remake .txt file with the text
                with open(os.path.join(os.getcwd(), "dump_sar_here", "readme.txt"), "w") as f:
                    f.write("Place SAR images here for processing.")

                with open(os.path.join(os.getcwd(), "dump_masks_here", "readme.txt"), "w") as f:
                    f.write("Place masks here for processing.")                

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
        response = input("Do you want to split the dataset images into quadrants? (y/n): ")
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
            val_mask_path = os.path.join(base_path, "val_mask")
            test_mask_path = os.path.join(base_path, "test_mask")
            train_mask_path = os.path.join(base_path, "train_mask")
            # Process and analyze each section
            check_image_count(val_mask_path, "val")
            check_image_count(test_mask_path, "test")
            check_image_count(train_mask_path, "train")
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
        response = input("Do you want to process the class distribution post filtration? (y/n): ")
        if response.lower() == "y":
            # Process datasets at each filtration level
            folders = os.listdir(os.getcwd())
            # Count the number of filtered datasets
            count = 0
            for folder in folders:
                if folder.startswith("filtered"):
                    count += 1
            
            if count != 0:
                for i in range(1, count+1):
                    filtered_images_path = os.path.join(os.getcwd(), f"filtered_images{i}")
                    val_mask_path = os.path.join(filtered_images_path, "val_mask")
                    test_mask_path = os.path.join(filtered_images_path, "test_mask")
                    train_mask_path = os.path.join(filtered_images_path, "train_mask")
                    print(f"\nFiltered Dataset {i}")

                    check_image_count(val_mask_path, "val")
                    check_image_count(test_mask_path, "test")
                    check_image_count(train_mask_path, "train")
            else:
                print("Please filter images first!")
            break
        elif response.lower() == "n":
            break
        else:
            print("Invalid response. Please enter 'y' or 'n'.")

    
if __name__ == "__main__":
    main()
