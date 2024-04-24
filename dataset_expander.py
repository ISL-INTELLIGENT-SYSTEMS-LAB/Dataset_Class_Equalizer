"""
    Author: Taylor J. Brown
    Date: 23APR24
    Orginization: Intelligent Systems Lab (ISL) at the University of Fayetteville
    Project: SAR Image Segmentation for IMPACT 1
"""

import os
from PIL import Image
import matplotlib.pyplot as plt


def save_image(image, path, filename, ext="png"):
    """
    Save the image with a specific suffix and original format. 
    Save the filename to a corresponding text file for later use.
    """
    try:
        # Save the image at the specified path with the given filename and extension
        image.save(os.path.join(path, f"{filename}.{ext}"))

        # Check if path does not contain "Mask", indicating it's not a mask image
        if "Mask" not in path:
            # Based on the directory name, append the filename to a corresponding text file
            if "test" in path:
                with open("split_images_test.txt", "a") as f:
                    f.write(f"{filename}.{ext}\n")
            elif "train" in path:
                with open("split_images_train.txt", "a") as f:
                    f.write(f"{filename}.{ext}\n")
            elif "val" in path:
                with open("split_images_val.txt", "a") as f:
                    f.write(f"{filename}.{ext}\n")
                    
    except Exception as e:
        print(f"Failed to save image: {str(e)}")
        return


def split_image_into_four(image_path):
    """
    Split the image into 4 images of equal size and resize back to original dimensions.
    """
    try:
        # Open the image from the specified path
        with Image.open(image_path) as img:
            width, height = img.size
            # Split the image into 4 quadrants
            quadrants = {
                "TL": img.crop((0, 0, width//2, height//2)),
                "TR": img.crop((width//2, 0, width, height//2)),
                "BL": img.crop((0, height//2, width//2, height)),
                "BR": img.crop((width//2, height//2, width, height))
            }
            # Resize each quadrant back to the original dimensions
            return {key: quadrant.resize((width, height)) for key, quadrant in quadrants.items()}
                
    except Exception as e:
        print(f"Failed to split image: {str(e)}")
        return


def show_segmented_image(save_location, image_files):
    """
    Display images found in image_files as a 2x2 grid with no axis labels and a narrow space
    between images. Assumes there are exactly four images to fit into the grid.
    """
    try:
        # Load images from specified files
        images = [Image.open(os.path.join(save_location, img)) for img in image_files]
        
        # Create a 2x2 grid for displaying images
        _, axes = plt.subplots(2, 2, figsize=(10, 10))
        for ax, img in zip(axes.ravel(), images):
            ax.imshow(img)
            ax.axis('off')  # Hide the axis

        # Adjust spacing between images in the grid
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.show()
    except Exception as e:
        print(f"Failed to display segmented images: {str(e)}")
        return
