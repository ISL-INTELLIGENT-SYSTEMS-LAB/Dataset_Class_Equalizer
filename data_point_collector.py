"""
    Author: Taylor J. Brown
    Date: 23APR24
    Orginization: Intelligent Systems Lab (ISL) at the University of Fayetteville
    Project: SAR Image Segmentation for IMPACT 1
"""

import os
from PIL import Image
from tqdm import tqdm

def count_pixels(image_path):
    color_counts = {}

    # Open the image from the specified path
    image = Image.open(image_path)
    pixels = image.load()

    # Initialize the color counts dictionary
    color_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    # Count the occurrence of each color in the image
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            color = pixels[i, j]
            if color not in color_counts:
                color_counts[color] = 1
            else:
                color_counts[color] += 1

    # Change the keys to the class names
    color_counts["urban"] = color_counts.pop(0)
    color_counts["agriculture"] = color_counts.pop(1)
    color_counts["forest"] = color_counts.pop(2)
    color_counts["peatland"] = color_counts.pop(3)
    color_counts["water"] = color_counts.pop(4)
    
    # Sort the color counts by occurrence in descending order and return the result
    return dict(sorted(color_counts.items(), key=lambda x: x[1], reverse=True))


def count_pixels_for_split_images(path, category, images):
    color_counts = {}

    # Create a progress bar using tqdm
    with tqdm(total=len(images), desc=f"Processing {category}", unit='img', bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}', dynamic_ncols=True) as pbar:
        # Count the occurrence of each color in the images
        for image_file in images:
            image = Image.open(os.path.join(path, image_file))
            pixels = image.load()

            for i in range(image.size[0]):
                for j in range(image.size[1]):
                    color = pixels[i, j]
                    if color not in color_counts:
                        color_counts[color] = 1
                    else:
                        color_counts[color] += 1

            # Update the progress bar
            pbar.update()

    # Change the keys to the class names
    color_counts["urban"] = color_counts.pop(0)
    color_counts["agriculture"] = color_counts.pop(1)
    color_counts["forest"] = color_counts.pop(2)
    color_counts["peatland"] = color_counts.pop(3)
    color_counts["water"] = color_counts.pop(4)
    

    # Sort the color counts by occurrence in descending order and return the result
    return dict(sorted(color_counts.items(), key=lambda x: x[1], reverse=True))

def add_count_for_two_images(image1, image2):
    color_counts = {}
    # Count the occurrence of each color in the images
    for image in [image1, image2]:
        image = Image.open(image)
        pixels = image.load()

        for i in range(image.size[0]):
            for j in range(image.size[1]):
                color = pixels[i, j]
                if color not in color_counts:
                    color_counts[color] = 1
                else:
                    color_counts[color] += 1

    # Change the keys to the class names
    color_counts["urban"] = color_counts.pop(0)
    color_counts["agriculture"] = color_counts.pop(1)
    color_counts["forest"] = color_counts.pop(2)
    color_counts["peatland"] = color_counts.pop(3)
    color_counts["water"] = color_counts.pop(4)
    

    # Sort the color counts by occurrence in descending order and return the result
    return dict(sorted(color_counts.items(), key=lambda x: x[1], reverse=True))


def percentage_of_class_pre(color_counts, total_pixels):
    # return the percentage to the 4th decimal place
    return [(color, round(count / total_pixels, 4)) for color, count in color_counts]


def percentage_of_class_post(color_counts, total_pixels):
    # return the percentage to the 4th decimal place
    return [(color, round((count / total_pixels)/4, 4)) for color, count in color_counts]


def messure_increase_in_data_points(pre, post):
    # Calculate the increase in data points for each class
    increase = []
    for i in range(len(pre)):
        increase.append((pre[i][0], post[i][1] - pre[i][1]))

    return increase


def main():
    # Get the current working directory
    cwd = os.getcwd()

    # Pre-split image
    image_path = os.path.join(cwd, "original_images", "val_Mask", "T112_converted_RGB_1302.png")

    # Pre-split image class count and percentage
    print("Pre-split Image class count: " + str(count_pixels(image_path)))
    print("Pre-split Image class percentage: " + str(percentage_of_class_pre(count_pixels(image_path), 512 * 512)))

    # Post-split images
    path = os.path.join(cwd, "split_images", "val_Mask")
    image_files = ["T112_converted_RGB_1302_TL.png", "T112_converted_RGB_1302_TR.png",
                   "T112_converted_RGB_1302_BL.png", "T112_converted_RGB_1302_BR.png"]

    # Post-split image class count and percentage
    print("\nPost-split Image class count: " + str(count_pixels_for_split_images(path, image_files)))
    print("Post-split Image class percentage: " + str(percentage_of_class_post(count_pixels_for_split_images(path, image_files), 512 * 512)))

    for image in image_files:
        print("\n" + image)
        print("Image class count: " + str(count_pixels(os.path.join(path, image))))
        print("Image class percentage: " + str(percentage_of_class_pre(count_pixels(os.path.join(path, image)), 512 * 512)))

    # Messure the increase in data points
    print("\nIncrease in data points: " + str(messure_increase_in_data_points(count_pixels(image_path),
                                              count_pixels_for_split_images(path, image_files))))

    # Add BL with the original image to get the total number of data points
    images = [os.path.join(cwd, "split_images", "val_Mask", "T112_converted_RGB_1302_BL.png"),
              os.path.join(cwd, "original_images", "val_Mask", "T112_converted_RGB_1302.png")]
    print("\nTotal number of data points: " + str(add_count_for_two_images(images[0], images[1])))


if __name__ == "__main__":
    main()
