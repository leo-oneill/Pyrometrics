import os
import shutil
from exif import Image
from datetime import datetime


INPUT_FOLDER =              "./Input Images/"
OUTPUT_FOLDER =             "./Output Images/"
THERMAL_FILENAME_CONTAINS = "T"
RGB_FILENAME_CONTAINS =     "W" # for wide camera
OUTPUT_FILENAME_DIGITS =    5   # Number of digits in output filename

"""
Script: 1_preprocess_images.py
Author: Leo O'Neill
Date: 07/06/2023
Description:
    This script processes images by:
    - Sorting them into RGB and thermal categories based on filename patterns.
    - Extracting EXIF metadata (including timestamps) from RGB images.
    - Renaming and organizing the images into a structured format.
    - Copying the processed images into an output folder.

    The script is intended to prepare images for further analysis, ensuring they
    are systematically named and categorized.

Requirements:
    - Python 3.x
    - exif module (install via `pip install exif`)
    
Usage:
    Ensure images are placed in the "Input Images" folder before running the script.
    Processed images will be saved in the "Output Images" folder.
"""


#image_names = os.listdir(INPUT_FOLDER)
image_names = [img for img in os.listdir(INPUT_FOLDER) if not img.startswith('._')]  # Exclude macOS '._' files

# Sort images into rgb and thermal
RGB_filenames = []
IR_filenames = []
for id, image in enumerate(image_names):
    if THERMAL_FILENAME_CONTAINS in image:
        IR_filenames.append(image)
    elif RGB_FILENAME_CONTAINS in image:
        RGB_filenames.append(image)
print("images sorted RGB/IR")
# Grab datetimes (strings) from RGB image exifs
RGB_datetimes = []
for file in RGB_filenames:
    with open(INPUT_FOLDER+file, 'rb') as src:
        img = Image(src)
    if img.has_exif:
        try:
            RGB_datetimes.append(img.datetime)
        except AttributeError:
            print('Image lacks datetime in exif')
            exit(1)
    else:
        print('Image lacks exif data')
        exit(1)

# Grab datetimes (strings) from IR image exifs
IR_datetimes = []
for file in IR_filenames:
    with open(INPUT_FOLDER+file, 'rb') as src:
        img = Image(src)
    if img.has_exif:
        try:
            IR_datetimes.append(img.datetime)
        except AttributeError:
            print('Image lacks datetime in exif')
            exit(1)
    else:
        print('Image lacks exif data')
        exit(1)

print("datetimes processed")

# convert string datetimes to datetime datetime
RGB_datetimes = [datetime.strptime(x, "%Y:%m:%d %H:%M:%S") for x in RGB_datetimes]
IR_datetimes = [datetime.strptime(x, "%Y:%m:%d %H:%M:%S") for x in IR_datetimes]

matched_filename_pairs = []
# now go through and find closest datetime matches for each RGB image
# NOTE: this may result in duplicate IR images if there are extra RGB images in folder.
for id, rgb_datetime in enumerate(RGB_datetimes):
    matched_filename_pairs.append((RGB_filenames[id], IR_filenames[min(range(len(IR_datetimes)), key = lambda i: abs(IR_datetimes[i]-rgb_datetime))]))

print("images paired")

# ensure output directories exist
if not os.path.exists(OUTPUT_FOLDER + 'RGB/'):
    os.makedirs(OUTPUT_FOLDER + 'RGB/')
if not os.path.exists(OUTPUT_FOLDER + 'Thermal/'):
    os.makedirs(OUTPUT_FOLDER + 'Thermal/')

# Copy images to output folders, renaming to X digit numbers as filenames.
for id, (RGB_file, IR_file) in enumerate(matched_filename_pairs):
    shutil.copy(INPUT_FOLDER + RGB_file, f'{OUTPUT_FOLDER}RGB/{"0"*(OUTPUT_FILENAME_DIGITS-len(str(id+1))) + str(id+1)}.{RGB_file.split(".")[1]}')
    shutil.copy(INPUT_FOLDER + IR_file, f'{OUTPUT_FOLDER}Thermal/{"0"*(OUTPUT_FILENAME_DIGITS-len(str(id+1))) + str(id+1)}.{IR_file.split(".")[1]}')

print("files coped to ouput.")
print("finished")
