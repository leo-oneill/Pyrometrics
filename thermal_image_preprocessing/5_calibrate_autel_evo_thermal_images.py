import cv2
import numpy as np
import os
from PIL import Image
import rasterio
from shutil import copy2
from tqdm import tqdm
import tkinter as tk
from tkinter.filedialog import askdirectory

"""
Script: 5_calibrate_autel_evo_thermal_images.py
Author: Leo O'Neill
Date: 10/15/2023
Description:
    This script processes and calibrates thermal images captured by the Autel EVO drone. 
    It performs the following steps:
    - Prompts the user to select a folder containing Autel thermal TIFF images.
    - Copies the images to a new 'calibrated' output directory.
    - Applies calibration adjustments using band math.
    - Saves the processed images with a modified filename.

Requirements:
    - Python 3.x
    - OpenCV (`pip install opencv-python`)
    - NumPy, PIL, Rasterio (`pip install numpy pillow rasterio`)
    - TQDM for progress tracking (`pip install tqdm`)
    - Tkinter for GUI-based folder selection (included in standard Python installations)

Usage:
    Run the script and select a folder containing the Autel thermal TIFF images when prompted.
    The script will copy and calibrate the images, saving them in a 'calibrated' folder.
"""

# prompt the user for the file path
# Create a tkinter window
root = tk.Tk()
root.withdraw() # Hide the root window
folder = askdirectory(initialdir = "/Volumes/Forest\ Eco/UAS\ imagery/", title="Select folder with Autel TIFF images")
input_folder = folder
output_folder = os.path.join(os.path.dirname(folder), 'calibrated')
# create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

images = [filename for filename in os.listdir(input_folder) if filename.endswith('.TIFF')]

for filename in tqdm(images, desc='Copying images', unit='image'):
    # Copy the image
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}Cal.TIFF')
    copy2(input_path, output_path)

# Now, apply the band math
for filename in tqdm(os.listdir(output_folder), desc='Processing images', unit='image'):
    if filename.endswith('.TIFF') and not filename.startswith('._'):
        file_path = os.path.join(output_folder, filename)

        # Open the image using PIL to preserve EXIF data
        img = Image.open(file_path)
        exif = img.getexif()

        # Convert the image to numpy array (int16) for calculations
        img_cv = np.array(img, dtype=np.int16)

        # Apply the calculation to the raster
        band_calibrated = img_cv * 0.0983 - 268.86

        # Convert back to PIL Image and save with original EXIF data
        img_calibrated = Image.fromarray(band_calibrated.astype(np.int16))
        img_calibrated.save(file_path, exif=exif)
