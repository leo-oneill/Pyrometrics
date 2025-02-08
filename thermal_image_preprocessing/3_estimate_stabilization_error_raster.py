import cv2
import numpy as np
import os
import pandas as pd

import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import rasterio
from rasterio.plot import show
from statistics import mean

"""
Script: 3_estimate_stabilization_error_raster.py
Author: Leo O'Neill
Date: 11/20/2023
Description:
    This script estimates the stabilization error of raster images by:
    - Loading raster images from a selected directory.
    - Allowing the user to manually select reference points for comparison.
    - Computing and analyzing error metrics between stabilized and reference images.
    - Using raster metadata and transformation matrices to quantify alignment accuracy.

    The script relies on user input for selecting reference points to assess the
    effectiveness of image stabilization.

Requirements:
    - OpenCV (`pip install opencv-python`)
    - NumPy, Pandas, Rasterio (`pip install numpy pandas rasterio`)
    - Tkinter for GUI-based file selection (included in standard Python installations)

Usage:
    Run the script and use the graphical interface to select images and reference points.
    The script will output error metrics to assess the stabilization accuracy.
"""

################# FUNCTIONS #######################
def get_file_number(file):
    try:
        # Split the file name at underscores ("_") and take the second element
        number_string = file.split("_")[1]
        # Extract only the numeric part from the string
        number = int(''.join(filter(str.isdigit, number_string)))
        return number
    except:
        return float('inf')

def load_images(PATH):
    files = []
    imgs = []
    geotransforms = []
#    imgs_mask = []
    for file in sorted(os.listdir(PATH), key=lambda x: get_file_number(x)):
        if file.endswith(".TIFF"):
            files += [str(file)]
            with rasterio.open(os.path.join(PATH, file)) as src:
                img = src.read(2)
#                show(img, cmap='viridis', vmin=0, vmax=100)
                img_mask = np.clip(img, None, 100)
                img_8bit = cv2.normalize(img_mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            geotransforms += [src.transform]
            imgs += [img_8bit]
            #imgs_mask += [cv2.bitwise_and(img_8bit, img_8bit, mask=binary_mask)]
    return files, imgs, geotransforms


def select_csv_file():
    """Prompts the user to select a CSV file and returns its path."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = askopenfilename(initialdir = "/Users/leo/Desktop", title="Select a CSV file", filetypes=[("CSV files", "*.csv")])
    root.destroy()
    return file_path

def extract_coordinates_from_csv(csv_path):
    """Extracts the X and Y coordinates from the first two columns of the CSV file.
    Returns a list of tuples containing the coordinates."""
    df = pd.read_csv(csv_path)
    coordinates = list(zip(df.iloc[:, 0], df.iloc[:, 1]))
    return coordinates

def pixel_to_real(px, py, geotransform):
    """Convert pixel coordinates to real-world coordinates using geotransform."""
    #print(rasterio.transform.xy(geotransform, px, py))
    real_x = geotransform.c + px * geotransform.a + py * geotransform.b
    real_y = geotransform.f + px * geotransform.d + py * geotransform.e
    return real_x, real_y

def click_it(event, x, y, flags, param):
    # If left button is clicked, record the point
    if event == cv2.EVENT_LBUTTONDOWN:
        pixels = (x,y)
        real_x, real_y = pixel_to_real(x, y, param[2])  # Convert to real-world coordinates
        point = (real_x, real_y)
        param[0].append(point)
        cv2.circle(param[1], pixels, 3, (255, 0, 0), -1)
        cv2.putText(param[1], f"({round(real_x,2)}, {round(real_y,2)})", (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.imshow("Image", param[1])

def click_images(image, geotransform):
    points = []
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", image)
    cv2.resizeWindow("Image", 2000, 1200)
    cv2.setMouseCallback("Image", click_it, param=[points, image, geotransform])
    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    return points

# Define a function to calculate the Euclidean distance between two points
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def find_nearest_ref_point(point, ref_points):
    distances = [euclidean_distance(point, ref) for ref in ref_points]
    print(ref_points[np.argmin(distances)])
    return ref_points[np.argmin(distances)]


################## EXECUTION #########################

#select folder w/ images
root = tk.Tk()
root.withdraw() # Hide the root window
print("\n select georeferenced image folder \n")
folder = askdirectory(initialdir = "/Users/leo/Desktop")
print("\n select CSV file \n")
csv_path = select_csv_file()
ref_points = extract_coordinates_from_csv(csv_path)


#read images
files, imgs, geotransforms = load_images(PATH = folder)

print("\n number of images loaded: ", len(imgs))
print("\n a few names of the files: ", files[0:4])

# randomly select X number of images
step = len(imgs) // 29
selected_imgs = [imgs[i] for i in range(0, len(imgs), step)]
selected_transforms = [geotransforms[i] for i in range(0, len(imgs), step)]
# select three+ points in each image, estimate length changes
#print("select reference image points. When complete, press q")
#ref_points = click_images(selected_imgs[0], geotransforms[0])
#print("reference points:", ref_points)

imgs_error = []

for i, img in enumerate(selected_imgs):
    print(f"Processing image {i + 1}, click on image points. When complete, press q")
    selected_points = click_images(img, selected_transforms[i])
    print(selected_points)
    distances = [euclidean_distance(find_nearest_ref_point(selected_points[i], ref_points), selected_points[i]) for i in range(len(selected_points))]
    error = mean(distances)
    print(f"image {i+1} sum error:", round(error,3))
    imgs_error.append(error)

# export each frames error value as csv
df = pd.DataFrame(imgs_error, columns=["euclidian error pts."])
df.to_csv(f"euclidian_error.csv", index=False)

