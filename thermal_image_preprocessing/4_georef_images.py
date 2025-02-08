import os
from osgeo import gdal
import tkinter as tk
from tkinter import filedialog

"""
Script: 4_georef_images.py
Author: Leo O'Neill
Date: 05/24/2023
Description:
    This script georeferences images using a reference image and Ground Control Points (GCPs).
    It performs the following steps:
    - Loads a georeferenced reference image.
    - Extracts the transformation matrix from the reference image.
    - Defines Ground Control Points (GCPs) for georeferencing.
    - Applies a georeferencing transformation using GDAL.
    - Saves the output as a georeferenced TIFF file.

Requirements:
    - GDAL (`pip install gdal`)
    - Tkinter for GUI-based file selection (included in standard Python installations)

Usage:
    Run the script and select a reference image when prompted.
    The script will apply georeferencing and output a corrected image.
"""

# Create a tkinter window
root = tk.Tk()
root.withdraw() # Hide the root window

# Show a file dialog to select the reference image
ref_img_path = filedialog.askopenfilename(title="Select reference image", filetypes=[("TIFF files", "*.tif")])

ref_ds = gdal.Open(ref_img_path)

# Get the transformation matrix from the georeferenced image
gt = ref_ds.GetGeoTransform()

print(gt)

# Set up the GCPs
gcp_list = [gdal.GCP(276.045, 385.009, 763553, 3395090),
            gdal.GCP(312.585, 269.657, 763560, 3395110),
            gdal.GCP(441.245, 235.363, 763579, 3395120),
            gdal.GCP(276.33, 164.479, 763554, 3395130)]
# Set up the warp options
warp_options = gdal.WarpOptions(format='GTiff',
                                outputType=gdal.GDT_Float32,
                                dstSRS='EPSG:26916',
                                #polynomialOrder = [1],
                                warpOptions=['-r', 'near'],
                                transformerOptions=['-order', '1'],
                                creationOptions=['COMPRESS=None'])

# Loop through all images in the folder
for filename in os.listdir('/Users/leo/Desktop/registration_results/hamok_results/plot1/stabilized'):
    if filename.endswith('.TIFF'):
        # Open the image
        ds = gdal.Open(os.path.join('/Users/leo/Desktop/registration_results/hamok_results/plot1/stabilized', filename), gdal.GA_Update)
        output = os.path.join('/Users/leo/Desktop/registration_results/hamok_results/plot1/georeferenced', f'{os.path.splitext(filename)[0]}_geo.TIFF')
        
        #ds.SetGCPs(gcp_list, ds.GetProjection())
        
        ds = gdal.Translate(output, ds, GCPs=gcp_list)
        
        # Set the transformation matrix to the same as the reference image (affine)
        # ds.SetGeoTransform(gt)
        gdal.Warp(output, ds, options=warp_options)

        # Set the projection to the same as the reference image
        #ds.SetProjection(ref_ds.GetProjection())

        # Close the image
        ds = None

# Close the georeferenced image
ref_ds = None












exit()

# Set up the GCPs
# Set up the GCPs
gcp_list = [gdal.GCP(276.045, 385.009, 763553, 3395090),
            gdal.GCP(312.585, 269.657, 763560, 3395110),
            gdal.GCP(441.245, 235.363, 763579, 3395120),
            gdal.GCP(276.33, 164.479, 763554, 3395130)]

# Set up the warp options
warp_options = gdal.WarpOptions(format='GTiff',
                                #outputType=gdal.GDT_Float32,
                                dstSRS='EPSG:26916',
                                warpOptions=['-r', 'near'],
                                polynomialOrder = [1],
                                #transformerOptions=['-order', '1'],
                                creationOptions=['COMPRESS=None'])

# Loop through all the tiff files in the image_reg folder
for filename in os.listdir('/Users/leo/Desktop/registration_results/hamok_results/plot1/stabilized'):
    if filename.endswith('.TIFF'):
        # Create input and output paths
        input_path = os.path.join('/Users/leo/Desktop/registration_results/hamok_results/plot1/stabilized', filename)
        output_path = os.path.join('/Users/leo/Desktop/registration_results/hamok_results/plot1/georeferenced', f'{os.path.splitext(filename)[0]}_geo.TIFF')
        # Apply gdal_translate to embed GCPs
        gdal.Translate(output_path, input_path, GCPs=gcp_list)
        
        # Apply gdalwarp to warp the image
        gdal.Warp(output_path, input_path, options=warp_options)

