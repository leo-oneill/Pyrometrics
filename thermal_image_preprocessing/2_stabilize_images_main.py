import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.ndimage import convolve
from skimage.metrics import structural_similarity as compare_ssim
from 2a_stabilize_helper_functions import *
import pandas as pd
from PIL import Image

"""
Script: 2_stabilize_images_main.py
Author: Leo O'Neill
Date: 04/20/2023
Description:
    This script stabilizes a sequence of images by:
    - Loading images from a specified directory.
    - Applying image processing techniques for alignment.
    - Computing image similarity metrics to assess stabilization.
    - Visualizing the image trajectories and transformations.
    - Saving the stabilized images for further analysis.

    The script relies on helper functions from `2a_stabilize_helper_functions.py`
    and uses OpenCV, NumPy, and SciPy for image transformations.

Requirements:
    - OpenCV (`pip install opencv-python`)
    - NumPy, SciPy, Matplotlib, ImageIO, Pandas, Scikit-Image, and Pillow (`pip install numpy scipy matplotlib imageio pandas scikit-image pillow`)

Usage:
    Ensure images are placed in the specified directory before running the script.
    The script will output transformed images and trajectory visualizations.
"""

#load the images and create a plot of the trajectory
model, mask = trans_select()
files, imgs, imgs_mask, imgs_times = load_images(PATH = "images", mask_min = mask)
name = 'result1'


#display loaded images, files, and warp
print(f" IMAGES: {imgs[0].dtype} min: {np.min(imgs[0])}, max: {np.max(imgs[0])} \n MASK: {imgs_mask[0].dtype}, shape: {imgs_mask[0].shape}, min: {np.min(imgs_mask[0])}, max: {np.max(imgs_mask[0])}")
print("\n number of images loaded, unmasked and masked: ", len(imgs), len(imgs_mask))
print("\n a few names of the files: ", files[0:4])
print("\n a few names of the image times: ", imgs_times[0:4])
concat = cv2.hconcat([cv2.convertScaleAbs(imgs[0], alpha = (255.0/np.amax(imgs[0]))), imgs_mask[0]])
cv2.imshow('Masked/original image', concat)
cv2.waitKey(1000)
cv2.destroyAllWindows()

if model != 1:
    ws = create_warp_stack(imgs_mask, model = model)
    print("\n warp stack shape: ", ws.shape)
    new_imgs, new_imgs_8bit = apply_warping_fullview(images=imgs, warp_stack=ws, files = files, images_8bit = imgs_mask, image_times = imgs_times)

if model == 1:
    print("model not ready")
    exit()

ssim = calc_similarity(new_imgs)
print("\n ssim shape: ", ssim.shape, "fist 5 values: ", ssim[0:4,])

#df = pd.DataFrame(ssim[0:int(ssim.shape[0])], columns=["ssim"])
#df.to_csv(f"ssim_{model}.csv", index=False)



original_trajectory = np.cumsum(ws, axis=0)
print("\n original trajectory shape: ", np.array(original_trajectory).shape)

#plot the original and smoothed trajectory
"""
f, (a0, a2, a1) = plt.subplots(3,1)
f.set_size_inches(8, 6)

i,j = 0,2
a0.scatter(np.arange(len(original_trajectory)), np.array(original_trajectory)[:,i,j], label='Original')
a0.plot(np.arange(len(original_trajectory)), np.array(original_trajectory)[:,i,j])
a0.legend()
a0.set_ylabel('X trajectory')
a0.xaxis.set_ticklabels([])

i,j = 1,2
a2.scatter(np.arange(len(original_trajectory)), np.array(original_trajectory)[:,i,j], label='Original')
a2.plot(np.arange(len(original_trajectory)), np.array(original_trajectory)[:,i,j])
a2.legend()
a2.set_ylabel('Y trajectory')
a2.xaxis.set_ticklabels([])

a1.scatter(np.arange(len(ssim)), np.array(ssim), label='SSIM')
a1.plot(np.arange(len(ssim)), np.array(ssim))
a1.set_ylim(ymax = 1, ymin = 0)
a1.set_xlabel('Frame')
a1.set_ylabel('SSIM')
plt.savefig(name+'SSIM')
"""

#create a images that show both the trajectory and video frames
filenames = imshow_with_trajectory(images=new_imgs_8bit, ssim = ssim, PATH='./out_'+name+'/')


#create gif
create_gif(filenames, './'+name+'.gif')
