import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.ndimage import convolve
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image, ExifTags
# from jupyterthemes import jtplot
# jtplot.style(theme='grade3', grid=False, ticks=True, context='paper', figsize=(20, 15), fscale=1.4)


### HELPER FUNCTIONS
## HELP WITH LOADING AND WRITING TO FILE

def load_images(PATH, mask_min):
    files = []
    imgs = []
    imgs_mask = []
    image_times = []  # List to store the image creation times
    for file in sorted(os.listdir(PATH), key=lambda x: get_file_number(x)):
        if file.endswith(".TIFF"):
            files += [str(file)]
            img = cv2.imread(os.path.join(PATH, file), cv2.IMREAD_UNCHANGED)
            img_mask = np.clip(img, None, 4000)
            img_8bit = cv2.normalize(img_mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            _, binary_mask = cv2.threshold(img_8bit, mask_min, 255, cv2.THRESH_BINARY_INV)
            imgs_mask += [cv2.bitwise_and(img_8bit, img_8bit, mask=binary_mask)]
            imgs += [img]
            image_times += [get_image_creation_time(os.path.join(PATH, file))]  # Store the image creation time
        if file.endswith(".jpg"):
            files += [str(file)]
            img = cv2.imread(os.path.join(PATH, file), 0)
            _, binary_mask = cv2.threshold(img, mask_min, 255, cv2.THRESH_BINARY_INV)
            imgs += [img]
            imgs_mask += [cv2.bitwise_and(img, img, mask=binary_mask)]

    # Sort all the lists based on image_times
    sorted_data = sorted(zip(files, imgs, imgs_mask, image_times), key=lambda x: x[3])
    files, imgs, imgs_mask, image_times = zip(*sorted_data)

    return list(files), list(imgs), list(imgs_mask), list(image_times)


def get_file_number(file):
    try:
        return int(file.split("_")[1].split(".")[0])
    except:
        return float('inf')

def get_image_creation_time(image_path):
    image = Image.open(image_path)
    exif_data = image.getexif()  # Retrieve the EXIF data
    if exif_data:
        for tag, value in exif_data.items():
            if tag == 306:  # EXIF tag for creation date and time
                return value
    return None

def create_gif(filenames, PATH):
    kargs = { 'duration': 0.0333}
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(PATH, images, **kargs)

## HELP WITH VISUALIZING
def imshow_with_trajectory(images, ssim, PATH):
    filenames = []
    for k in range(0,len(ssim)):
        f, (a0, a1) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]})

        a0.axis('off')
        a0.imshow(images[k])

        a1.plot(np.arange(len(ssim)), np.array(ssim))
        a1.scatter(k, np.array(ssim)[k], c='r', s=100)
        a1.set_ylim(ymax = 1, ymin = 0)
        a1.set_xlabel('Frame')
        a1.set_ylabel('SSIM')
        
        if not PATH is None:
            filename = PATH + "".join([str(0)]*(3-len(str(k)))) + str(k) +'.png'
            plt.savefig(filename)
            filenames += [filename]
        plt.close()
    return filenames

# user selected transformation
def trans_select():
    options = ["1. Manual (in development)", "2. KAZE", "3. SIFT", "4. ECC (homography)", "5. ECC (aphine)"]
    print("\n choose the type of transformation:", *options, sep = "\n")
    choice = int(input("enter number:"))
    print("\n select the mask bit value (1-255, 148 reccomended) or '255' for no mask")
    choice1 = int(input("enter number:"))
    return choice, choice1
    
# borders
def get_border_pads(img_shape, warp_stack):
    maxmin = []
    corners = np.array([[0,0,1], [img_shape[1], 0, 1], [0, img_shape[0],1], [img_shape[1], img_shape[0], 1]]).T
    warp_prev = np.eye(3)
    print(warp_stack[0].shape)
    for warp in warp_stack:
        if warp.shape[0] == 2:
            warp = np.concatenate([warp, [[0,0,1]]])
        warp = np.matmul(warp, warp_prev)
        warp_invs = np.linalg.inv(warp)
        new_corners = np.matmul(warp_invs, corners)
        xmax,xmin = new_corners[0].max(), new_corners[0].min()
        ymax,ymin = new_corners[1].max(), new_corners[1].min()
        maxmin += [[ymax,xmax], [ymin,xmin]]
        warp_prev = warp.copy()
    maxmin = np.array(maxmin)
    bottom = maxmin[:,0].max()
    print('bottom', maxmin[:,0].argmax()//2)
    top = maxmin[:,0].min()
    print('top', maxmin[:,0].argmin()//2)
    left = maxmin[:,1].min()
    print('right', maxmin[:,1].argmax()//2)
    right = maxmin[:,1].max()
    print('left', maxmin[:,1].argmin()//2)
    return int(-top), int(bottom-img_shape[0]), int(-left), int(right-img_shape[1])

### CORE FUNCTIONS
## FINDING THE TRAJECTORY
#1. Manual", "2. KAZE", "3. SIFT", "4. ECC (homography)", "5. ECC (aphine)

def get_homography(img1, img2, model):
    options = [1, 2, 3, cv2.MOTION_HOMOGRAPHY, cv2.MOTION_AFFINE]
    imga = img1.copy().astype(np.float32)
    imgb = img2.copy().astype(np.float32)
    if len(imga.shape) == 3:
        imga = cv2.cvtColor(imga, cv2.COLOR_BGR2GRAY)
    if len(imgb.shape) == 3:
        imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
    if model == 1:
        print("model not ready")
        warp_matrix = 0
    if model == 2:
        feat = cv2.KAZE_create()
        # Find the key points and descriptors
        keypoints1, descriptors1 = feat.detectAndCompute(img1, None)
        keypoints2, descriptors2 = feat.detectAndCompute(img2, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2,k=2)
        good_matches = [m[0] for m in matches if len(m) > 1 and m[0].distance < m[1].distance * 0.75]
        
        src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
        
        
        warp_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 7.0)
        
    if model == 3:
        feat = cv2.SIFT_create()
        # Find the key points and descriptors
        keypoints1, descriptors1 = feat.detectAndCompute(img1, None)
        keypoints2, descriptors2 = feat.detectAndCompute(img2, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2,k=2)
        good_matches = [m[0] for m in matches if len(m) > 1 and m[0].distance < m[1].distance * 0.75]
        
        src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
        
        
        warp_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 7.0)
        
    if model == 4:
        warpMatrix=np.eye(3, 3, dtype=np.float32)
        warp_matrix = cv2.findTransformECC(templateImage=imga,inputImage=imgb,warpMatrix=warpMatrix, motionType=options[model-1])[1]
        
    if model == 5:
        warpMatrix=np.eye(2, 3, dtype=np.float32)
        warp_matrix = cv2.findTransformECC(templateImage=imga,inputImage=imgb,warpMatrix=warpMatrix, motionType=options[model-1])[1]
        
    return warp_matrix


def calc_similarity(imgs):
    ssim = []
    for i, img in enumerate(imgs[:-1]):
        ssim += [compare_ssim(img, imgs[i+1], channel_axis=False)]
    return np.array(ssim)

def create_warp_stack(imgs, model):
    warp_stack = []
    ssim_stack = []
    if model != 1:
        for i, img in enumerate(imgs[:-1]):
            warp_stack += [get_homography(img, imgs[i+1], model)]
        return np.array(warp_stack)

def homography_gen(warp_stack):
    H_tot = np.eye(3)
    wsp = np.dstack([warp_stack[:,0,:], warp_stack[:,1,:], np.array([[0,0,1]]*warp_stack.shape[0])])
    for i in range(len(warp_stack)):
        H_tot = np.matmul(wsp[i].T, H_tot)
        yield np.linalg.inv(H_tot)#[:2]


def apply_warping_fullview(images, images_8bit, warp_stack, files, image_times):
    top, bottom, left, right = get_border_pads(img_shape=images[0].shape, warp_stack=warp_stack)
    H = homography_gen(warp_stack)
    transformed_images = []
    transformed_images_8bit = []
    for i, (img, img_8bit, image_time) in enumerate(zip(images[1:], images_8bit[1:], image_times)):
        H_tot = next(H) + np.array([[0, 0, left], [0, 0, top], [0, 0, 0]])
        img_warp = cv2.warpPerspective(img, H_tot, (img.shape[1] + left + right, img.shape[0] + top + bottom))
        img_warp_8bit = cv2.warpPerspective(img_8bit, H_tot, (img_8bit.shape[1] + left + right, img_8bit.shape[0] + top + bottom))
        transformed_time = ''.join(filter(str.isdigit, image_time))  # Extract digits from the string
        new_filename = "IRX_" + transformed_time[8:] + ".TIFF"
        #new_filename = "IRX_" + files[i][4:8] + "_ref8" + ".TIFF"
        cv2.imwrite(os.path.join("images/trans", new_filename), img_warp)
        #cv2.imwrite(os.path.join("images/trans_8bit", new_filename), img_warp_8bit)
        transformed_images += [img_warp]
        transformed_images_8bit += [img_warp_8bit]
    return transformed_images, transformed_images_8bit
