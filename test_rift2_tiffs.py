"""
This script runs the RIFT2 (Radiation-variation Insensitive Feature Transform) algorithm
to find matching points (keypoints) between pairs of multimodal images (e.g., optical and SAR)
across multiple resolution levels, from highly downsampled ('sub32') to full resolution ('full').

It handles essential preprocessing, matching, and outlier removal, while carefully managing
system memory to avoid crashes on large images.
"""

import os        # For interacting with the operating system, like joining file paths
import cv2       # OpenCV for image processing tasks (loading images, drawing matches)
import numpy as np # NumPy for fast numerical array operations (matrices, math)
import time      # For timing how long different steps take
import csv       # For writing the final matching points to a CSV file
import sys       # For system-level operations like flushing standard output
import gc        # Garbage Collector interface to manually free up memory when needed
import joblib    # For running the RIFT2 pipeline in parallel across multiple CPU cores
from tqdm import tqdm # For displaying a progress bar in the terminal

# Import the core RIFT2 algorithm and helper matching functions from local source files
from src.RIFT2 import RIFT2
from src.matcher_functions import match_keypoints_nn, draw_matches, outlier_removal

def scale_to_uint8(img):
    """
    Min-Max scaling to convert raw image data (which might be floats or 16-bit) to 8-bit (uint8).
    8-bit format is required by many computer vision algorithms, including OpenCV tools.
    This function also masks out invalid 'NoData' pixels and stretches the contrast 
    using the 2nd and 98th percentiles to ignore extreme dark/bright outliers.
    """
    if img is None:
        return None
    
    # If the image is already 8-bit, no scaling is needed.
    if img.dtype == np.uint8:
        return img
        
    print(f"Applying robust radiometric correction from {img.dtype} to uint8")
    
    # Create a mask to identify valid pixels. 
    # Assumes -9999 is used as a 'NoData' proxy, common in geospatial data.
    # np.isfinite ensures we don't include NaN (Not a Number) or Infinity values.
    valid_mask = np.logical_and(np.isfinite(img), img > -9999)
    
    # If there are no valid pixels at all, return an empty (black) image
    if not np.any(valid_mask):
        return np.zeros_like(img, dtype=np.uint8)
        
    # Get robust percentiles (2nd and 98th), ignoring NoData and extreme outliers.
    # This prevents a single super-bright/super-dark pixel from ruining the contrast stretch.
    valid_pixels = img[valid_mask]
    p_low, p_high = np.percentile(valid_pixels, (2, 98))
    
    # If the image is completely uniform (all valid pixels have the same value)
    if p_high == p_low:
        scaled = np.zeros_like(img, dtype=np.float32)
    else:
        # Clip image values to fall strictly within our robust min/max range
        clipped = np.clip(img, p_low, p_high)
        # Scale the remaining values to range exactly between 0 and 255
        scaled = (clipped - p_low) / (p_high - p_low) * 255.0
        
    # Ensure all values are purely 8-bit integers
    scaled = np.clip(scaled, 0, 255).astype(np.uint8)
    
    # Important: Re-apply the NoData mask. Set these regions strictly to 0 (Black)
    scaled[~valid_mask] = 0
    
    return scaled

# The progression of resolutions to test. 'sub32' means 1/32th of the original resolution.
# We start with the smallest (coarsest) and work up to 'full' resolution.
resolutions = ['sub32', 'sub16', 'sub8', 'sub4', 'sub2', 'full']

# Dynamic optimization parameters tailored for each resolution scale.
# As the image gets larger, we adjust parameters:

# npt_map defines the maximum number of keypoints to extract.
# Larger images have more details, so we want more keypoints.
npt_map = {
    'sub32': 5000,
    'sub16': 10000,
    'sub8': 20000,
    'sub4': 40000,
    'sub2': 80000,
    'full': 150000
}

# patch_size_map determines the localized area size considered around a feature.
# High resolutions require a larger patch to capture enough context for a robust match.
patch_size_map = {
    'sub32': 96,
    'sub16': 96,
    'sub8': 96,
    'sub4': 96,
    'sub2': 192,
    'full': 384
}

# min_wl_map defines the minimum wavelength for the phase congruency step in RIFT.
# Filtering out high-frequency noise requires a larger minimum wavelength at higher resolutions.
min_wl_map = {
    'sub32': 3,
    'sub16': 3,
    'sub8': 3,
    'sub4': 3,
    'sub2': 6,
    'full': 12
}

# Directory where the input TIFF files are located
image_folder_path = "tif_test"

# Iterate over each resolution progressively
for res in tqdm(resolutions, desc="Overall Progress"):
    print(f"\n{'='*50}")
    print(f"Testing resolution: {res}")
    print(f"{'='*50}")
    
    # Determine the filenames based on the current loop iteration's resolution
    if res == 'full':
        left_name = 'left_proj.tif'
        right_name = 'right_proj.tif'
    else:
        left_name = f'left_proj_{res}.tif'
        right_name = f'right_proj_{res}.tif'
        
    img1_path = os.path.join(image_folder_path, left_name)
    img2_path = os.path.join(image_folder_path, right_name)
    
    # Verify that the necessary image files actually exist before trying to process them
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"Skipping {res}: Files {img1_path} or {img2_path} not found.")
        continue
        
    try:
        start_mem_time = time.time()
        
        # Load the images. cv2.IMREAD_UNCHANGED ensures we don't accidentally drop bit-depth 
        # (e.g. keeping 16-bit instead of auto-converting to 8-bit).
        print(f"Loading {img1_path}...")
        img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
        print(f"Loading {img2_path}...")
        img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)
        
        # Guard against corrupt image files
        if img1 is None or img2 is None:
            print(f"Error loading images for {res}")
            continue
            
        print(f"Images loaded. Dimensions: left={img1.shape}, right={img2.shape}")
        
        # Convert images into 8-bit format while simultaneously stretching the contrast.
        # This is critical for RIFT algorithm to find reliable phase and edge maps.
        img1 = scale_to_uint8(img1)
        img2 = scale_to_uint8(img2)
        
        # Force Python's Garbage Collector to free the memory held by the unscaled 
        # original images. Crucial when working with giant 'full' resolution TIFFs.
        gc.collect()
        
        # Fetch the predefined optimization parameters tailored to this resolution.
        # If a resolution isn't in the dict, it falls back to a safe default (e.g., 5000).
        npt_val = npt_map.get(res, 5000)
        patch_size_val = patch_size_map.get(res, 96)
        min_wl_val = min_wl_map.get(res, 3)
        print(f"Initializing RIFT2 with npt={npt_val}, patch_size={patch_size_val}, minWaveLength={min_wl_val}")
        sys.stdout.flush() # Force print to terminal immediately
        
        start_time = time.time()
        
        # Run the RIFT2 feature extraction pipeline
        # joblib.parallel_backend allows multithreading across 4 CPU cores via 'loky' backend.
        # This vastly speeds up creating the complex RIFT descriptors (like Phase Congruency).
        with joblib.parallel_backend('loky', n_jobs=6):
            rift2_pipeline = RIFT2(npt=npt_val, patch_size=patch_size_val, minWaveLength=min_wl_val)
            
            # kp1, kp2: Keypoints (X,Y coordinates of features)
            # des1, des2: Descriptors (Mathematical representations of those features used for matching)
            kp1, des1, kp2, des2 = rift2_pipeline(img1, img2)
            
        end_time = time.time()
        print("RIFT2 pipeline time elapsed {:.3f} seconds".format(end_time - start_time))
        sys.stdout.flush()
        
        # MATCHING PHASE
        # We find corresponding points between the left and right image by matching 
        # the mathematical descriptors. We use Nearest Neighbors.
        time1 = time.time()
        # lowes_ratio=0.95 filters out ambiguous matches (where the 1st best match is 
        # very similar to the 2nd best match). 
        # mutual=False means a match doesn't strictly have to be reciprocal.
        points1, points2, mutual_matches = match_keypoints_nn(des1, des2, kp1, kp2, lowes_ratio=0.95, mutual=False)
        time2 = time.time()
        print("Matching time elapsed {:.3f} seconds".format(time2 - time1))
        
        # OUTLIER REMOVAL (GEOMETRIC VERIFICATION)
        # Even with good descriptors, some matches are wrong (outliers).
        # MAGSAC (a modern variant of RANSAC) estimates the geometric relationship 
        # (like an Homography or Fundamental Matrix) and throws out matches that don't fit that model.
        time1 = time.time()
        inliers1, inliers2, matchesMask = outlier_removal(points1, points2)
        time2 = time.time()
        print("Outlier removal time elapsed {:.3f} seconds".format(time2 - time1))
        print("Total time elapsed {:.3f} seconds".format(time2 - start_time))
        
        # DRAW MATCHES
        # Draw the resulting connections between the two images for visualization.
        # matchesMask ensures we only draw the good 'inlier' points in green.
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, mutual_matches, None, **draw_params)
        
        # Tally up the statistics to report to the user
        num_inliers = int(np.sum(matchesMask))
        num_outliers = len(mutual_matches) - num_inliers
        print(f'Number of kp1: {len(kp1)}')
        print(f'Number of kp2: {len(kp2)}')
        print(f'Number of matches with N.N : {len(mutual_matches)}')
        print(f'Number of inliers after MAGSAC: {num_inliers}')
        
        # Save visual output Image to disk
        out_img_name = f'result_matches_{res}.jpg'
        cv2.imwrite(out_img_name, img3)
        print(f"Saved visual matches to {out_img_name}")
        
        # Extremely important: Delete the combined image (img3 is left+right concatenated) 
        # to free RAM, as it's double the size of an individual image.
        del img3
        gc.collect()
        
        # SAVE MATCHER DATA to CSV
        # We need the direct point correlation data to use downstream 
        # (e.g. for Bundle Adjustment or Stereo matching in external software like ASP).
        csv_name = f'matches_{res}.csv'
        with open(csv_name, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['left_x', 'left_y', 'right_x', 'right_y', 'inlier'])
            
            for idx, (p1, p2) in enumerate(zip(points1, points2)):
                # Store whether MAGSAC determined this point was an inlier (1) or outlier (0)
                inlier = 1 if matchesMask[idx] else 0
                writer.writerow([p1[0], p1[1], p2[0], p2[1], inlier])
                
        print(f"Saved match coordinates to {csv_name}")
        
        # End of loop iteration. 
        # Aggressively nuke all large arrays from RAM before loading the next, higher-resolution pair.
        del img1, img2, kp1, des1, kp2, des2, points1, points2, mutual_matches, matchesMask
        gc.collect()
        
    except MemoryError:
        # A MemoryError occurs when the VM runs out of RAM. Common when 'full' resolution is huge.
        print(f"\n[ERROR] MemoryError encountered while processing {res}.")
        print("SUGGESTIONS FOR OPTIMIZATION:")
        print("1. Memory consumption is likely exceeding the VM limits (RAM/Swap).")
        print(f"2. Consider reducing the number of keypoints 'npt' (current: {npt_map.get(res, 5000)}).")
        print("3. Implement block-wise matching: splitting the large image into tiles, running RIFT2 per tile, and merging the results.")
        print("4. Decrease the number of wavelet scales 'nscale' or orientations 'norient' in the RIFT2 configuration.")
        print("5. Upgrade the VM instance to a higher RAM tier if possible.")
        sys.stderr.flush()
        break
    except Exception as e:
        # Generic error fallback to ensure the script doesn't just crash silently.
        print(f"\n[ERROR] Exception encountered during {res}: {e}")
        import traceback
        traceback.print_exc()
        break
