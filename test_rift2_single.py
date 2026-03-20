"""
This script runs the RIFT2 (Radiation-variation Insensitive Feature Transform) algorithm
to find matching points (keypoints) between pairs of multimodal images (e.g., optical and SAR)
for a SINGLE specified resolution level.

It handles essential preprocessing, matching, and outlier removal, while carefully managing
system memory to avoid crashes on large images, and scales algorithmic parameters 
dynamically based on the selected resolution.
"""

import os        # For interacting with the operating system, like joining file paths
import cv2       # OpenCV for image processing tasks (loading images, drawing matches)
import numpy as np # NumPy for fast numerical array operations (matrices, math)
import time      # For timing how long different steps take
import csv       # For writing the final matching points to a CSV file
import sys       # For system-level operations like flushing standard output
import gc        # Garbage Collector interface to manually free up memory when needed
import joblib    # For running the RIFT2 pipeline in parallel across multiple CPU cores
import argparse  # For parsing command-line arguments (e.g., --res)
from tqdm import tqdm # For displaying progress bars in the terminal

# Import the core RIFT2 algorithm and helper matching functions from local source files
from src.RIFT2 import RIFT2
from src.matcher_functions import match_keypoints_nn, draw_matches, outlier_removal

def scale_to_uint8(img):
    """
    Min-Max scaling to convert raw image data (which might be floats or 16-bit) to 8-bit (uint8).
    8-bit format is exactly what is required by many computer vision algorithms, including OpenCV tools.
    
    This function specifically:
    1. Masks out invalid 'NoData' pixels (often set to extreme values like -32768 in GeoTIFFs).
    2. Stretches the contrast using the 2nd and 98th percentiles inside the valid regions 
       to ignore extreme dark/bright outliers, maximizing visibility of radar features.
    """
    # Guard against empty image inputs
    if img is None:
        return None
        
    # If the image is already 8-bit, no scaling is needed.
    if img.dtype == np.uint8:
        return img
        
    print(f"Applying robust radiometric correction from {img.dtype} to uint8")
    
    # Create a mask to identify valid pixels. 
    # This prevents the NoData values (e.g. -32768) from skewing the contrast algorithm.
    # np.isfinite ensures we don't include NaN (Not a Number) or Infinity values.
    valid_mask = np.logical_and(np.isfinite(img), img > -9999)
    
    # If there are no valid pixels at all, return an empty (black) image
    if not np.any(valid_mask):
        return np.zeros_like(img, dtype=np.uint8)
        
    # Get robust percentiles (2nd and 98th), considering only valid pixels.
    # This prevents a single super-bright/super-dark pixel from ruining the 0-255 stretch range.
    valid_pixels = img[valid_mask]
    p_low, p_high = np.percentile(valid_pixels, (2, 98))
    
    # If the image is completely uniform (all valid pixels have the exact same value)
    if p_high == p_low:
        scaled = np.zeros_like(img, dtype=np.float32)
    else:
        # Clip individual pixel values to fall strictly within our robust min/max range
        clipped = np.clip(img, p_low, p_high)
        # Scale the remaining clipped values to distribute evenly between 0 and 255
        scaled = (clipped - p_low) / (p_high - p_low) * 255.0
        
    # Ensure all values are purely 8-bit integers by casting
    scaled = np.clip(scaled, 0, 255).astype(np.uint8)
    
    # Very important: Re-apply the NoData mask. Set these omitted regions strictly to 0 (Black).
    # This ensures that RIFT2 doesn't try finding keypoints inside empty space.
    scaled[~valid_mask] = 0
    
    return scaled

if __name__ == "__main__":
    # Setup the command line argument parser
    parser = argparse.ArgumentParser(description="Test RIFT2 matching on a single designated resolution pair.")
    parser.add_argument("--res", type=str, required=True, choices=['sub32', 'sub16', 'sub8', 'sub4', 'sub2', 'full'],
                        help="The predefined resolution prefix to test.")
    args = parser.parse_args()

    # Capture the user's choice
    res = args.res
    print(f"\n{'='*50}")
    print(f"Testing resolution: {res}")
    print(f"{'='*50}")

    # Dynamic optimization parameters tailored for each relative resolution scale.
    # As the physical image size (pixels) increases, we must scale up parameters 
    # to maintain the original physical spatial context.

    # npt_map controls the maximum cap of keypoints to extract.
    # Huge resolution arrays hold more data, requiring more keypoint samples for full coverage.
    npt_map = { 'sub32': 5000, 'sub16': 10000, 'sub8': 20000, 'sub4': 40000, 'sub2': 80000, 'full': 150000 }
    
    # patch_size_map determines the localized 'window' size around a feature.
    # Since we can't downsample the large images natively, we must DOUBLE the patch size 
    # for each spatial doubling (e.g., sub4->sub2) to guarantee the feature descriptor observes 
    # the exact same geographical extent. If left at 96, a patch on 'sub2' would be too tiny and fail.
    patch_size_map = { 'sub32': 96, 'sub16': 96, 'sub8': 96, 'sub4': 96, 'sub2': 192, 'full': 384 }
    
    # min_wl_map determines the minimum wavelength cut-off for the phase congruency math.
    # Filtering out noisy high-frequency artifacts requires a progressively larger wavelength at higher resolutions.
    min_wl_map = { 'sub32': 3, 'sub16': 3, 'sub8': 3, 'sub4': 3, 'sub2': 6, 'full': 12 }

    # Set the target folder and interpolate the expected file names
    image_folder_path = "tif_test"
    if res == 'full':
        left_name = 'left_proj.tif'
        right_name = 'right_proj.tif'
    else:
        left_name = f'left_proj_{res}.tif'
        right_name = f'right_proj_{res}.tif'

    # Stitch the full relative paths together
    img1_path = os.path.join(image_folder_path, left_name)
    img2_path = os.path.join(image_folder_path, right_name)

    # Sanity check: abort if the expected images don't exist
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"Skipping {res}: Required files {img1_path} or {img2_path} were not found on disk.")
        sys.exit(1)

    try:
        start_mem_time = time.time()
        
        # Load the images from disk. cv2.IMREAD_UNCHANGED ensures we don't accidentally drop bit-depth 
        # (e.g., preserving raw 16-bit float values instead of having OpenCV brutally auto-convert to 8-bit).
        print(f"Loading {img1_path}...")
        img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
        print(f"Loading {img2_path}...")
        img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)

        # Fallback guard against corrupt or malformed image files
        if img1 is None or img2 is None:
            print(f"Error loading image containers for {res}")
            sys.exit(1)

        print(f"Images successfully loaded. Native Dimensions: left={img1.shape}, right={img2.shape}")

        # Pipe the raw arrays through our pre-processing function to stretch intensities safely to 8-bit.
        img1 = scale_to_uint8(img1)
        img2 = scale_to_uint8(img2)
        
        # Flush the Python garbage collector to forcefully release the memory held by the huge raw unscaled arrays.
        gc.collect()

        # Fetch our targeted scaling variables
        npt_val = npt_map.get(res, 5000)
        patch_size_val = patch_size_map.get(res, 96)
        min_wl_val = min_wl_map.get(res, 3)
        print(f"Initializing RIFT2 core with npt={npt_val}, patch_size={patch_size_val}, minWaveLength={min_wl_val}")
        
        # sys.stdout.flush() forces the print message to the terminal immediately, overriding arbitrary buffer delays.
        sys.stdout.flush()

        # Execute Parallel RIFT2 Feature Matching Routine
        start_time = time.time()
        
        # Using a joblib context natively restricts parallel threads to 6 (preventing loky fork bomb freezes).
        with joblib.parallel_backend('loky', n_jobs=6):
            # Instantiate the RIFT2 core algorithm logic, passing in our dynamically boosted variables
            rift2_pipeline = RIFT2(npt=npt_val, patch_size=patch_size_val, minWaveLength=min_wl_val)
            
            # Fire the main callable: extracts keypoint coordinates (kp1, kp2) and mathematical array descriptors (des1, des2)
            kp1, des1, kp2, des2 = rift2_pipeline(img1, img2)

        end_time = time.time()
        print("RIFT2 pipeline extraction elapsed {:.3f} seconds".format(end_time - start_time))
        sys.stdout.flush()

        # ==========================
        # MATCHING DESCRIPTORS PHASE
        # ==========================
        # Calculate Nearest Neighbors distances to find candidate pairings between features
        time1 = time.time()
        
        # lowes_ratio=0.95 rejects highly ambiguous matches (where the best match is mathematically too 
        # identical to the 2nd best match, indicating a repetitive textured pattern).
        # mutual=False means a match relationship doesn't strictly have to be verified in reverse to be included.
        points1, points2, mutual_matches = match_keypoints_nn(des1, des2, kp1, kp2, lowes_ratio=0.95, mutual=False)
        time2 = time.time()
        print("Matching Nearest Neighbors elapsed {:.3f} seconds".format(time2 - time1))

        # =============================
        # GEOMETRIC OUTLIER ELIMINATION
        # =============================
        # Even the best mathematical descriptors will produce some completely false pairings (outliers).
        # MAGSAC acts as an intelligent geometric sieve. It models a transformation plane (Homography matrix)
        # between the images and marks any match points that don't obey that transformation model as outliers.
        time1 = time.time()
        inliers1, inliers2, matchesMask = outlier_removal(points1, points2)
        time2 = time.time()
        print("MAGSAC geometric outlier removal elapsed {:.3f} seconds".format(time2 - time1))
        print("Total processing time elapsed {:.3f} seconds".format(time2 - start_time))

        # ============================
        # VISUALIZATION GRAPHIC OUTPUT
        # ============================
        # Generate a dual-pane image drawing colored lines connecting our identical points across images.
        # matchesMask array guarantees we only physically draw lines for verified 'inliers' (throwing out junk matches).
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, mutual_matches, None, **draw_params)

        # Statistical rundown
        num_inliers = int(np.sum(matchesMask))
        num_outliers = len(mutual_matches) - num_inliers
        print(f'Total candidates in Left Img: {len(kp1)}')
        print(f'Total candidates in Right Img: {len(kp2)}')
        print(f'Raw matched pairs from Nearest Neighbors: {len(mutual_matches)}')
        print(f'Verified solid geometric Inliers (MAGSAC output): {num_inliers}')

        # Save the drawn visual comparison straight to disk
        out_img_name = f'result_matches_{res}.jpg'
        cv2.imwrite(out_img_name, img3)
        print(f"Saved visual match map preview to: {out_img_name}")
        
        # Clean up visualization footprint: img3 concatenates both images side-by-side (200% memory), delete it immediately.
        del img3
        gc.collect()

        # ==============================
        # CSV RECORD EXPORT FOR PIPELINE
        # ==============================
        # Export the matched left/right X,Y coordinates. This generic CSV format can then be independently
        # digested, evaluated, or translated to .match binary files for execution in the AMES Stereo Pipeline (ASP).
        csv_name = f'matches_{res}.csv'
        with open(csv_name, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['left_x', 'left_y', 'right_x', 'right_y', 'inlier'])
            
            # Iterate sequentially through the correlated points 
            for idx, (p1, p2) in enumerate(zip(points1, points2)):
                # Write a binary '1' if MAGSAC validated this feature point, otherwise '0'
                inlier = 1 if matchesMask[idx] else 0
                writer.writerow([p1[0], p1[1], p2[0], p2[1], inlier])
                
        print(f"Saved raw X/Y match mapping coordinates to: {csv_name}")

        # Drop large structural references ahead of Python shutdown to prevent delayed garbage collection faults
        del img1, img2, kp1, des1, kp2, des2, points1, points2, mutual_matches, matchesMask
        gc.collect()

    except MemoryError:
        # Failsafe specifically for out-of-memory deaths. Prevents a generic lockup by gracefully capturing the Exception
        # and printing recommendations that you might execute if the VM starts thrashing swap memory.
        print(f"\n[FATAL ERROR] MemoryError encountered while processing {res}.")
        print("Your VM exhausted available RAM/Swap allocations. Consider running with smaller NPT maps or scaling down the workers (loky n_jobs=2).")
        sys.exit(1)
    except Exception as e:
        # Blanket exception catch for any undefined failure strings (OpenCV bindings exceptions, permission denials, etc.)
        print(f"\n[ERROR] Exception encountered during {res} execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
