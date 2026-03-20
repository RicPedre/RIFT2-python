import os
import cv2
import numpy as np
import time
import csv
import sys
import gc
import joblib

from src.RIFT2 import RIFT2
from src.matcher_functions import match_keypoints_nn, draw_matches, outlier_removal

def scale_to_uint8(img):
    """Min-Max scaling to convert to 8-bit, masking out NoData and stretching using percentiles."""
    if img is None:
        return None
    
    if img.dtype == np.uint8:
        return img
        
    print(f"Applying robust radiometric correction from {img.dtype} to uint8")
    valid_mask = np.logical_and(np.isfinite(img), img > -9999)
    
    if not np.any(valid_mask):
        return np.zeros_like(img, dtype=np.uint8)
        
    # Get robust percentiles, ignoring NoData
    valid_pixels = img[valid_mask]
    p_low, p_high = np.percentile(valid_pixels, (2, 98))
    
    if p_high == p_low:
        scaled = np.zeros_like(img, dtype=np.float32)
    else:
        # Clip image values to robust min and max range
        clipped = np.clip(img, p_low, p_high)
        scaled = (clipped - p_low) / (p_high - p_low) * 255.0
        
    scaled = np.clip(scaled, 0, 255).astype(np.uint8)
    
    # Important: Set NoData regions strictly to 0 (Black)
    scaled[~valid_mask] = 0
    
    return scaled

resolutions = ['sub32', 'sub16', 'sub8', 'sub4', 'sub2', 'full']
# Dynamic optimization parameters
npt_map = {
    'sub32': 5000,
    'sub16': 10000,
    'sub8': 20000,
    'sub4': 40000,
    'sub2': 80000,
    'full': 150000
}

image_folder_path = "tif_test"

for res in resolutions:
    print(f"\n{'='*50}")
    print(f"Testing resolution: {res}")
    print(f"{'='*50}")
    
    if res == 'full':
        left_name = 'left_proj.tif'
        right_name = 'right_proj.tif'
    else:
        left_name = f'left_proj_{res}.tif'
        right_name = f'right_proj_{res}.tif'
        
    img1_path = os.path.join(image_folder_path, left_name)
    img2_path = os.path.join(image_folder_path, right_name)
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"Skipping {res}: Files {img1_path} or {img2_path} not found.")
        continue
        
    try:
        start_mem_time = time.time()
        print(f"Loading {img1_path}...")
        img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
        print(f"Loading {img2_path}...")
        img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)
        
        if img1 is None or img2 is None:
            print(f"Error loading images for {res}")
            continue
            
        print(f"Images loaded. Dimensions: left={img1.shape}, right={img2.shape}")
        
        # Radiometric correction
        img1 = scale_to_uint8(img1)
        img2 = scale_to_uint8(img2)
        
        # Free up memory (the old unscaled arrays if gc picks them)
        gc.collect()
        
        npt_val = npt_map.get(res, 5000)
        print(f"Initializing RIFT2 with npt={npt_val}")
        sys.stdout.flush()
        
        start_time = time.time()
        with joblib.parallel_backend('loky', n_jobs=4):
            rift2_pipeline = RIFT2(npt=npt_val)
            kp1, des1, kp2, des2 = rift2_pipeline(img1, img2)
        end_time = time.time()
        
        print("RIFT2 pipeline time elapsed {:.3f} seconds".format(end_time - start_time))
        sys.stdout.flush()
        
        # Matching
        time1 = time.time()
        points1, points2, mutual_matches = match_keypoints_nn(des1, des2, kp1, kp2, lowes_ratio=0.95, mutual=False)
        time2 = time.time()
        print("Matching time elapsed {:.3f} seconds".format(time2 - time1))
        
        # MAGSAC
        time1 = time.time()
        inliers1, inliers2, matchesMask = outlier_removal(points1, points2)
        time2 = time.time()
        print("Outlier removal time elapsed {:.3f} seconds".format(time2 - time1))
        print("Total time elapsed {:.3f} seconds".format(time2 - start_time))
        
        # Draw Matches
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, mutual_matches, None, **draw_params)
        
        num_inliers = int(np.sum(matchesMask))
        num_outliers = len(mutual_matches) - num_inliers
        print(f'Number of kp1: {len(kp1)}')
        print(f'Number of kp2: {len(kp2)}')
        print(f'Number of matches with N.N : {len(mutual_matches)}')
        print(f'Number of inliers after MAGSAC: {num_inliers}')
        
        # Save images
        out_img_name = f'result_matches_{res}.jpg'
        cv2.imwrite(out_img_name, img3)
        print(f"Saved visual matches to {out_img_name}")
        
        # Delete img3 to save memory
        del img3
        gc.collect()
        
        # Save to CSV
        csv_name = f'matches_{res}.csv'
        with open(csv_name, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['left_x', 'left_y', 'right_x', 'right_y', 'inlier'])
            for idx, (p1, p2) in enumerate(zip(points1, points2)):
                inlier = 1 if matchesMask[idx] else 0
                writer.writerow([p1[0], p1[1], p2[0], p2[1], inlier])
        print(f"Saved match coordinates to {csv_name}")
        
        # Free memory aggressively before next iteration
        del img1, img2, kp1, des1, kp2, des2, points1, points2, mutual_matches, matchesMask
        gc.collect()
        
    except MemoryError:
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
        print(f"\n[ERROR] Exception encountered during {res}: {e}")
        import traceback
        traceback.print_exc()
        break
