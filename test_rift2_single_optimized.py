"""
FLANN-ONLY OPTIMIZED VERSION of test_rift2_single.py
Identical to the baseline script in all aspects (full-image RIFT2 feature extraction,
global MAGSAC outlier removal), but replaces the slow Brute-Force Nearest Neighbor matcher
with the FLANN KD-Tree based matcher. This gives a major speedup in the matching step
(from O(N^2) to O(N log N)) without any changes to feature extraction or geometric verification.

No tiling is applied — the full image is processed as a single block to preserve
correct global geometry for MAGSAC and avoid boundary artefacts.
"""

import os
import cv2
import numpy as np
import time
import csv
import sys
import gc
import joblib
import argparse

# Import RIFT2 core and the NEW FLANN-based matcher function
from src.RIFT2 import RIFT2
from src.matcher_functions import match_keypoints_flann, outlier_removal

def scale_to_uint8(img):
    """
    Robust radiometric correction: masks NoData values (< -9999),
    then stretches contrast using 2nd–98th percentiles of valid pixels.
    NoData regions are painted black (0).
    """
    if img is None:
        return None
    if img.dtype == np.uint8:
        return img
    print(f"Applying robust radiometric correction from {img.dtype} to uint8")
    valid_mask = np.logical_and(np.isfinite(img), img > -9999)
    if not np.any(valid_mask):
        return np.zeros_like(img, dtype=np.uint8)
    valid_pixels = img[valid_mask]
    p_low, p_high = np.percentile(valid_pixels, (2, 98))
    if p_high == p_low:
        scaled = np.zeros_like(img, dtype=np.float32)
    else:
        clipped = np.clip(img, p_low, p_high)
        scaled = (clipped - p_low) / (p_high - p_low) * 255.0
    scaled = np.clip(scaled, 0, 255).astype(np.uint8)
    scaled[~valid_mask] = 0
    return scaled

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLANN-only optimized RIFT2: same full-image pipeline as baseline, FLANN matcher only.")
    parser.add_argument("--res", type=str, required=True,
                        choices=['sub32', 'sub16', 'sub8', 'sub4', 'sub2', 'full'],
                        help="Resolution level to test.")
    args = parser.parse_args()

    res = args.res
    print(f"\n{'='*50}")
    print(f"Testing (FLANN-only optimized) resolution: {res}")
    print(f"{'='*50}")

    # Parameter maps — identical to baseline
    npt_map =        { 'sub32': 5000,  'sub16': 10000, 'sub8': 20000, 'sub4': 40000, 'sub2': 80000, 'full': 150000 }
    patch_size_map = { 'sub32': 96,    'sub16': 96,    'sub8': 96,    'sub4': 96,    'sub2': 192,   'full': 384  }
    min_wl_map =     { 'sub32': 3,     'sub16': 3,     'sub8': 3,     'sub4': 3,     'sub2': 6,     'full': 12   }

    image_folder_path = "tif_test"
    left_name  = 'left_proj.tif'  if res == 'full' else f'left_proj_{res}.tif'
    right_name = 'right_proj.tif' if res == 'full' else f'right_proj_{res}.tif'

    img1_path = os.path.join(image_folder_path, left_name)
    img2_path = os.path.join(image_folder_path, right_name)

    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"Files not found: {img1_path} / {img2_path}")
        sys.exit(1)

    try:
        # ---- LOAD ----
        print(f"Loading {img1_path}...")
        img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
        print(f"Loading {img2_path}...")
        img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)

        if img1 is None or img2 is None:
            print("Failed to load images."); sys.exit(1)

        print(f"Dimensions: left={img1.shape}, right={img2.shape}")

        # ---- RADIOMETRIC CORRECTION ----
        img1 = scale_to_uint8(img1)
        img2 = scale_to_uint8(img2)
        gc.collect()

        # ---- RIFT2 FEATURE EXTRACTION (full-image, no tiling) ----
        npt_val        = npt_map.get(res, 5000)
        patch_size_val = patch_size_map.get(res, 96)
        min_wl_val     = min_wl_map.get(res, 3)
        print(f"RIFT2 config: npt={npt_val}, patch_size={patch_size_val}, minWaveLength={min_wl_val}")
        sys.stdout.flush()

        start_time = time.time()
        with joblib.parallel_backend('loky', n_jobs=6):
            rift2_pipeline = RIFT2(npt=npt_val, patch_size=patch_size_val, minWaveLength=min_wl_val)
            kp1, des1, kp2, des2 = rift2_pipeline(img1, img2)

        t_extract = time.time() - start_time
        print(f"RIFT2 extraction elapsed {t_extract:.3f} s")
        sys.stdout.flush()

        # ---- FLANN MATCHING (replaces BruteForce) ----
        # FLANN uses a KD-Tree index: O(N log N) vs BruteForce O(N^2)
        t1 = time.time()
        # FLANN uses approximate distances — a strict ratio (0.75) is needed to compensate.
        # BruteForce can use 0.95 because its distances are exact; FLANN's are not.
        # mutual=True adds an extra cross-check that significantly reduces false positives.
        points1, points2, mutual_matches = match_keypoints_flann(
            des1, des2, kp1, kp2, lowes_ratio=0.75, mutual=True)
        t_match = time.time() - t1
        print(f"FLANN matching elapsed {t_match:.3f} s")
        sys.stdout.flush()

        # ---- MAGSAC OUTLIER REMOVAL (global, unchanged from baseline) ----
        t1 = time.time()
        inliers1, inliers2, matchesMask = outlier_removal(points1, points2)
        t_magsac = time.time() - t1
        t_total = time.time() - start_time
        print(f"MAGSAC elapsed {t_magsac:.3f} s")
        print(f"Total time elapsed {t_total:.3f} s")

        num_inliers  = int(np.sum(matchesMask))
        num_outliers = len(mutual_matches) - num_inliers
        print(f"kp1: {len(kp1)}  kp2: {len(kp2)}")
        print(f"FLANN matches: {len(mutual_matches)}")
        print(f"Inliers (MAGSAC): {num_inliers}  Outliers: {num_outliers}")

        # ---- VISUALISATION ----
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None,
                           matchesMask=matchesMask, flags=2)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, mutual_matches, None, **draw_params)
        out_img = f'result_matches_{res}_flann.jpg'
        cv2.imwrite(out_img, img3)
        print(f"Saved visual preview to: {out_img}")
        del img3; gc.collect()

        # ---- PLAIN-TEXT ASP MATCH EXPORT ----
        # Export the matched left/right X,Y coordinates in plain-text format compatible with the
        # Ames Stereo Pipeline (ASP) bundle adjustment.
        # Format requirement: x1 y1 unc1 x2 y2 unc2
        # where x1, y1 are left image coordinates, x2, y2 are right image coordinates.
        # The coordinates are scaled back to the full resolution image size to allow testing 
        # on downscaled imagery while seamlessly applying the resulting match file to the 
        # original full-res pipeline.
        
        # Calculate the dynamic scaling factor required to scale points back to full-res
        if res == 'full':
            scale_factor = 1.0
        else:
            # Extract the scaling multiplier from the resolution string (e.g. 'sub4' -> 4.0)
            scale_factor = float(res.replace('sub', ''))

        match_out = f'matches_{res}_flann.match'
        with open(match_out, mode='w') as f:
            for idx, (p1, p2) in enumerate(zip(points1, points2)):
                # Only write MAGSAC-verified inliers to avoid injecting mathematically invalid points 
                # into the bundle adjustment solver.
                if matchesMask[idx]:
                    # Project local subset coordinates back to the full overarching resolution footprint
                    x1 = p1[0] * scale_factor
                    y1 = p1[1] * scale_factor
                    x2 = p2[0] * scale_factor
                    y2 = p2[1] * scale_factor
                    
                    # Uncertainty weighting: in ASP, bundle adjustment weight = 1 / uncertainty. 
                    # The uncertainties must be strictly positive.
                    # We assign a standard nominal uncertainty of 1.0 pixel for all verified matches.
                    unc1 = 1.0
                    unc2 = 1.0
                    
                    # Write the 6 space-separated floating-point values to generate the plain-text file
                    f.write(f"{x1:.4f} {y1:.4f} {unc1:.4f} {x2:.4f} {y2:.4f} {unc2:.4f}\n")
                    
        print(f"Saved scaled ASP-compatible matches to: {match_out}")

        del img1, img2, kp1, des1, kp2, des2, points1, points2, mutual_matches, matchesMask
        gc.collect()

    except MemoryError:
        print(f"\n[ERROR] MemoryError on {res}. Try reducing npt or running on a larger VM.")
        sys.exit(1)
    except Exception as e:
        import traceback; traceback.print_exc()
        sys.exit(1)
