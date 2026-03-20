import os
import cv2
import numpy as np
from src.RIFT2 import RIFT2
from src.matcher_functions import match_keypoints_nn,draw_matches,outlier_removal
import time

image_folder_path = "sar-sar/proj"
img1_path = os.path.join(image_folder_path, 'left_proj.bmp')
img2_path = os.path.join(image_folder_path, 'right_proj.bmp')

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

start_time = time.time()
rift2_pipeline = RIFT2()
kp1, des1, kp2, des2 = rift2_pipeline(img1, img2)
end_time = time.time()
#print information
print("RIFT2 pipeline time elapsed {:.3f} seconds".format(end_time - start_time))

# Perform keypoint matching
time1 = time.time()
points1, points2, mutual_matches = match_keypoints_nn(des1, des2, kp1, kp2, lowes_ratio=0.95, mutual=False)
time2 = time.time()
print("Matching time elapsed {:.3f} seconds".format(time2 - time1))

# Outlier removal using MAGSAC
time1 = time.time()
inliers1, inliers2, matchesMask = outlier_removal(points1, points2)
time2 = time.time()
print("Outlier removal time elapsed {:.3f} seconds".format(time2 - time1))
print("Total time elapsed {:.3f} seconds".format(time2 - start_time))

# Draw matches (calculate params here to avoid GUI)
draw_params = dict(matchColor=(0, 255, 0),
                    singlePointColor=None,
                    matchesMask=matchesMask,
                    flags=2)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, mutual_matches, None, **draw_params)

# Print statistics
num_inliers = np.sum(matchesMask)
num_outliers = len(mutual_matches) - num_inliers
print(f'Number of kp1: {len(kp1)}')
print(f'Number of kp2: {len(kp2)}')
print(f'Number of matches with N.N : {len(mutual_matches)}')
print(f'Number of inliers after MAGSAC: {num_inliers}')
print(f'Number of outliers after MAGSAC: {num_outliers}')

# Save result to file instead of showing
cv2.imwrite('result_matches.jpg', img3)
print("Result saved internally as 'result_matches.jpg'")


