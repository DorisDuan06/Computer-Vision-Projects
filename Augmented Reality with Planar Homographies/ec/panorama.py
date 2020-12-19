import numpy as np
import cv2

# Import necessary functions
import os
import sys

scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../python")

from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac
from matplotlib import pyplot as plt


# Write script for Q4.2x
def generate_panorama(img1, img2, H2to1):
    H, W = img1.shape[:2]
    warpped_img2 = cv2.warpPerspective(img2, H2to1, (W+667, H))

    img1 = np.hstack((img1, np.zeros((H, 667, 3)))).astype('uint8')
    panorama = np.maximum(img1, warpped_img2)
    plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.show()
    return panorama


opts = get_opts()
pano_left = cv2.imread('pano_left.jpg')
pano_right = cv2.imread('pano_right.jpg')

matches, locs1, locs2 = matchPics(pano_left, pano_right, opts)
x1 = locs1[matches[:, 0], :]
x2 = locs2[matches[:, 1], :]
x1[:, [0, 1]] = x1[:, [1, 0]]
x2[:, [0, 1]] = x2[:, [1, 0]]
bestH2to1, inliers = computeH_ransac(x1, x2, opts)

panorama = generate_panorama(pano_left, pano_right, bestH2to1)
