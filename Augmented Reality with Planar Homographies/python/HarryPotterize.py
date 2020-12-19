import numpy as np
import cv2
import skimage.io
import skimage.color
from opts import get_opts

# Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
from matplotlib import pyplot as plt

# Write script for Q2.2.4
opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

matches, locs1, locs2 = matchPics(cv_desk, cv_cover, opts)
x1 = locs1[matches[:, 0], :]
x2 = locs2[matches[:, 1], :]
x1[:, [0, 1]] = x1[:, [1, 0]]
x2[:, [0, 1]] = x2[:, [1, 0]]
bestH2to1, inliers = computeH_ransac(x1, x2, opts)

H, W = cv_cover.shape[:2]
hp_cover = cv2.resize(hp_cover, (W, H))
composite_img = compositeH(bestH2to1, hp_cover, cv_desk)

plt.imshow(cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB))
plt.show()
