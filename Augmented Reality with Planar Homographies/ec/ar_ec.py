import numpy as np
import cv2

# Import necessary functions
import os
import sys
import time

scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../python")

from loadVid import loadVid


# Write script for Q3.1
def compositeH(H2to1, template, img):
    H, W = img.shape[:2]
    mask = np.ones(template.shape)  # cover

    warpped_mask = cv2.warpPerspective(mask, H2to1, (W, H))
    warpped_mask = (warpped_mask == 0).astype('uint8')

    warpped_template = cv2.warpPerspective(template, H2to1, (W, H))

    composite_img = img * warpped_mask + warpped_template

    return composite_img


ar_frames = loadVid('../data/ar_source.mov')
book_frames = loadVid('../data/book.mov')
cv_cover = cv2.imread('../data/cv_cover.jpg')
f, H, W = ar_frames.shape[:3]
ar_frames = ar_frames[:, 44:-44, 200:430, :]

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
locs2, desc2 = orb.detectAndCompute(cv_cover, None)  # src, model

start = time.time()
for i in range(f):
    book = book_frames[i]
    ar = ar_frames[i]

    locs1, desc1 = orb.detectAndCompute(book, None)  # dst, frame
    matches = bf.match(desc2, desc1)

    src_pts = np.float32([locs2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([locs1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H2to1, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)

    ar = cv2.resize(ar, (345, 444))
    composite_img = compositeH(H2to1, ar, book)
end = time.time()
print("FPS:", f / (end - start))
