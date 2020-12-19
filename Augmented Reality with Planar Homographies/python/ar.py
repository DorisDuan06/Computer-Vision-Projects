import numpy as np
import cv2

# Import necessary functions
import os
from opts import get_opts
from loadVid import loadVid
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

# Write script for Q3.1
opts = get_opts()
ar_frames = loadVid('../data/ar_source.mov')
book_frames = loadVid('../data/book.mov')
cv_cover = cv2.imread('../data/cv_cover.jpg')
f, H, W = ar_frames.shape[:3]
ar_frames = ar_frames[:, 44:-44, 200:430, :]

if not os.path.exists('../result'):
    os.mkdir('../result')

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter('../result/ar.avi', fourcc, 25, (book_frames.shape[2], book_frames.shape[1]), True)

for i in range(f):
    book = book_frames[i]
    ar = ar_frames[i]

    matches, locs1, locs2 = matchPics(book, cv_cover, opts)
    if matches.shape[0] >= 4:
        x1 = locs1[matches[:, 0], :]
        x2 = locs2[matches[:, 1], :]
        x1[:, [0, 1]] = x1[:, [1, 0]]
        x2[:, [0, 1]] = x2[:, [1, 0]]
        bestH2to1, inliers = computeH_ransac(x1, x2, opts)

        ar = cv2.resize(ar, (345, 444))
        composite_img = compositeH(bestH2to1, ar, book)
        writer.write(composite_img)
