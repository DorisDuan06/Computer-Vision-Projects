import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

import os
import matplotlib.pyplot as plt

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions
    image_denoised = skimage.restoration.denoise_wavelet(image, multichannel=True)  # denoise
    image_grey = skimage.color.rgb2grey(image_denoised)  # greyscale
    thresh = skimage.filters.threshold_otsu(image_grey)  # threshold
    image_binary = image_grey > thresh
    bw = skimage.morphology.opening(image_binary, np.ones((5, 5)))  # morphology
    image_label, num = skimage.measure.label(bw, background=1, connectivity=2, return_num=True)  # label

    for i in range(num):  # skip small boxes
        ys, xs = np.where(image_label == i + 1)
        y1, x1, y2, x2 = ys.min(), xs.min(), ys.max(), xs.max()
        if y2 - y1 < 30:
            continue
        bboxes.append([y1, x1, y2, x2])
    return bboxes, bw
