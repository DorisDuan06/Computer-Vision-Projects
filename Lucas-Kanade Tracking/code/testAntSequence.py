import argparse
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from SubtractDominantMotion import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-3, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.14, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/antseq.npy')

T = seq.shape[2]
image1 = seq[:, :, 0]
for t in range(T):
    image2 = seq[:, :, t]
    mask = SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance)
    mask = ndimage.morphology.binary_dilation(mask, structure=np.ones((7, 7)))

    if t in [30, 60, 90, 120]:
        img = np.stack((image2, ) * 3, axis=-1)
        img[:, :, 0][mask] = 0.
        img[:, :, 1][mask] = 0.
        img[:, :, 2][mask] = 1.
        fig = plt.figure()
        plt.imshow(img, cmap='gray')
        plt.show()
    image1 = image2
