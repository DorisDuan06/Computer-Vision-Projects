import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

if not os.path.exists('../result'):
    os.mkdir('../result')

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]

T = seq.shape[2]
It = seq[:, :, 0]
rects = np.empty((0, 4))
for t in range(T):
    It1 = seq[:, :, t]
    p = LucasKanade(It, It1, rect, threshold, num_iters)

    rect[0] += p[0]
    rect[1] += p[1]
    rect[2] += p[0]
    rect[3] += p[1]

    if t in [1, 100, 200, 300, 400]:
        x, y = rect[:2]
        width = rect[3] - rect[1] + 1
        height = rect[2] - rect[0] + 1
        fig, ax = plt.subplots(1)
        plt.imshow(It1, cmap='gray')
        ax.add_patch(patches.Rectangle((x, y), height, width, fill=False, edgecolor='r'))
        plt.show()
    It = It1
    rects = np.vstack((rects, np.array(rect)))
np.save('../result/carseqrects.npy', rects)
