import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e2, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-3, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

if not os.path.exists('../result'):
    os.mkdir('../result')

seq = np.load("../data/carseq.npy")
carseqrects = np.load("../result/carseqrects.npy")
rect0 = [59, 116, 145, 151]
rect = [59, 116, 145, 151]

T = seq.shape[2]
I0 = seq[:, :, 0]
rects = np.array(rect0)
for t in range(1, T):
    It1 = seq[:, :, t]
    It = seq[:, :, t-1]
    p = LucasKanade(It, It1, rect, threshold, num_iters)
    p_n = p + np.array([rect[0] - rect0[0], rect[1] - rect0[1]])
    p_star = LucasKanade(I0, It1, rect0, threshold, num_iters, p_n)
    if np.linalg.norm(p_star - p_n) <= template_threshold:
        rect[0] += p[0]
        rect[1] += p[1]
        rect[2] += p[0]
        rect[3] += p[1]
    else:
        rect[0] = rect0[0] + p_star[0]
        rect[1] = rect0[1] + p_star[1]
        rect[2] = rect0[2] + p_star[0]
        rect[3] = rect0[3] + p_star[1]
    rects = np.vstack((rects, np.array(rect)))

    if t in [1, 100, 200, 300, 400]:
        x_wcrt, y_wcrt = rect[:2]
        width_wcrt = rect[3] - rect[1] + 1
        height_wcrt = rect[2] - rect[0] + 1

        carrect = carseqrects[t]
        x, y = carrect[:2]
        width = carrect[3] - carrect[1] + 1
        height = carrect[2] - carrect[0] + 1
        fig, ax = plt.subplots(1)
        plt.imshow(It1, cmap='gray')
        ax.add_patch(patches.Rectangle((x_wcrt, y_wcrt), height_wcrt, width_wcrt, fill=False, edgecolor='r'))
        ax.add_patch(patches.Rectangle((x, y), height, width, fill=False, edgecolor='b'))
        plt.show()
np.save('../result/carseqrects-wcrt.npy', rects)
