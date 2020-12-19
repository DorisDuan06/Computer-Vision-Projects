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
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

if not os.path.exists('../result'):
    os.mkdir('../result')

seq = np.load("../data/girlseq.npy")
girlseqrects = np.load("../result/girlseqrects.npy")
rect0 = [280, 152, 330, 318]
rect = [280, 152, 330, 318]

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

    if t in [1, 20, 40, 60, 80]:
        x_wcrt, y_wcrt = rect[:2]
        width_wcrt = rect[3] - rect[1] + 1
        height_wcrt = rect[2] - rect[0] + 1

        girlrect = girlseqrects[t]
        x, y = girlrect[:2]
        width = girlrect[3] - girlrect[1] + 1
        height = girlrect[2] - girlrect[0] + 1
        fig, ax = plt.subplots(1)
        plt.imshow(It1, cmap='gray')
        ax.add_patch(patches.Rectangle((x_wcrt, y_wcrt), height_wcrt, width_wcrt, fill=False, edgecolor='r'))
        ax.add_patch(patches.Rectangle((x, y), height, width, fill=False, edgecolor='b'))
        plt.show()
np.save('../result/girlseqrects-wcrt.npy', rects)
