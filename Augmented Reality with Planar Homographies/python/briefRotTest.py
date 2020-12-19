import numpy as np
import cv2
import scipy
from matplotlib import pyplot as plt
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts


# Q2.1.6
opts = get_opts()
# Read the image and convert to grayscale, if necessary
cover = cv2.imread('../data/cv_cover.jpg')

histogram = []
choice = [1, 17, 35]
for i in range(36):
    # Rotate Image
    cover_rotate = scipy.ndimage.rotate(cover, i * 10, reshape=False)

    # Compute features, descriptors and Match features
    matches, locs1, locs2 = matchPics(cover, cover_rotate, opts)
    if i in choice:
        plotMatches(cover, cover_rotate, matches, locs1, locs2)

    # Update histogram
    histogram.append(matches.shape[0])

# Display histogram
chosen_histogram = [histogram[1]] + [histogram[17]] + [histogram[35]]
plt.bar(np.arange(3), chosen_histogram)
plt.xlabel('Rotation Degrees')
plt.ylabel('Matches Count')
plt.xticks(np.arange(3), choice * 10)
plt.title('Number of Matches for BRIEF Rotation')
plt.show()
