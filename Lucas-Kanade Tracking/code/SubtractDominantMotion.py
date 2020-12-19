import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine


def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """

    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)
    M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    p = M.flatten()

    H1, W1 = image1.shape
    H2, W2 = image2.shape
    image1_interpolate = RectBivariateSpline(np.arange(0, H1, 1), np.arange(0, W1, 1), image1)
    image2_interpolate = RectBivariateSpline(np.arange(0, H2, 1), np.arange(0, W2, 1), image2)

    x1_warp, y1_warp = np.meshgrid(np.arange(0, W1, 1), np.arange(0, H1, 1))
    image1_warp = image1_interpolate.ev(y1_warp, x1_warp)

    x2, y2 = np.meshgrid(np.arange(0, W2, 1), np.arange(0, H2, 1))
    x2_warp = p[0] * x2 + p[1] * y2 + p[2]
    y2_warp = p[3] * x2 + p[4] * y2 + p[5]
    image2_warp = image2_interpolate.ev(y2_warp, x2_warp)

    ''' Uncomment below for Ant tracking, since applying binary_erosion and
    binary dilation on warpped image segments out black objects (they have
    pixel values 0) '''
    # tolerance = 0.8
    # image2_warp = ndimage.morphology.binary_erosion(image2_warp)
    # image2_warp = ndimage.morphology.binary_dilation(image2_warp)
    ''' Comment them out for Aerial tracking '''

    subtraction = np.abs(image2_warp - image1_warp)
    non_overlap = (x2_warp < 0) | (x2_warp >= W2) | (y2_warp < 0) | (y2_warp >= H2)
    mask = subtraction > tolerance
    mask[non_overlap] = 0

    return mask
