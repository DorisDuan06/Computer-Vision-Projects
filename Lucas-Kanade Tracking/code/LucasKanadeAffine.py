import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.eye(3)
    p = M[:2].flatten()
    H0, W0 = It.shape
    H1, W1 = It1.shape

    It1_interpolate = RectBivariateSpline(np.arange(0, H1, 1), np.arange(0, W1, 1), It1)

    for _ in range(int(num_iters)):
        # Warp It+1
        x1, y1 = np.meshgrid(np.arange(0, W1, 1), np.arange(0, H1, 1))
        x1_warp = p[0] * x1 + p[1] * y1 + p[2]
        y1_warp = p[3] * x1 + p[4] * y1 + p[5]

        # Only get common region for It and warpped It+1
        overlap = (0 <= x1_warp) & (x1_warp < W1) & (0 <= y1_warp) & (y1_warp < H1)
        x1_warp = x1_warp[overlap]
        y1_warp = y1_warp[overlap]
        It1_warp = It1_interpolate.ev(y1_warp, x1_warp)

        # Gradient of warpped It+1
        delta_I = np.array([It1_interpolate.ev(y1_warp, x1_warp, dx=0, dy=1).flatten(),
                            It1_interpolate.ev(y1_warp, x1_warp, dx=1, dy=0).flatten()]).T

        A = np.array([delta_I[:, 0] * x1_warp.flatten(), delta_I[:, 0] * y1_warp.flatten(), delta_I[:, 0], delta_I[:, 1] * x1_warp.flatten(), delta_I[:, 1] * y1_warp.flatten(), delta_I[:, 1]]).T
        b = It[overlap] - It1_warp
        delta_p = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), b.flatten())
        p = p + delta_p

        if np.linalg.norm(delta_p) < threshold:
            break
    M = np.concatenate((p, np.array([0, 0, 1]))).reshape((3, 3))
    return M
