import numpy as np
from scipy.interpolate import RectBivariateSpline


def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array]
    """

    # put your implementation here
    M = np.eye(3)
    H0, W0 = It.shape
    H1, W1 = It1.shape

    It_interpolate = RectBivariateSpline(np.arange(0, H0, 1), np.arange(0, W0, 1), It)
    It1_interpolate = RectBivariateSpline(np.arange(0, H1, 1), np.arange(0, W1, 1), It1)

    # Gradient of warpped It
    x_warp, y_warp = np.meshgrid(np.arange(0, W0, 1), np.arange(0, H0, 1))
    delta_I = np.array([It_interpolate.ev(y_warp, x_warp, dx=0, dy=1).flatten(),
                        It_interpolate.ev(y_warp, x_warp, dx=1, dy=0).flatten()]).T
    A_origin = np.array([delta_I[:, 0] * x_warp.flatten(), delta_I[:, 0] * y_warp.flatten(), delta_I[:, 0], delta_I[:, 1] * x_warp.flatten(), delta_I[:, 1] * y_warp.flatten(), delta_I[:, 1]]).T

    for _ in range(int(num_iters)):
        # Warp It+1
        x1, y1 = np.meshgrid(np.arange(0, W1, 1), np.arange(0, H1, 1))
        x1_warp = M[0, 0] * x1 + M[0, 1] * y1 + M[0, 2]
        y1_warp = M[1, 0] * x1 + M[1, 1] * y1 + M[1, 2]

        # Only get common region for It and warpped It+1
        overlap = (0 <= x1_warp) & (x1_warp < W1) & (0 <= y1_warp) & (y1_warp < H1)
        x1_warp = x1_warp[overlap]
        y1_warp = y1_warp[overlap]
        It1_warp = It1_interpolate.ev(y1_warp, x1_warp)

        A = A_origin[overlap.flatten()]
        b = It1_warp - It[overlap]
        delta_p = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), b.flatten())
        delta_M = np.concatenate((delta_p, np.array([0, 0, 1]))).reshape((3, 3))
        delta_M[0, 0] += 1
        delta_M[1, 1] += 1
        M = np.dot(M, np.linalg.inv(delta_M))

        if np.linalg.norm(delta_p) < threshold:
            break

    return M
