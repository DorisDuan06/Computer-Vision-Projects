import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """

    # Put your implementation here
    p = p0
    H0, W0 = It.shape
    H1, W1 = It1.shape

    It_interpolate = RectBivariateSpline(np.arange(0, H0, 1), np.arange(0, W0, 1), It)
    It1_interpolate = RectBivariateSpline(np.arange(0, H1, 1), np.arange(0, W1, 1), It1)

    x1, y1, x2, y2 = rect

    # Evaluate the template
    x = np.linspace(x1, x2, x2 - x1 + 1)
    y = np.linspace(y1, y2, y2 - y1 + 1)
    xi, yi = np.meshgrid(x, y)
    It_warp = It_interpolate.ev(yi, xi)

    for _ in range(int(num_iters)):
        # Warp It+1
        x = np.linspace(x1 + p[0], x2 + p[0], x2 - x1 + 1)
        y = np.linspace(y1 + p[1], y2 + p[1], y2 - y1 + 1)
        xi_warp, yi_warp = np.meshgrid(x, y)
        It1_warp = It1_interpolate.ev(yi_warp, xi_warp)

        # Gradient of warpped It+1
        A = np.array([It1_interpolate.ev(yi_warp, xi_warp, dx=0, dy=1).flatten(),
                      It1_interpolate.ev(yi_warp, xi_warp, dx=1, dy=0).flatten()]).T
        b = It_warp - It1_warp
        delta_p = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), b.flatten())
        p = p + delta_p

        if np.linalg.norm(delta_p) < threshold:
            break
    return p
