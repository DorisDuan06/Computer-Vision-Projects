"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import sys
import numpy as np
import helper
import util
import scipy


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    n = pts1.shape[0]

    # Normalize the points
    pts1 = pts1 * 1.0 / M
    pts2 = pts2 * 1.0 / M
    T = np.array([[1.0 / M, 0, 0], [0, 1.0 / M, 0], [0, 0, 1.0]])

    x_l, y_l = pts2[:, 0], pts2[:, 1]
    x_r, y_r = pts1[:, 0], pts1[:, 1]
    A = np.array([x_l * x_r, x_l * y_r, x_l, x_r * y_l, y_l * y_r, y_l, x_r, y_r, np.ones(n)]).T
    U, S, VT = np.linalg.svd(A)
    F = np.reshape(VT.T[:, -1], (3, 3))

    F = util.refineF(F, pts1, pts2)

    # Unscale F
    F = np.dot(np.dot(T.T, F), T)
    return F


def eightpoint_non_refine(pts1, pts2, M):
    n = pts1.shape[0]

    # Normalize the points
    pts1 = pts1 * 1.0 / M
    pts2 = pts2 * 1.0 / M
    T = np.array([[1.0 / M, 0, 0], [0, 1.0 / M, 0], [0, 0, 1.0]])

    x_l, y_l = pts2[:, 0], pts2[:, 1]
    x_r, y_r = pts1[:, 0], pts1[:, 1]
    A = np.array([x_l * x_r, x_l * y_r, x_l, x_r * y_l, y_l * y_r, y_l, x_r, y_r, np.ones(n)]).T
    U, S, VT = np.linalg.svd(A)
    F = np.reshape(VT.T[:, -1], (3, 3))

    # Unscale F
    F = np.dot(np.dot(T.T, F), T)
    return F


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    E = np.dot(np.dot(K2.T, F), K1)
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    n = pts1.shape[0]

    u1 = pts1[:, 0].reshape((n, 1))
    v1 = pts1[:, 1].reshape((n, 1))
    u2 = pts2[:, 0].reshape((n, 1))
    v2 = pts2[:, 1].reshape((n, 1))

    A = np.zeros((4 * n, 4))
    A[0::4, :] = np.hstack((u1 * C1[2, 0] - C1[0, 0], u1 * C1[2, 1] - C1[0, 1], u1 * C1[2, 2] - C1[0, 2], u1 * C1[2, 3] - C1[0, 3]))
    A[1::4, :] = np.hstack((v1 * C1[2, 0] - C1[1, 0], v1 * C1[2, 1] - C1[1, 1], v1 * C1[2, 2] - C1[1, 2], v1 * C1[2, 3] - C1[1, 3]))
    A[2::4, :] = np.hstack((u2 * C2[2, 0] - C2[0, 0], u2 * C2[2, 1] - C2[0, 1], u2 * C2[2, 2] - C2[0, 2], u2 * C2[2, 3] - C2[0, 3]))
    A[3::4, :] = np.hstack((v2 * C2[2, 0] - C2[1, 0], v2 * C2[2, 1] - C2[1, 1], v2 * C2[2, 2] - C2[1, 2], v2 * C2[2, 3] - C2[1, 3]))

    w = []
    err = 0
    for i in range(n):
        # Reconstruct 3D point Pi
        Ai = A[i * 4: i * 4 + 4]
        U, S, VT = np.linalg.svd(Ai)
        Pi = VT.T[:, -1]
        wi = Pi / Pi[-1]  # wi is in homogeneous coordinates
        w.append(wi[:-1])

        # Calculate reprojection error
        x1_hat = np.dot(C1, wi)
        x2_hat = np.dot(C2, wi)
        x1_hat = x1_hat[:-1] / x1_hat[-1]
        x2_hat = x2_hat[:-1] / x2_hat[-1]
        err += np.linalg.norm(pts1[i] - x1_hat) ** 2 + np.linalg.norm(pts2[i] - x2_hat) ** 2

    w = np.array(w)
    return w, err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    H, W = im1.shape[:2]
    window_size = 10
    search_size = 60
    sigma = 5

    # Gaussian weighting of the window
    X, Y = np.mgrid[-window_size//2: window_size//2, -window_size//2: window_size//2]
    gaussian_weight = np.exp(-((X**2 + Y**2)/(2.0*sigma**2)))
    gaussian_weight = np.stack((gaussian_weight, ) * 3, axis=-1)

    im1_window = im1[y1-window_size//2: y1+window_size//2, x1-window_size//2: x1+window_size//2]
    im1_window = im1_window * gaussian_weight

    # Epipolar line
    l = np.dot(F, np.array([x1, y1, 1]))

    Y2 = np.arange(y1-search_size//2, y1+search_size//2)
    Y2 = Y2[(window_size//2 <= Y2) & (Y2 < H - window_size//2)]

    min_err = sys.maxsize
    correct_x2, correct_y2 = None, None
    for i in range(len(Y2)):
        y2 = Y2[i]
        x2 = -(l[2] + y2 * l[1]) / l[0]
        x2 = int(x2)
        im2_window = im2[y2-window_size//2: y2+window_size//2, x2-window_size//2: x2+window_size//2]
        im2_window = im2_window * gaussian_weight
        err = np.linalg.norm(im2_window - im1_window)
        if err < min_err:
            min_err = err
            correct_x2, correct_y2 = x2, y2
    return correct_x2, correct_y2


'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=4):
    n = pts1.shape[0]
    pts1_homo = np.hstack((pts1, np.ones((n, 1))))
    pts2_homo = np.hstack((pts2, np.ones((n, 1))))

    bestF = np.zeros((3, 3))
    best_inliers = 0
    best_distance = np.zeros((n, 1))
    for _ in range(nIters):
        chose_indices = np.random.choice(n, size=8, replace=False)
        pts1_chose = pts1[chose_indices, :]
        pts2_chose = pts1[chose_indices, :]

        # Use 8 points to calculate F
        F = eightpoint_non_refine(pts1_chose, pts2_chose, M)  # 3x3

        # Epipolar lines l1 and l2
        l1 = np.dot(F.T, pts2_homo.T).T
        l2 = np.dot(F, pts1_homo.T).T

        # Calculate sum of distance between each 2D point to its epipolar line
        distances = np.sum(pts1_homo * l1, axis=1)**2 / (l1[:, 0]**2 + l1[:, 1]**2)
        distances += np.sum(pts2_homo * l2, axis=1)**2 / (l2[:, 0]**2 + l2[:, 1]**2)
        inliers = np.sum(distances < tol)
        if inliers > best_inliers:
            best_inliers = inliers
            bestF = F
            best_distance = distances
    print(bestF, best_inliers)
    F = eightpoint(pts1[best_distance < tol], pts2[best_distance < tol], M)
    return F, best_distance < tol


'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    theta = np.linalg.norm(r)
    if theta == 0:
        return np.eye(3)
    u = r / theta
    u_cross = np.array([[0, -u[2, 0], u[1, 0]],
                        [u[2, 0], 0, -u[0, 0]],
                        [-u[1, 0], u[0, 0], 0]])
    R = np.eye(3) * np.cos(theta) + (1 - np.cos(theta)) * np.dot(u, u.T) + u_cross * np.sin(theta)
    return R


'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    A = (R - R.T) / 2
    p = np.array([A[2, 1], A[0, 2], A[1, 0]]).reshape((3, 1))
    s = np.linalg.norm(p)
    c = (np.trace(R) - 1) / 2.

    if s == 0 and c == 1:
        return np.zeros((3, 1))
    if s == 0 and c == -1:
        for i in range(3):
            v = (R + np.eye(3))[:, i]
            if np.linalg.norm(v) != 0:
                break
        u = v / np.linalg.norm(v)
        r = u * np.pi

        if np.linalg.norm(r) == np.pi and ((r[0] == r[1] == 0 and r[2] < 0) or (r[0] == 0 and r[1] < 0) or r[0] < 0):
            r = -r

    u = p / s
    theta = arctan2(s, c)
    r = u * theta
    return r


def arctan2(y, x):
    if x > 0:
        return np.arctan(y / x)
    if x < 0:
        return np.pi + np.arctan(y / x)
    if x == 0 and y > 0:
        return np.pi / 2
    if x == 0 and y < 0:
        return -np.pi / 2


'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    P = x[:-6].reshape((-1, 3))  # N x 3
    r2 = x[-6:-3].reshape((3, 1))
    t2 = x[-3:].reshape((3, 1))

    N = P.shape[0]
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))

    # Project 3D points into 2D
    P_homo = np.hstack((P, np.ones((N, 1))))
    p1_hat = np.dot(np.dot(K1, M1), P_homo.T).T  # N x 3
    p2_hat = np.dot(np.dot(K2, M2), P_homo.T).T
    p1_hat = p1_hat[:, :-1] / p1_hat[:, -1].reshape((N, 1))  # N x 2
    p2_hat = p2_hat[:, :-1] / p2_hat[:, -1].reshape((N, 1))

    residuals = np.concatenate([(p1 - p1_hat).reshape([-1]), (p2 - p2_hat).reshape([-1])]).reshape((4*N, 1))
    return residuals


'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Prepare parameter x for error function rodriguesResidual()
    R2_init = M2_init[:, :3]
    t2_init = M2_init[:, -1].reshape((3, 1))
    r2_init = invRodrigues(R2_init)
    x = np.concatenate([P_init.flatten().reshape((-1, 1)), r2_init, t2_init])

    # Find least square of residual function
    residual_func = lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x).flatten()
    sol, _ = scipy.optimize.leastsq(residual_func, x)

    # Extract M2 and P2 from solution
    P2 = x[:-6].reshape((-1, 3))  # N x 3
    r2 = x[-6:-3].reshape((3, 1))
    t2 = x[-3:].reshape((3, 1))
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))

    return M2, P2
