import numpy as np
import cv2


def computeH(x1, x2):
    # Q2.2.1
    # Compute the homography between two sets of points
    N = x1.shape[0]
    x1_1, x1_2 = x1[:, 0].reshape((N, 1)), x1[:, 1].reshape((N, 1))
    x2_1, x2_2 = x2[:, 0].reshape((N, 1)), x2[:, 1].reshape((N, 1))

    A = np.zeros((2 * N, 9))
    A[0::2, :3] = np.hstack((x2_1, x2_2, np.ones((N, 1))))
    A[0::2, -3:] = np.hstack((-x2_1 * x1_1, -x2_2 * x1_1, -x1_1))
    A[1::2, 3:] = np.hstack((x2_1, x2_2, np.ones((N, 1)), -x2_1 * x1_2, -x2_2 * x1_2, -x1_2))
    U, S, VT = np.linalg.svd(A)
    H2to1 = np.reshape(VT.T[:, -1], (3, 3))

    return H2to1


def computeH_norm(x1, x2):
    # Q2.2.2
    # Compute the centroid of the points
    centroid1 = np.mean(x1, axis=0)
    centroid2 = np.mean(x2, axis=0)

    # Shift the origin of the points to the centroid
    x1_new = x1 - centroid1
    x2_new = x2 - centroid2

    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    max_coordinate1 = np.amax(np.abs(x1_new))
    max_coordinate2 = np.amax(np.abs(x2_new))
    x1_new /= max_coordinate1
    x2_new /= max_coordinate2

    # Similarity transform 1
    T1 = np.array([[1/max_coordinate1, 0, -centroid1[0]/max_coordinate1],
                   [0, 1/max_coordinate1, -centroid1[1]/max_coordinate1],
                   [0, 0, 1]])

    # Similarity transform 2
    T2 = np.array([[1/max_coordinate2, 0, -centroid2[0]/max_coordinate2],
                   [0, 1/max_coordinate2, -centroid2[1]/max_coordinate2],
                   [0, 0, 1]])

    # Compute homography
    H = computeH(x1_new, x2_new)

    # Denormalization
    H2to1 = np.matmul(np.matmul(np.linalg.inv(T1), H), T2)

    return H2to1


def computeH_ransac(locs1, locs2, opts):
    # Q2.2.3
    # Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol  # the tolerance value for considering a point to be an inlier

    N = locs1.shape[0]
    x2_homo = np.hstack((locs2, np.ones((N, 1)))).T

    bestH2to1 = np.zeros((3, 3))
    best_inliers = 0

    for _ in range(max_iters):
        chose_indices = np.random.choice(N, size=4, replace=False)
        x1_chose = locs1[chose_indices, :]
        x2_chose = locs2[chose_indices, :]

        H2to1 = computeH_norm(x1_chose, x2_chose)

        x1_transformed = H2to1.dot(x2_homo)
        x1_transformed = x1_transformed / (x1_transformed[-1, :] + 1e-10)
        distances = np.linalg.norm(x1_transformed[:-1, :].T - locs1, axis=1)
        inliers = np.sum(distances < inlier_tol)
        if inliers > best_inliers:
            best_inliers = inliers
            bestH2to1 = H2to1

    return bestH2to1, best_inliers


def compositeH(H2to1, template, img):

    # Create a composite image after warping the template image on top
    # of the image using the homography

    # Note that the homography we compute is from the image to the template;
    # x_template = H2to1*x_photo
    # For warping the template to the image, we need to invert it.

    # Create mask of same size as template
    H, W = img.shape[:2]
    mask = np.ones(template.shape[:2])  # cover

    # Warp mask by appropriate homography
    warpped_mask = cv2.warpPerspective(mask, H2to1, (W, H))

    # Warp template by appropriate homography
    warpped_template = cv2.warpPerspective(template, H2to1, (W, H))

    # Use mask to combine the warped template and the image
    composite_img = img
    for i in range(H):
        for j in range(W):
            if warpped_mask[i, j] == 1:
                composite_img[i, j, :] = warpped_template[i, j, :]

    return composite_img
