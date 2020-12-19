'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper


def no_bundle_adjustment():
    data = np.load('../data/some_corresp.npz')
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    N = data['pts1'].shape[0]
    M = max(im1.shape[0], im1.shape[1])

    # Estimate fundamental matrix F
    F = sub.eightpoint(data['pts1'], data['pts2'], M)

    # Get essential matrix E
    intrinsics = np.load('../data/intrinsics.npz')
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']
    E = sub.essentialMatrix(F, K1, K2)

    # Read temple coordinates file
    temple_coords = np.load('../data/templeCoords.npz')
    x1_temple = temple_coords['x1']
    y1_temple = temple_coords['y1']

    # Get 2D points in image2 using F
    n = x1_temple.shape[0]
    x2_temple, y2_temple = [], []
    for i in range(n):
        x2, y2 = sub.epipolarCorrespondence(im1, im2, F, x1_temple[i, 0], y1_temple[i, 0])
        x2_temple.append(x2)
        y2_temple.append(y2)
    x2_temple = np.array(x2_temple).reshape((n, 1))
    y2_temple = np.array(y2_temple).reshape((n, 1))

    pts1_temple = np.hstack((x1_temple, y1_temple))
    pts2_temple = np.hstack((x2_temple, y2_temple))

    # Get four possible decomposition M2 from E
    M2s = helper.camera2(E)

    # Testing four M2 through triangulation to get a correct M2
    M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    C1 = np.dot(K1, M1)

    errs = np.zeros(4)
    for i in range(4):
        M2 = M2s[:, :, i]
        C2 = np.dot(K2, M2)
        w, err = sub.triangulate(C1, pts1_temple, C2, pts2_temple)
        if np.all(w[:, -1] > 0):
            break
    print("Reprojection error with no bundle adjustment:", err)

    np.savez('q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)

    # Get 3D points
    P_temple = w

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P_temple[:, 0], P_temple[:, 1], P_temple[:, 2], marker='o', s=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show(block=True)


def bundle_adjustment():
    data = np.load('../data/some_corresp_noisy.npz')
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    N = data['pts1'].shape[0]
    M = max(im1.shape[0], im1.shape[1])

    # Estimate fundamental matrix F
    F = sub.eightpoint(data['pts1'], data['pts2'], M)

    # Get 2D points and essential matrix E
    intrinsics = np.load('../data/intrinsics.npz')
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']
    pts1 = data['pts1']
    pts2 = data['pts2']
    F, inliers = sub.ransacF(pts1, pts2, M)
    E = sub.essentialMatrix(F, K1, K2)

    # Get four possible decomposition M2 from E
    M2s = helper.camera2(E)

    # Testing four M2 through triangulation to get a correct M2
    M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    C1 = np.dot(K1, M1)

    for i in range(4):
        M2 = M2s[:, :, i]
        C2 = np.dot(K2, M2)
        w, err = sub.triangulate(C1, pts1[inliers], C2, pts2[inliers])
        if np.all(w[:, -1] > 0):
            break
    print("Reprojection error with bundle adjustment:", err)

    # Get 3D points
    M2, P = sub.bundleAdjustment(K1, M1, pts1[inliers], K2, M2, pts2[inliers], w)
    print("Optimized matrix:", M2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], marker='o', s=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show(block=True)


if __name__ == '__main__':
    # Q4.2
    no_bundle_adjustment()

    # Q5.3
    # bundle_adjustment()
