'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper


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

# Get four possible decomposition M2 from E
M2s = helper.camera2(E)

# Testing four M2 through triangulation to get a correct M2
M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
C1 = np.dot(K1, M1)

for i in range(4):
    M2 = M2s[:, :, i]
    C2 = np.dot(K2, M2)
    w, _ = sub.triangulate(C1, data['pts1'], C2, data['pts2'])
    if np.all(w[:, -1] > 0):
        break

np.savez('q3_3.npz', M2=M2, C2=C2, P=w)
