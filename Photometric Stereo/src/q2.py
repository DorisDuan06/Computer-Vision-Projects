# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# Dec 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface
from utils import enforceIntegrability
from matplotlib import pyplot as plt


def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions.

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """

    u, s, vh = np.linalg.svd(I, full_matrices=False)
    sigma = np.zeros((3, 3))
    u = u[:, :3]
    vh = vh[:3]
    for i in range(3):
        sigma[i, i] = s[i]
    B = np.dot(sigma**(1/2), vh)
    L = np.dot(u, sigma**(1/2)).T

    return B, L

if __name__ == "__main__":
    # Load images into I, light directions into L0, shape into s
    I, L0, s = loadData()

    # Question 2 (b)
    B, L = estimatePseudonormalsUncalibrated(I)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    # Question 2 (c)
    print("Estimated L:", L)
    print("Ground truth L:", L0)

    # Question 2 (d)
    surface = estimateShape(normals, s)
    plotSurface(surface)

    # Question 2 (e)
    Nt = enforceIntegrability(B, s)
    surface = estimateShape(Nt, s)
    plotSurface(surface)

    # Question 2 (f)
    mu, nu = 0, 0
    for lamda in [0.1, 0.5]:
        G = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [mu, nu, lamda]])
        new_B = np.dot(np.linalg.inv(G.T), Nt)
        surface = estimateShape(new_B, s)
        plotSurface(surface)

    lamda, nu = 0.7, 0
    for mu in [1.5, 3]:
        G = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [mu, nu, lamda]])
        new_B = np.dot(np.linalg.inv(G.T), Nt)
        surface = estimateShape(new_B, s)
        plotSurface(surface)

    lamda, mu = 0.7, 0
    for nu in [1.5, 3]:
        G = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [mu, nu, lamda]])
        new_B = np.dot(np.linalg.inv(G.T), Nt)
        surface = estimateShape(new_B, s)
        plotSurface(surface)
