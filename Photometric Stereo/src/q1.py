# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# Dec 2020
# ##################################################################### #

# Imports
import numpy as np
import scipy
from skimage.io import imread
from skimage.color import rgb2xyz
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image

from utils import integrateFrankot

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centered on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the sphere in an array of size (3,)

    rad : float
        The radius of the sphere

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the sphere
    """

    H, W = res[0], res[1]
    image = np.zeros((H, W))

    x = np.arange(W * pxSize / 2, -W * pxSize / 2, -pxSize)
    y = np.arange(H * pxSize / 2, -H * pxSize / 2, -pxSize)[:-1]
    xs, ys = np.meshgrid(x, y)

    # Calculate valid sphere area for z
    valid = xs**2 + ys**2 <= rad**2
    zs = np.sqrt((rad**2 - xs**2 - ys**2) * valid)
    # Surface normals on sphere
    n = np.stack((xs, ys, zs), axis=-1)
    n /= np.linalg.norm(n)
    # n-dot-l lighting
    image = np.dot(n, light.reshape(-1, 1)).reshape(H, W) * valid
    plt.imshow(np.flip(image, axis=1), cmap='gray')
    plt.axis('off')
    plt.show()

    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    I = None
    L = None
    s = None

    # Load images to I
    for i in range(1, 8):
        img = imread(path + 'input_' + str(i) + '.tif', dtype=np.uint16)
        img_xyz = rgb2xyz(img)
        img_luminance = img_xyz[:, :, 1]
        if I is None:
            s = img_luminance.shape
            I = img_luminance.flatten().reshape(1, -1)
        else:
            I = np.vstack((I, img_luminance.flatten().reshape(1, -1)))

    # Load light directions to L
    L = np.load('../data/sources.npy').T
    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    B = np.linalg.inv(L.dot(L.T)).dot(L).dot(I)
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = np.linalg.norm(B, axis=0)
    normals = B / albedos
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `gray` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    H, W = s
    albedoIm = albedos.reshape(s)
    normalIm = normals.reshape(3, H, W).transpose(1, 2, 0)

    plt.imshow(albedoIm, cmap='gray')
    plt.title('Albedos')
    plt.show()
    plt.imshow(normalize(normalIm), cmap='rainbow')
    plt.title('Normals')
    plt.colorbar()
    plt.show()

    return albedoIm, normalIm


def normalize(normalIm):
    normalIm_norm = normalIm - np.amin(normalIm)
    normalIm_norm /= np.amax(normalIm_norm)
    return normalIm_norm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    H, W = s
    normals = normals.reshape(3, H, W).transpose(1, 2, 0)
    nx = normals[:, :, 0]
    ny = normals[:, :, 1]
    nz = normals[:, :, 2]
    zx = nx / nz
    zy = ny / nz
    surface = integrateFrankot(zx, zy)
    return surface


def plotSurface(surface):

    """
    Question 1 (i)

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    H, W = surface.shape
    X = np.arange(0, W)
    Y = np.arange(0, H)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, surface, cmap=cm.coolwarm)
    plt.show()


if __name__ == '__main__':
    # Question 1 (b)
    center = np.zeros(3)
    rad = 0.75
    light1 = np.ones(3) / np.sqrt(3)
    light2 = np.array([1, -1, 1]) / np.sqrt(3)
    light3 = np.array([-1, -1, 1]) / np.sqrt(3)
    pxSize = 7e-4
    res = np.array([3840, 2160])

    renderNDotLSphere(center, rad, light1, pxSize, res)
    renderNDotLSphere(center, rad, light2, pxSize, res)
    renderNDotLSphere(center, rad, light3, pxSize, res)

    # Question 1 (c)
    I, L, s = loadData()

    # Question 1 (d)
    S = scipy.linalg.svdvals(I)
    print("Singular values for I:")
    print(S)

    # Question 1 (e)
    B = estimatePseudonormalsCalibrated(I, L)
    albedos, normals = estimateAlbedosNormals(B)

    # Question 1 (f)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    # Question 1 (i)
    surface = estimateShape(normals, s)
    plotSurface(surface)
