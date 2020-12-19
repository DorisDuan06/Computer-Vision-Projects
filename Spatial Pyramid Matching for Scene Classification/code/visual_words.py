import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
import sklearn.cluster


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''

    filter_scales = opts.filter_scales
    # ----- TODO -----
    if len(img.shape) < 3:
        img = np.stack((img, img, img), axis=-1)
    img = skimage.color.rgb2lab(img)

    filter_responses = []
    for scale in filter_scales:
        for c in range(3):
            filter_responses.append(scipy.ndimage.gaussian_filter(img[:, :, c], scale))
        for c in range(3):
            filter_responses.append(scipy.ndimage.gaussian_laplace(img[:, :, c], scale))
        for c in range(3):
            filter_responses.append(scipy.ndimage.gaussian_filter(img[:, :, c], scale, order=(0,1)))
        for c in range(3):
            filter_responses.append(scipy.ndimage.gaussian_filter(img[:, :, c], scale, order=(1,0)))

    filter_responses = np.stack(filter_responses, axis=-1)
    return filter_responses


def compute_dictionary_one_image(i, img_path, opts):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255
    filter_responses = extract_filter_responses(opts, img)

    H, W, _ = filter_responses.shape
    alpha = opts.alpha
    pixels = np.random.choice(H * W, alpha, replace=False)
    filter_responses = filter_responses.reshape(-1, filter_responses.shape[2])
    chosen_responses = filter_responses[pixels, :]

    if not os.path.exists(opts.feat_dir):
        os.mkdir(opts.feat_dir)
    np.save(opts.feat_dir + '/' + str(i) + '.npy', chosen_responses)


def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # ----- TODO -----

    args = []
    for i, file in enumerate(train_files):
        args.append((i, join(data_dir, file), opts))
    p = multiprocessing.Pool(n_worker)
    p.starmap(compute_dictionary_one_image, args)

    filter_responses = []
    for i in range(len(train_files)):
        filter_responses.append(np.load(feat_dir + '/' + str(i) + '.npy'))
    filter_responses = np.vstack(filter_responses)
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_

    # example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dictionary)


def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''

    # ----- TODO -----
    filter_responses = extract_filter_responses(opts, img)  # (H,W,3F)
    filter_responses = np.reshape(filter_responses, (-1, filter_responses.shape[2]))
    dist_matrix = scipy.spatial.distance.cdist(filter_responses, dictionary)  # (HxW, K)
    wordmap = np.argmin(dist_matrix, axis=1)
    wordmap = np.reshape(wordmap, (img.shape[0], img.shape[1]))
    return wordmap
