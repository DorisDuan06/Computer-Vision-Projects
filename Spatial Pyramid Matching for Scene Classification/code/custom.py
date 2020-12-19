import multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import util
import visual_words
import visual_recog
from opts import get_opts


def get_feature_from_wordmap_SPM_new(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''

    K = opts.K
    L = opts.L
    # ----- TODO -----
    H, W = wordmap.shape
    hist_all = []
    finest = np.empty((2 ** (L-1), 2 ** (L-1)), dtype=object)
    H_step = H // (2 ** (L-1))
    W_step = W // (2 ** (L-1))
    weight = 1/2 if L > 2 else 2 ** (-L+1)
    for i in range(2 ** (L-1)):
        for j in range(2 ** (L-1)):
            finest[i, j], _ = np.histogram(wordmap[i*H_step:(i+1)*H_step, j*W_step:(j+1)*W_step],
                                           bins=list(range(K+1)))
    finest = finest / (H * W)
    hist_all.append(weight * np.concatenate(finest.flatten(), axis=0))

    for l in range(L-2, -1, -1):
        weight = 2 ** (-L+1) if l <= 1 else 2 ** (l-L)
        # finer -> coarser
        coarser = np.empty((2 ** l, 2 ** l), dtype=object)
        for i in range(0, 2 ** (l+1), 2):
            for j in range(0, 2 ** (l+1), 2):
                coarser[i//2, j//2] = np.sum(finest[i:i+2, j:j+2])
        finest = coarser
        hist_all.append(weight * np.concatenate(coarser.flatten(), axis=0))
    hist_all = np.concatenate(hist_all, axis=0)
    return hist_all


def get_image_feature_new(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^L-1)/3)
    '''

    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feature = get_feature_from_wordmap_SPM_new(opts, wordmap)
    return feature


def distance_to_set_new(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    sim = np.sum(np.abs(word_hist - histograms), axis=1)  # Matching Distance
    return sim


def evaluate_recognition_system_new(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'), allow_pickle=True)
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    args = []
    for file in test_files:
        img_path = join(data_dir, file)
        args.append((opts, img_path, dictionary))
    p = multiprocessing.Pool(n_worker)
    test_features = p.starmap(get_image_feature_new, args)

    predict_indices = []
    train_features = trained_system['features']
    train_labels = trained_system['labels']
    for feature in test_features:
        distance = distance_to_set_new(feature, train_features)
        predict_indices.append(np.argmin(distance))
    predict_labels = train_labels[predict_indices]

    conf = np.zeros((8, 8))
    for i in range(len(test_labels)):
        conf[test_labels[i], predict_labels[i]] += 1
    accuracy = np.trace(conf) / np.sum(conf)
    return conf, accuracy


def main():
    opts = get_opts()
    opts.filter_scales = [1, 2, 4, 8, 12]
    opts.K = 200
    opts.alpha = 300
    opts.L = 4

    n_cpu = util.get_num_CPU()
    visual_words.compute_dictionary(opts, n_worker=n_cpu)

    n_cpu = util.get_num_CPU()
    visual_recog.build_recognition_system_new(opts, n_worker=n_cpu)

    n_cpu = util.get_num_CPU()
    conf, accuracy = evaluate_recognition_system_new(opts, n_worker=n_cpu)

    print(conf)
    print(accuracy)
    np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')


if __name__ == '__main__':
    main()
