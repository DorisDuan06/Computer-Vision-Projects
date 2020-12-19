import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def cluster(bboxes, K):
    from sklearn.cluster import KMeans
    bboxes = bboxes[np.argsort(bboxes[:, 2])]
    rows = bboxes[:, 2].reshape(-1, 1)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(rows)
    labels = kmeans.labels_

    lines, line = [], []
    prev_label = labels[0]
    for i in range(len(labels)):
        curr_label = labels[i]
        if curr_label == prev_label:
            line.append(bboxes[i])
        else:
            line = sorted(line, key=lambda x: x[1])
            lines.append(line)
            line = [bboxes[i]]
        prev_label = curr_label
    if len(line) > 0:
        line = sorted(line, key=lambda x: x[1])
        lines.append(line)
    return lines


true_K = [8, 5, 3, 3]  # ground truth lines of texts
characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
test_y = ["TODOLIST1MAKEATODOLIST2CHECKOFFTHEFIRSTTHINGONTODOLIST3REALIZEYOUHAVEALREADYCOMPLETED2THINGS4REWARDYOURSELFWITHANAP",
          "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
          "HAIKUSAREEASYBUTSOMETIMESTHEYDONTMAKESENSEREFRIGERATOR",
          "DEEPLEARNINGDEEPERLEARNINGDEEPESTLEARNING"]
for i, img in enumerate(os.listdir('../images')):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw, cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    lines = cluster(np.array(bboxes), true_K[i])

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    classified_texts = ""
    pad_width = 5
    acc, j = 0, 0
    for line in lines:
        for bbox in line:
            y1, x1, y2, x2 = bbox
            max_side = max(y2 - y1, x2 - x1)
            y_center, x_center = (y1 + y2) // 2, (x1 + x2) // 2
            image = bw[y_center - max_side // 2: y_center + max_side // 2, x_center - max_side // 2: x_center + max_side // 2]
            image_cropped = skimage.transform.resize(image.astype(float), (32 - 2 * pad_width, 32 - 2 * pad_width))
            image_padded = np.pad(image_cropped, ((pad_width, pad_width), (pad_width, pad_width)), 'constant', constant_values=1)
            image_padded = (image_padded > 0.9).astype(int)
            x = image_padded.transpose().reshape(1, -1)

            h1 = forward(x, params, 'layer1')
            probs = forward(h1, params, 'output', softmax)
            character = characters[np.argmax(probs)]
            acc += int(character == test_y[i][j])
            classified_texts += character
            j += 1
        classified_texts += '\n'
    print("-"*80 + "\n" + img + ":\n")
    print(classified_texts)
    print("Accuracy:", acc / len(test_y[i]), "\n\n")
