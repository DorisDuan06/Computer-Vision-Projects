import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt


def compute_loss(x, x_hat):
    return np.sum(np.square(x - x_hat))


train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36
learning_rate = 3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x, np.ones((train_x.shape[0], 1)), batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
N, C = train_x.shape
initialize_weights(C, hidden_size, params, 'layer1')
initialize_weights(hidden_size, hidden_size, params, 'layer2')
initialize_weights(hidden_size, hidden_size, params, 'layer3')
initialize_weights(hidden_size, C, params, 'output')

names = list(params)
for name in names:
    params["m_" + name] = np.zeros_like(params[name])

# should look like your previous training loops
train_loss = []
for itr in range(max_iters):
    total_loss = 0
    for xb, _ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        ''' Train '''
        # forward
        h1 = forward(xb, params, 'layer1', relu)
        h2 = forward(h1, params, 'layer2', relu)
        h3 = forward(h2, params, 'layer3', relu)
        x_hat = forward(h3, params, 'output', sigmoid)
        # loss
        loss = compute_loss(xb, x_hat)
        total_loss += loss
        # backward
        delta = -2 * (xb - x_hat)
        delta = backwards(delta, params, 'output')
        delta = backwards(delta, params, 'layer3', relu_deriv)
        delta = backwards(delta, params, 'layer2', relu_deriv)
        delta = backwards(delta, params, 'layer1', relu_deriv)
        # apply gradient
        for name in names:
            # Update momentum params
            params['m_' + name] = 0.9 * params['m_' + name] - learning_rate * params['grad_' + name]
            # Update non-momentum params
            params[name] += params['m_' + name]

    total_loss /= N
    train_loss.append(total_loss)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

plt.plot(range(max_iters), train_loss, 'b')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
N_val = valid_x.shape[0]
selected = [(421, 430), (704, 766), (1503, 1505), (3119, 3153), (3506, 3579)]
for i, j in selected:
    fig, axes = plt.subplots(1, 4)
    img1 = valid_x[i].reshape(32, 32).T
    axes[0].imshow(img1, cmap='gray')

    img2 = valid_x[j].reshape(32, 32).T
    axes[2].imshow(img2, cmap='gray')

    h1 = forward(valid_x[i], params, 'layer1', relu)
    h2 = forward(h1, params, 'layer2', relu)
    h3 = forward(h2, params, 'layer3', relu)
    img_hat1 = forward(h3, params, 'output', sigmoid)
    axes[1].imshow(img_hat1.reshape(32, 32).T, cmap='gray')

    h1 = forward(valid_x[j], params, 'layer1', relu)
    h2 = forward(h1, params, 'layer2', relu)
    h3 = forward(h2, params, 'layer3', relu)
    img_hat2 = forward(h3, params, 'output', sigmoid)
    axes[3].imshow(img_hat2.reshape(32, 32).T, cmap='gray')
    plt.show()

# Q5.3.2
from skimage.measure import compare_psnr as psnr
# evaluate PSNR
h1 = forward(valid_x, params, 'layer1', relu)
h2 = forward(h1, params, 'layer2', relu)
h3 = forward(h2, params, 'layer3', relu)
img_hat = forward(h3, params, 'output', sigmoid)

PSNR = 0
for i in range(N_val):
    PSNR += psnr(valid_x[i], img_hat[i])

print("Average PSNR on validation set:", PSNR / N_val)
