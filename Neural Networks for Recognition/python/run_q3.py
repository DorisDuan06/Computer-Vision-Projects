import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 64
learning_rate = 5e-3
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
N_train, D = train_x.shape
N_val = valid_x.shape[0]
C = train_y.shape[1]
initialize_weights(D, hidden_size, params, 'layer1')
initialize_weights(hidden_size, C, params, 'output')
# visualize weights here
W = params['Wlayer1']

fig = plt.figure()
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.1)
for i in range(hidden_size):
    im = grid[i].imshow(W[:, i].reshape(32, 32), cmap='gray')
plt.axis('off')
plt.show()

# with default settings, you should get loss < 150 and accuracy > 80%
train_loss, train_accuracy = [], []
val_loss, val_accuracy = [], []
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb, yb in batches:
        ''' Train '''
        # forward
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        # loss
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc / batch_num
        # backward
        delta = probs - yb
        delta = backwards(delta, params, 'output', linear_deriv)
        delta = backwards(delta, params, 'layer1')
        # apply gradient
        params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
        params['blayer1'] -= learning_rate * params['grad_blayer1']
        params['Woutput'] -= learning_rate * params['grad_Woutput']
        params['boutput'] -= learning_rate * params['grad_boutput']

    ''' Validation '''
    # forward
    h1 = forward(valid_x, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)
    # loss
    loss, acc = compute_loss_and_acc(valid_y, probs)

    train_loss.append(total_loss / N_train)
    train_accuracy.append(total_acc)
    val_loss.append(loss / N_val)
    val_accuracy.append(acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, total_loss, total_acc))

# run on validation set and report accuracy! should be above 75%
h1 = forward(valid_x, params, 'layer1')
probs = forward(h1, params, 'output', softmax)
valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)

print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

plt.plot(range(max_iters), train_accuracy, 'b', label='Train')
plt.plot(range(max_iters), val_accuracy, 'g', label='Validation')
plt.title('Training and Validation Accuracies')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(range(max_iters), train_loss, 'b', label='Train')
plt.plot(range(max_iters), val_loss, 'g', label='Validation')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

''' Test '''
test_data = scipy.io.loadmat('../data/nist36_test.mat')
test_x, test_y = test_data['test_data'], test_data['test_labels']
h1 = forward(test_x, params, 'layer1')
probs = forward(h1, params, 'output', softmax)
test_loss, test_acc = compute_loss_and_acc(test_y, probs)
print('Test accuracy:', test_acc)

# Q3.3
# visualize weights here
W = params['Wlayer1']

fig = plt.figure()
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.1)
for i in range(hidden_size):
    im = grid[i].imshow(W[:, i].reshape(32, 32), cmap='gray')
plt.axis('off')
plt.show()

# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
N_test = test_x.shape[0]
h1_test = forward(test_x, params, 'layer1')
probs_test = forward(h1_test, params, 'output', softmax)
label = np.argmax(test_y, axis=1)
label_hat = np.argmax(probs_test, axis=1)
for i in range(N_test):
    confusion_matrix[label[i]][label_hat[i]] += 1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
