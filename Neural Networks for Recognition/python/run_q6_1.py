from os.path import join

import numpy as np
from PIL import Image
import scipy.io
import skimage.transform

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')


class MLP(nn.Module):
    def __init__(self, hidden_size, input_size=1024, num_classes=36):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.Sigmoid(),
                    nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        return self.net(x)


class Conv_NIST(nn.Module):
    def __init__(self, in_channel=1, num_classes=36):
        super(Conv_NIST, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(in_channel, 16, 3, 1, 1),
                    nn.BatchNorm2d(16),
                    nn.ReLU6(inplace=True),  # 32x32x16
                    nn.Conv2d(16, 32, 3, 1, 1),
                    nn.BatchNorm2d(32),
                    nn.ReLU6(inplace=True))  # 32x32x32
        self.fc = nn.Linear(32*32*32, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 32*32*32)
        out = self.fc(x)
        return out


class Conv_CIFAR10(nn.Module):
    def __init__(self, in_channel=3, num_classes=10):
        super(Conv_CIFAR10, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(in_channel, 32, 3, 1, 1),
                    nn.ReLU6(inplace=True),
                    nn.MaxPool2d(2, 2),  # 16x16x32
                    nn.Conv2d(32, 64, 3, 1, 1),
                    nn.ReLU6(inplace=True),
                    nn.MaxPool2d(2, 2),  # 8x8x64
                    nn.Conv2d(64, 128, 3, 1, 1),
                    nn.ReLU6(inplace=True),
                    nn.MaxPool2d(2, 2))  # 4x4x128
        self.fc = nn.Sequential(
                    nn.Linear(4*4*128, 512),
                    nn.ReLU(),
                    nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 4*4*128)
        out = self.fc(x)
        return out


class Conv_SUN(nn.Module):
    def __init__(self, in_channel=3, num_classes=8):
        super(Conv_SUN, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(in_channel, 16, 3, 1, 1),
                    nn.ReLU6(inplace=True),
                    nn.MaxPool2d(2, 2),  # 16x16x16
                    nn.Conv2d(16, 32, 3, 1, 1),
                    nn.ReLU6(inplace=True),
                    nn.MaxPool2d(2, 2))  # 8x8x32
        self.fc = nn.Sequential(
                    nn.Linear(8*8*32, 128),
                    nn.ReLU(),
                    nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 8*8*32)
        out = self.fc(x)
        return out


def train(model, train_loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    total, correct = 0.0, 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)

        scores = model(x)
        _, predict = torch.max(scores.data, 1)
        _, label = torch.max(y.data, 1)
        total += label.size(0)
        correct += (label == predict).sum().item()

        loss = criterion(scores, label)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
    running_loss /= len(train_loader)
    accuracy = correct / total

    return running_loss, accuracy


def train_cifar10(model, train_loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    total, correct = 0.0, 0.0
    for data in train_loader:
        optimizer.zero_grad()
        x, y = data
        x = x.to(device)
        y = y.to(device)

        scores = model(x)
        _, predict = torch.max(scores.data, 1)
        total += y.size(0)
        correct += (y == predict).sum().item()

        loss = criterion(scores, y)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
    running_loss /= len(train_loader)
    accuracy = correct / total

    return running_loss, accuracy


def train_sun(model, train_loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    total, correct = 0.0, 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)

        scores = model(x)
        _, predict = torch.max(scores.data, 1)
        total += y.size(0)
        correct += (y == predict).sum().item()

        loss = criterion(scores, y)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
    running_loss /= len(train_loader)
    accuracy = correct / total

    return running_loss, accuracy


def evaluation(model, val_loader, criterion):
    model.eval()

    running_loss = 0.0
    total, correct = 0.0, 0.0
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        scores = model(x)
        _, predict = torch.max(scores.data, 1)
        _, label = torch.max(y.data, 1)
        total += y.size(0)
        correct += (label == predict).sum().item()

        loss = criterion(scores, label).detach()
        running_loss += loss.item()

    running_loss /= len(val_loader)
    accuracy = correct / total
    return running_loss, accuracy


def evaluation_sun(model, val_loader, criterion):
    model.eval()

    running_loss = 0.0
    total, correct = 0.0, 0.0
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        scores = model(x)
        _, predict = torch.max(scores.data, 1)
        total += y.size(0)
        correct += (y == predict).sum().item()

        loss = criterion(scores, y).detach()
        running_loss += loss.item()

    running_loss /= len(val_loader)
    accuracy = correct / total
    return running_loss, accuracy


if __name__ == '__main__':
    train_data = scipy.io.loadmat('../data/nist36_train.mat')
    valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

    # Q6.1.1
    print('-'*80, '\nQ6.1.1\n')
    train_x = torch.from_numpy(train_data['train_data']).type(torch.float32)
    train_y = torch.from_numpy(train_data['train_labels']).type(torch.float32)
    valid_x = torch.from_numpy(valid_data['valid_data']).type(torch.float32)
    valid_y = torch.from_numpy(valid_data['valid_labels']).type(torch.float32)

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    valid_dataset = torch.utils.data.TensorDataset(valid_x, valid_y)

    batch_size = 64
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=batch_size)

    C, D = train_x.shape[1], train_y.shape[1]
    epochs = 50
    hidden_size = 64
    learning_rate = 1e-2

    ''' Build the MLP model '''
    model = MLP(hidden_size=hidden_size, input_size=C, num_classes=D)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    model.to(device)

    ''' Train MLP '''
    train_losses, train_accs = [], []
    for e in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_loss, val_acc = evaluation(model, valid_loader, criterion)
        print("Epoch", e + 1, ", Train Loss:", train_loss, ", Train Accuracy:", train_acc, ", Validation Loss:", val_loss, ", Validation Accuracy:", val_acc)
    np.savez('MLP_NIST_results.npz', train_losses=train_losses, train_accs=train_accs)

    plt.plot(range(epochs), train_accs, 'b')
    plt.title('Training Accuracy on NIST36')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(range(epochs), train_losses, 'b')
    plt.title('Training Loss on NIST36')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


    # Q6.1.2
    print('\n\n')
    print('-'*80, '\nQ6.1.2\n')
    train_x = torch.from_numpy(train_data['train_data'].reshape(-1, 1, 32, 32)).type(torch.float32)
    train_y = torch.from_numpy(train_data['train_labels']).type(torch.float32)
    valid_x = torch.from_numpy(valid_data['valid_data'].reshape(-1, 1, 32, 32)).type(torch.float32)
    valid_y = torch.from_numpy(valid_data['valid_labels']).type(torch.float32)

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    valid_dataset = torch.utils.data.TensorDataset(valid_x, valid_y)

    batch_size = 64
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=batch_size)

    C, D = train_x.shape[1], train_y.shape[1]
    epochs = 50
    hidden_size = 64
    learning_rate = 1e-2

    ''' Build the Conv_NIST model '''
    model = Conv_NIST()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    model.to(device)

    ''' Train Conv_NIST '''
    train_losses, train_accs = [], []
    for e in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_loss, val_acc = evaluation(model, valid_loader, criterion)
        print("Epoch", e + 1, ", Train Loss:", train_loss, ", Train Accuracy:", train_acc, ", Validation Loss:", val_loss, ", Validation Accuracy:", val_acc)
    np.savez('Conv_NIST_results.npz', train_losses=train_losses, train_accs=train_accs)

    plt.plot(range(epochs), train_accs, 'b')
    plt.title('Training Accuracy on NIST36')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(range(epochs), train_losses, 'b')
    plt.title('Training Loss on NIST36')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


    # Q6.1.3
    print('\n\n')
    print('-'*80, '\nQ6.1.3\n')
    batch_size = 64
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    epochs = 20
    hidden_size = 64
    learning_rate = 1e-3

    ''' Build the Conv_CIFAR10 model '''
    model = Conv_CIFAR10()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    ''' Train Conv_CIFAR10 '''
    train_losses, train_accs = [], []
    for e in range(epochs):
        train_loss, train_acc = train_cifar10(model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print("Epoch", e + 1, ", Train Loss:", train_loss, ", Train Accuracy:", train_acc)
    np.savez('Conv_CIFAR10_results.npz', train_losses=train_losses, train_accs=train_accs)

    plt.plot(range(epochs), train_accs, 'b')
    plt.title('Training Accuracy on CIFAR10')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(range(epochs), train_losses, 'b')
    plt.title('Training Loss on CIFAR10')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


    # Q6.1.4
    print('\n\n')
    print('-'*80, '\nQ6.1.4\n')
    train_x, test_x = [], []
    with open('../data/train_files.txt', 'r') as f:
        for line in f:
            img_path = join('../data/', line.rstrip('\n'))
            img = Image.open(img_path)
            img = np.array(img).astype(np.float32) / 255
            if len(img.shape) < 3:
                img = np.stack((img, img, img), axis=-1)
            img_resized = skimage.transform.resize(img, (32, 32))
            train_x.append(img_resized.transpose(2, 0, 1))
    with open('../data/test_files.txt', 'r') as f:
        for line in f:
            img_path = join('../data/', line.rstrip('\n'))
            img = Image.open(img_path)
            img = np.array(img).astype(np.float32) / 255
            if len(img.shape) < 3:
                img = np.stack((img, img, img), axis=-1)
            img_resized = skimage.transform.resize(img, (32, 32))
            test_x.append(img_resized.transpose(2, 0, 1))
    train_x = torch.FloatTensor(train_x)
    test_x = torch.FloatTensor(test_x)

    train_y, test_y = None, None
    with open('../data/train_labels.txt', 'r') as f:
        labels = f.read().splitlines()
        labels = [int(x) for x in labels]
        train_y = torch.LongTensor(labels)
    with open('../data/test_labels.txt', 'r') as f:
        labels = f.read().splitlines()
        labels = [int(x) for x in labels]
        test_y = torch.LongTensor(labels)

    batch_size = 64
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    epochs = 50
    hidden_size = 64
    learning_rate = 1e-3

    ''' Build the Conv_SUN model '''
    model = Conv_SUN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    ''' Train Conv_SUN '''
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    for e in range(epochs):
        train_loss, train_acc = train_sun(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluation_sun(model, test_loader, criterion)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        print("Epoch", e + 1, ", Train Loss:", train_loss, ", Train Accuracy:", train_acc, ", Test Loss:", test_loss, ", Test Accuracy:", test_acc)
    np.savez('Conv_SUN_results.npz', train_losses=train_losses, train_accs=train_accs)

    plt.plot(range(epochs), train_accs, 'b', label='Train')
    plt.plot(range(epochs), test_accs, 'g', label='Test')
    plt.title('Train and Test Accuracy on SUN')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(range(epochs), train_losses, 'b', label='Train')
    plt.plot(range(epochs), test_losses, 'g', label='Test')
    plt.title('Train and Test Loss on SUN')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
