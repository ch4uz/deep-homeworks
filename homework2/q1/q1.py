# -*- coding: utf-8 -*-


#https://github.com/MedMNIST/MedMNIST


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms

from medmnist import BloodMNIST, INFO

import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from q1_network import Q1Net

BASE_DIR="homework2/q1/"

# Training results:
#
# No softmax, no max pooling:
# Loss: 0.0000 | Val Acc: 0.9252 | Test Acc: 0.9249 | Time: 8.73 sec
# Total training time: 147.14 minutes (8828.69 seconds)
# Final Test acc: 0.9249
#
# With softmax:
# Loss: 2.1367 | Val Acc: 0.1373 | Test Acc: 0.1374 | Time: 9.11 sec
# Total training time: 30.51 minutes (1830.88 seconds)
# Final Test acc: 0.1374
#
# With max pooling: 7.74 minutes
# Loss: 0.0000 | Val Acc: 0.9474 | Test Acc: 0.9456 | Time: 2.47 sec
# Total training time: 7.74 minutes (464.31 seconds)
# Final Test acc: 0.9456
#
# With softmax and max pooling:
# Loss: 1.3619 | Val Acc: 0.8721 | Test Acc: 0.8661 | Time: 2.37 sec
# Total training time: 12.71 minutes (762.58 seconds)
# Final Test acc: 0.8661
#

with_maxpool=True
with_softmax=True
batch_size = 64
learning_rate = 0.001
model = Q1Net(with_softmax, with_maxpool)
epochs = 200
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device found")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA device found")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU")
print(f"Using device: {device}")
model = model.to(device)

# Data Loading

data_flag = 'bloodmnist'
print(data_flag)
info = INFO[data_flag]
print(len(info['label']))
n_classes = len(info['label'])

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

import time

# --------- Before Training ----------
total_start = time.time()

#Training Function

def train_epoch(loader, model, criterion, optimizer):
    model.train()
    total_loss = 0
    for _, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = criterion(pred, y.squeeze())

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    return total_loss / len(loader)

#Evaluation Function

def evaluate(loader, model):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.squeeze().long()

            outputs = model(imgs)
            preds += outputs.argmax(dim=1).cpu().tolist()
            targets += labels.tolist()

    return accuracy_score(targets, preds)


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig(f"{BASE_DIR}/charts/{name}.pdf", bbox_inches='tight')

train_dataset = BloodMNIST(split='train', transform=transform, download=True, size=28)
val_dataset   = BloodMNIST(split='val',   transform=transform, download=True, size=28)
test_dataset  = BloodMNIST(split='test',  transform=transform, download=True, size=28)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_losses = []
val_accs = []
test_accs = []

for epoch in range(epochs):

    epoch_start = time.time()

    train_loss = train_epoch(train_loader, model, criterion, optimizer)
    val_acc = evaluate(val_loader, model)
    test_acc = evaluate(test_loader, model)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

    train_losses.append(train_loss)
    val_accs.append(val_acc)
    test_accs.append(test_acc)

    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | "
          f"Time: {epoch_time:.2f} sec")



# --------- After Training ----------
total_end = time.time()
total_time = total_end - total_start

print(f"\nTotal training time: {total_time/60:.2f} minutes "
      f"({total_time:.2f} seconds)")

print('Final Test acc: %.4f' % (evaluate(model=model, loader=test_loader)))

config_parts = [f"{epochs}-epochs"]
if with_softmax:
    config_parts.append("with-softmax")
if with_maxpool:
    config_parts.append("with-maxpool")
config = "_".join(config_parts)

#Save the model
checkpoint_path = f"{BASE_DIR}/checkpoints/bloodmnist-cnn_{config}.pth"
torch.save(model.state_dict(), checkpoint_path)
print("Model saved as q1/bloodmnist_cnn.pth")

plot(range(1, epochs+1), train_losses, ylabel='Loss', name='train-loss_{}'.format(config))
plot(range(1, epochs+1), val_accs, ylabel='Accuracy', name='val-accuracy_{}'.format(config))
plot(range(1, epochs+1), test_accs, ylabel='Accuracy', name='test-accuracy_{}'.format(config))
