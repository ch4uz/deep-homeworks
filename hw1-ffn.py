#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt

import time
import utils


class FeedforwardNetwork(nn.Module):
    def __init__(
            self, t, n_features, hidden_size, layers,
            activation_type, dropout, **kwargs):
        """ Define a vanilla multiple-layer FFN with `layers` hidden layers 
        Args:
            n_classes (int)
            n_features (int)
            hidden_size (int)
            layers (int)
            activation_type (str)
            dropout (float): dropout probability
        """
        super().__init__()
        
        raise NotImplementedError()

    def forward(self, x, **kwargs):
        """ Compute a forward pass through the FFN
        Args:
            x (torch.Tensor): a batch of examples (batch_size x n_features)
        Returns:
            scores (torch.Tensor)
        """
        raise NotImplementedError()
    
    
def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """ Do an update rule with the given minibatch
    Args:
        X (torch.Tensor): (n_examples x n_features)
        y (torch.Tensor): gold labels (n_examples)
        model (nn.Module): a PyTorch defined model
        optimizer: optimizer used in gradient step
        criterion: loss function
    Returns:
        loss (float)
    """
    raise NotImplementedError()


def predict(model, X):
    """ Predict the labels for the given input
    Args:
        model (nn.Module): a PyTorch defined model
        X (torch.Tensor): (n_examples x n_features)
    Returns:
        preds: (n_examples)
    """
    raise NotImplementedError()


@torch.no_grad()
def evaluate(model, X, y, criterion):
    """ Compute the loss and the accuracy for the given input
    Args:
        model (nn.Module): a PyTorch defined model
        X (torch.Tensor): (n_examples x n_features)
        y (torch.Tensor): gold labels (n_examples)
        criterion: loss function
    Returns:
        loss, accuracy (Tuple[float, float])
    """
    raise NotImplementedError()


def plot(epochs, plottables, filename=None, ylim=None):
    """Plot the plottables over the epochs.
    
    Plottables is a dictionary mapping labels to lists of values.
    """
    plt.clf()
    plt.xlabel('Epoch')
    for label, plottable in plottables.items():
        plt.plot(epochs, plottable, label=label)
    plt.legend()
    if ylim:
        plt.ylim(ylim)
    if filename:
        plt.savefig(filename, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=30, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=64, type=int,
                        help="Size of training batch.")
    parser.add_argument('-hidden_size', type=int, default=32)
    parser.add_argument('-layers', type=int, default=1)
    parser.add_argument('-learning_rate', type=float, default=0.001)
    parser.add_argument('-l2_decay', type=float, default=0.0)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-activation',
                        choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-data_path', type=str, default='emnist-letters.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_dataset(opt.data_path)
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
    train_X, train_y = dataset.train_X, dataset.train_y
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    n_classes = torch.unique(dataset.y).shape[0]  # 26
    n_feats = dataset.X.shape[1]

    print(f"N features: {n_feats}")
    print(f"N classes: {n_classes}")

    # initialize the model
    model = FeedforwardNetwork(
        n_classes,
        n_feats,
        opt.hidden_size,
        opt.layers,
        opt.activation,
        opt.dropout
    )

    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )

    # get a loss criterion
    criterion = nn.CrossEntropyLoss()

    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    start = time.time()

    model.eval()
    initial_train_loss, initial_train_acc = evaluate(model, train_X, train_y, criterion)
    initial_val_loss, initial_val_acc = evaluate(model, dev_X, dev_y, criterion)
    train_losses.append(initial_train_loss)
    train_accs.append(initial_train_acc)
    valid_losses.append(initial_val_loss)
    valid_accs.append(initial_val_acc)
    print('initial val acc: {:.4f}'.format(initial_val_acc))

    for ii in epochs:
        print('Training epoch {}'.format(ii))
        epoch_train_losses = []
        model.train()
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            epoch_train_losses.append(loss)

        model.eval()
        epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
        _, train_acc = evaluate(model, train_X, train_y, criterion)
        val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

        print('train loss: {:.4f} | val loss: {:.4f} | val acc: {:.4f}'.format(
            epoch_train_loss, val_loss, val_acc
        ))

        train_losses.append(epoch_train_loss)
        train_accs.append(train_acc)
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))

    _, test_acc = evaluate(model, test_X, test_y, criterion)
    print('Final test acc: {:.4f}'.format(test_acc))

    # plot
    config = (
        f"batch-{opt.batch_size}-lr-{opt.learning_rate}-epochs-{opt.epochs}-"
        f"hidden-{opt.hidden_size}-dropout-{opt.dropout}-l2-{opt.l2_decay}-"
        f"layers-{opt.layers}-act-{opt.activation}-opt-{opt.optimizer}"
    )

    losses = {
        "Train Loss": train_losses,
        "Valid Loss": valid_losses,
    }

    plot(epochs, losses, filename=f'{opt.model}-training-loss-{config}.pdf')
    print(f"Final Training Accuracy: {train_accs[-1]:.4f}")
    print(f"Best Validation Accuracy: {max(valid_accs):.4f}")
    val_accuracy = { "Valid Accuracy": valid_accs }
    plot(epochs, val_accuracy, filename=f'{opt.model}-validation-accuracy-{config}.pdf')


if __name__ == '__main__':
    main()
