#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import time
import pickle
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

import utils.utils as utils


class LogisticRegressor:
    def __init__(self, n_classes, n_features):
        self.W = np.zeros((n_classes, n_features))
        self.learning_rate = 0.0001
        self.l2_penalty = 0.00001

    def save(self, path):
        """
        Save to the provided path
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """
        Load from the provided path
        """
        with open(path, "rb") as f:
            return pickle.load(f)


    def train_epoch(self, X, y):
        """
        X (n_examples, n_features): features for the whole dataset
        y (n_examples,): labels for the whole dataset
        """
        for i in range(X.shape[0]):
            logits = self.W @ X[i]
            logits_max = np.max(logits)
            exp_logits = np.exp(logits - logits_max)
            P = exp_logits / np.sum(exp_logits)

            ey = np.zeros(self.W.shape[0])
            ey[y[i]] = 1.0
            gradient = np.outer(P - ey, X[i]) + self.l2_penalty * self.W

            self.W = self.W - self.learning_rate * gradient


    def predict(self, X):
        """
        X (n_examples, n_features)
        returns predicted labels y_hat, whose shape is (n_examples,)
        """
        scores = self.W @ X.T
        return np.argmax(scores, axis=0)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels

        returns classifier accuracy
        """
        y_hat = self.predict(X)
        num_correct = 0
        for i in range(X.shape[0]):
            pred_class = y_hat[i]
            true_class = y[i]
            if pred_class == true_class:
                num_correct += 1

        accuracy = num_correct/X.shape[0]
        return accuracy


def feature_extractor(X):
    """
    X: (n_examples, 785) - flattened 28x28 images + bias term
    Returns: (n_examples, 1226) - original pixels + HOG features + bias

    Applies Histogram of Oriented Gradients (HOG) feature extraction:
    - Computes gradient magnitude and orientation at each pixel
    - Divides 28x28 image into 7x7 grid of 4x4 pixel cells
    - Creates 9-bin histogram of gradient orientations per cell (0-180 degrees)
    - Results in 7x7x9 = 441 HOG features per image

    JUSTIFICATION FOR HOG FEATURES:
    Research shows that HOG features significantly improve handwritten character
    recognition accuracy compared to raw pixels, especially for linear classifiers:

    1. Local gradient feature descriptors significantly outperform raw pixel intensities.
       When combined with classifiers like SVM, they achieve very high accuracies on
       handwritten datasets (Thai, Latin, Bangla). [1]

    2. Gradient features from gray-scale images yield the best performance in feature
       extraction studies for handwritten digit recognition. [2]

    3. Histogram of Oriented Gradients (HOG) descriptors are invariant to geometric
       transformation and are among the best descriptors for character recognition.
       Experiments show using HOG to extract features improves recognition rates. [3]

    4. Switching to gradient-based features improves system performance significantly
       compared to using raw pixels. Linear SVMs achieve competitive performance when
       using carefully designed gradient features. [4]

    5. CNNs' initial layers naturally learn to extract edge features, demonstrating
       their importance. By manually extracting HOG features, we give the linear
       model access to orientation patterns that make deep learning successful. [5]

    HOG specifically captures stroke orientation, which is critical for distinguishing
    letters like 'I' (vertical), 'T' (horizontal+vertical), 'O' (circular), etc.

    Sources:
    [1] https://www.sciencedirect.com/science/article/abs/pii/S0952197615001724
    [2] https://www.sciencedirect.com/science/article/abs/pii/S0031320303002243
    [3] https://www.researchgate.net/publication/269329013
    [4] https://www2.eecs.berkeley.edu/Pubs/TechRpts/2009/EECS-2009-159.pdf
    [5] https://pmc.ncbi.nlm.nih.gov/articles/PMC7349603/
    """
    n_examples = X.shape[0]

    # Separate the bias term (last column)
    X_pixels = X[:, :-1]  # (n_examples, 784)
    bias = X[:, -1:]      # (n_examples, 1)

    # Reshape to (n_examples, 28, 28)
    X_reshaped = X_pixels.reshape(n_examples, 28, 28)

    # Compute gradients using central differences
    gx = np.zeros_like(X_reshaped)
    gy = np.zeros_like(X_reshaped)
    gx[:, :, 1:-1] = (X_reshaped[:, :, 2:] - X_reshaped[:, :, :-2]) / 2.0
    gy[:, 1:-1, :] = (X_reshaped[:, 2:, :] - X_reshaped[:, :-2, :]) / 2.0

    # Compute gradient magnitude and orientation
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * 180 / np.pi  # Convert to degrees
    orientation = (orientation + 180) % 180  # Map to [0, 180)

    # HOG parameters
    cell_size = 4  # 4x4 pixels per cell (28/4 = 7 cells per dimension)
    n_bins = 9     # 9 orientation bins (0-180 degrees, 20 degrees per bin)
    n_cells = 28 // cell_size  # 7 cells per dimension

    # Extract HOG features for all examples
    hog_features = []

    for img_idx in range(n_examples):
        cell_histograms = []

        for i in range(n_cells):
            for j in range(n_cells):
                # Extract cell region
                cell_mag = magnitude[img_idx,
                                   i*cell_size:(i+1)*cell_size,
                                   j*cell_size:(j+1)*cell_size]
                cell_orient = orientation[img_idx,
                                        i*cell_size:(i+1)*cell_size,
                                        j*cell_size:(j+1)*cell_size]

                # Compute histogram of oriented gradients for this cell
                hist = np.zeros(n_bins)
                bin_width = 180.0 / n_bins  # 20 degrees per bin

                for y in range(cell_size):
                    for x in range(cell_size):
                        angle = cell_orient[y, x]
                        mag = cell_mag[y, x]
                        bin_idx = int(angle / bin_width) % n_bins
                        hist[bin_idx] += mag

                cell_histograms.append(hist)

        # Flatten all cell histograms for this image
        # 7x7 cells x 9 bins = 441 features
        hog_features.append(np.concatenate(cell_histograms))

    hog_features = np.array(hog_features)

    # Concatenate: original pixels + HOG features + bias
    # Total: 784 + 441 + 1 = 1226 features
    return np.concatenate([X_pixels, hog_features, bias], axis=1)

def main(args):
    utils.configure_seed(seed=args.seed)

    data = utils.load_dataset(data_path=args.data_path, bias=True)
    X_train_raw, y_train = data["train"]
    X_valid_raw, y_valid = data["dev"]
    X_test_raw, y_test = data["test"]

    X_train = feature_extractor(X_train_raw)
    X_valid = feature_extractor(X_valid_raw)
    X_test = feature_extractor(X_test_raw)

    n_classes = np.unique(y_train).size
    n_feats = X_train.shape[1]

    # initialize the model
    model = LogisticRegressor(n_classes, n_feats)

    epochs = np.arange(1, args.epochs + 1)

    valid_accs = []
    train_accs = []

    start = time.time()

    best_valid = 0.0
    best_epoch = -1
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(X_train.shape[0])
        X_train = X_train[train_order]
        y_train = y_train[train_order]

        model.train_epoch(X_train, y_train)

        train_acc = model.evaluate(X_train, y_train)
        valid_acc = model.evaluate(X_valid, y_valid)

        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        print('train acc: {:.4f} | val acc: {:.4f}'.format(train_acc, valid_acc))

        if valid_acc > best_valid:
            best_valid = valid_acc
            best_epoch = i
            model.save(args.save_path)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))

    print("Reloading best checkpoint")
    best_model = LogisticRegressor.load(args.save_path)
    test_acc = best_model.evaluate(X_test, y_test)

    print('Best model test acc: {:.4f}'.format(test_acc))

    utils.plot(
        "Epoch", "Accuracy",
        {"train": (epochs, train_accs), "valid": (epochs, valid_accs)},
        filename=args.accuracy_plot
    )

    with open(args.scores, "w") as f:
        json.dump(
            {"best_valid": float(best_valid),
             "selected_epoch": int(best_epoch),
             "test": float(test_acc),
             "time": elapsed_time},
            f,
            indent=4
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int,
                        help="""Number of epochs to train for.""")
    parser.add_argument('--data-path', type=str, default="data/emnist-letters.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default="q1/checkpoints/checkpoint-lr-b.pickle")
    parser.add_argument("--accuracy-plot", default="q1/plots/Q1-lr-accs-b.pdf")
    parser.add_argument("--scores", default="q1/scores/Q1-lr-scores-b.json")
    args = parser.parse_args()
    main(args)
