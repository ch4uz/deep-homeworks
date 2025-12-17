import os
import random

import numpy as np
import matplotlib.pyplot as plt
import torch


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_dataset(data_path, bias=False):
    """
    Load EMNIST data (provided as a compressed npz file)
    """

    data_dict = np.load(data_path)

    # casting is necessary because the data is stored as uint8 to reduce file size
    X_train = data_dict["X_train"].astype(float) / 256  # normalizes between 0 and 1
    X_valid = data_dict["X_valid"].astype(float) / 256
    X_test = data_dict["X_test"].astype(float) / 256

    y_train = data_dict["y_train"].astype(int)
    y_valid = data_dict["y_valid"].astype(int)
    y_test = data_dict["y_test"].astype(int)

    if np.min(y_train) == 1:
        # This is necessary in case the dataset uses 1-based indexing for class labels
        y_train -= 1
        y_valid -= 1
        y_test -= 1

    if bias:
        X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
        X_valid = np.hstack((X_valid, np.ones((X_valid.shape[0], 1))))
        X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

    return {
        "train": (X_train, y_train), "dev": (X_valid, y_valid), "test": (X_test, y_test),
    }


# curve_dict, key is label, value is (x, y)
def plot(x_label, y_label, curves, filename=None):
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for curve_label, (x, y) in curves.items():
        plt.plot(x, y, label=curve_label)

    plt.legend()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.clf()


class ClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        """
        data: the dict returned by utils.load_pneumonia_data
        """
        train_X, train_y = data["train"]
        dev_X, dev_y = data["dev"]
        test_X, test_y = data["test"]

        self.X = torch.tensor(train_X, dtype=torch.float32)
        self.y = torch.tensor(train_y, dtype=torch.long)

        self.dev_X = torch.tensor(dev_X, dtype=torch.float32)
        self.dev_y = torch.tensor(dev_y, dtype=torch.long)

        self.test_X = torch.tensor(test_X, dtype=torch.float32)
        self.test_y = torch.tensor(test_y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
