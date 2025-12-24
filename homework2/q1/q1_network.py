from torch import nn

class Q1Net(nn.Module):
    def __init__(self, with_softmax=False, with_maxpool=False):
        super().__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2) if with_maxpool else nn.Identity(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2) if with_maxpool else nn.Identity(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2) if with_maxpool else nn.Identity(),
        )

        # Calculate flattened size based on pooling
        if with_maxpool:
            # 28 → 14 → 7 → 3 (rounded down from 3.5)
            flattened_size = 128 * 3 * 3  # 1,152
        else:
            flattened_size = 128 * 28 * 28  # 100,352

        # Fully connected layers
        fc_layers = [
            nn.Flatten(),
            nn.Linear(in_features=flattened_size, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=8)
        ]

        if with_softmax:
            fc_layers.append(nn.Softmax(dim=1))

        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        logits = self.fc_layers(x)
        return logits
