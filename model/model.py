import torch.nn as nn
import torch


class Cifar10(nn.Module):
    """
        CNNs to train CIFAR10 dataset
    """
    def __init__(self, dropout=0.0, n_classes=10):
        super(Cifar10, self).__init__()
        # input size should be (batch_size x 3 x 32 x 32)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),  # (batch_size x 16 x 32 x 32)
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),  # (batch_size x 32 x 32 x 32)
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size x 32 x 16 x 16)
            nn.BatchNorm2d(num_features=32),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # (batch_size x 64 x 16 x 16)
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),  # (batch_size x 64 x 16 x 16)
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size x 64 x 8 x 8)
            nn.BatchNorm2d(num_features=64),
        )
        self.flatten_size = 64 * 8 * 8

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=self.flatten_size, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_classes),
        )

        self.init_weights()

    def init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, self.flatten_size)
        return self.classifier(x)
