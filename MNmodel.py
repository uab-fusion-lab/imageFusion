import torch.nn as nn
import torch.nn.functional as F


class OneLayerCNN(nn.Module):
    def __init__(self):
        super(OneLayerCNN, self).__init__()

        # Single convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Fully connected layer
        self.fc1 = nn.Linear(32 * 14 * 14, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Dimension: 14x14
        x = x.view(-1, 32 * 14 * 14)  # Flatten
        x = self.fc1(x)
        return x