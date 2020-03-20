import torch
import torch.nn.functional as F


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.batchnorm2d = torch.nn.BatchNorm2d(num_features=out_channels, momentum=1.0, track_running_stats=False)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batchnorm2d(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        return x


class MetaConv(torch.nn.Module):
    def __init__(self, in_size=84, in_channels=3, out_channels=5, hidden_size=32):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.block1 = ConvBlock(in_channels, hidden_size)
        self.block2 = ConvBlock(hidden_size, hidden_size)
        self.block3 = ConvBlock(hidden_size, hidden_size)
        self.block4 = ConvBlock(hidden_size, hidden_size)

        self.fc = torch.nn.Linear(int(in_size / 16) ** 2 * hidden_size, out_channels)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
