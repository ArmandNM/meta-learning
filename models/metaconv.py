import torch


def conv_block(in_channels, out_channels):
    block = torch.nn.Sequential(torch.nn.Conv2d(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=3,
                                                padding=1),
                                torch.nn.BatchNorm2d(num_features=out_channels,
                                                     momentum=1.0,
                                                     track_running_stats=False),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(2))
    return block


class MetaConv(torch.nn.Module):
    def __init__(self, in_size=84, in_channels=3, out_channels=5, hidden_size=32):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.net = torch.nn.Sequential(conv_block(in_channels, hidden_size),
                                       conv_block(hidden_size, hidden_size),
                                       conv_block(hidden_size, hidden_size),
                                       conv_block(hidden_size, hidden_size),
                                       Flatten(),
                                       torch.nn.Linear(int(in_size / 16) ** 2 * hidden_size, out_channels))

    def forward(self, x):
        return self.net(x)


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
