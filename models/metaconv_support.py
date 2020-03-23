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


class MetaConvSupport(torch.nn.Module):
    def __init__(self, in_size=84, in_channels=3, out_channels=5, hidden_size=32):
        super().__init__()

        self.hidden_size = hidden_size
        self.support_params = torch.nn.Parameter(torch.zeros(size=[hidden_size], requires_grad=False))

        # TODO: maybe rename hidden_size to num_filters
        self.block1 = ConvBlock(in_channels, hidden_size)
        self.block2 = ConvBlock(hidden_size, hidden_size)
        self.block3 = ConvBlock(hidden_size, hidden_size)
        self.block4 = ConvBlock(hidden_size, hidden_size)

        self.film_fc = torch.nn.Linear(hidden_size, 2 * hidden_size)

        self.fc = torch.nn.Linear(int(in_size / 16) ** 2 * hidden_size, out_channels)

    def forward(self, x, intermediate=False):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        if intermediate:
            return x

        film_params = self.film_fc(self.support_params)
        gamma = film_params[:self.hidden_size].view(1, -1, 1, 1)
        beta = film_params[self.hidden_size:].view(1, -1, 1, 1)

        x = gamma * x + beta

        x = self.block4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def set_support_params(self, support_set):
        intermediate_features = self.forward(support_set, intermediate=True)
        gap_features = F.avg_pool2d(intermediate_features.detach(), intermediate_features.size()[-1])
        self.support_params.data = gap_features.sum(0).view(-1)
        self.support_params.requires_grad = False

