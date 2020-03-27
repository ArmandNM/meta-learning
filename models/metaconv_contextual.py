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


class MetaConvContextual(torch.nn.Module):
    def __init__(self, in_size=84, in_channels=3, out_channels=5, hidden_size=32, num_context_params=100):
        super().__init__()

        self.hidden_size = hidden_size
        self.context_params = torch.nn.Parameter(torch.zeros(size=[num_context_params], requires_grad=True))

        # TODO: maybe rename hidden_size to num_filters
        self.block1 = ConvBlock(in_channels, hidden_size)
        self.block2 = ConvBlock(hidden_size, hidden_size)
        self.block3 = ConvBlock(hidden_size, hidden_size)
        self.block4 = ConvBlock(hidden_size, hidden_size)

        self.film_fc = torch.nn.Linear(num_context_params, 2 * hidden_size)

        self.fc = torch.nn.Linear(int(in_size / 16) ** 2 * hidden_size, out_channels)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        film_params = self.film_fc(self.context_params)
        gamma = film_params[:self.hidden_size].view(1, -1, 1, 1)
        beta = film_params[self.hidden_size:].view(1, -1, 1, 1)

        x = gamma * x + beta

        x = self.block4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def freeze_shared_params(self):
        for p in self.parameters():
            p.requires_grad = False
        self.context_params.requires_grad = True

    def unfreeze_all_params(self):
        for p in self.parameters():
            p.requires_grad = True

    def reset_context_params(self):
        self.context_params.data = self.context_params.data.detach() * 0
        self.context_params.requires_grad = True


if __name__ == '__main__':
    model = MetaConvContextual()
    print(list(model.named_parameters()))
    for p in model.parameters():
        p.requires_grad = False
    print(list(model.named_parameters()))
    print(model.context_params)
