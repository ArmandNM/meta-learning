import torch
import torch.nn.functional as F
import math

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
    def __init__(self, in_size=84, in_channels=3, out_channels=5, hidden_size=32, k_spt=1):
        super().__init__()

        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.k_spt = k_spt

        self.support_params = None

        self.fc_x_query = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_spt_key = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_spt_value = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        self.fc_update = torch.nn.Linear(2 * hidden_size, 2 * hidden_size, bias=True)

        # TODO: maybe rename hidden_size to num_filters
        self.block1 = ConvBlock(in_channels, hidden_size)
        self.block2 = ConvBlock(hidden_size, hidden_size)
        self.block3 = ConvBlock(hidden_size, hidden_size)
        self.block4 = ConvBlock(hidden_size, hidden_size)

        # self.film_fc = torch.nn.Linear(hidden_size, 2 * hidden_size)

        self.fc = torch.nn.Linear(int(in_size / 16) ** 2 * hidden_size, out_channels)

    def forward(self, x, intermediate=False):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        if intermediate:
            return x  # shape: [n_ways, k_spt, 10, 10]

        proto_x = x.mean(axis=3).mean(axis=2)
        proto_spt = self.support_params

        # Self-attention
        query = self.fc_x_query(proto_x).unsqueeze(dim=1)
        key = self.fc_spt_key(proto_spt).unsqueeze(dim=0)
        value = self.fc_spt_value(proto_spt).unsqueeze(dim=0)

        key_t = torch.transpose(key, dim0=1, dim1=2)
        correlation = torch.matmul(query, key_t) / math.sqrt(self.hidden_size)
        correlation = F.softmax(correlation, dim=-1)
        aggregated_messages = torch.matmul(correlation, value)[:, 0, :]

        # Compute film params based on current x and context
        film_params = self.fc_update(torch.cat([proto_x, aggregated_messages], dim=-1))

        # film_params = self.film_fc(self.support_params)
        gamma = film_params[:, :self.hidden_size].unsqueeze(dim=2).unsqueeze(dim=3)
        beta = film_params[:, self.hidden_size:].unsqueeze(-1).unsqueeze(-1)

        x = gamma * x + beta

        x = self.block4(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

    def set_support_params(self, support_set):
        # TODO: write shapes in comments
        intermediate_features = self.forward(support_set, intermediate=True)
        intermediate_features = intermediate_features
        gap_features = F.avg_pool2d(intermediate_features.detach(), intermediate_features.size()[-1])
        gap_features_by_class = gap_features.view(self.out_channels, self.k_spt, self.hidden_size)
        class_embeddings = gap_features_by_class.mean(axis=1)
        self.support_params = class_embeddings
        # TODO: check requires_grad
        self.support_params.requires_grad = False

