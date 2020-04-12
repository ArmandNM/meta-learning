import torch
import torch.nn.functional as F
import math


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.batchnorm2d = torch.nn.BatchNorm2d(num_features=out_channels, momentum=1.0, track_running_stats=False)
        self.cached_support_features = None

    def forward(self, x, is_support=False):
        x = self.conv2d(x)
        x = self.batchnorm2d(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        if is_support:
            self.cached_support_features = x.detach()

        return x


class AttentionModule(torch.nn.Module):
    def __init__(self, hidden_size, fc_x_query=None, fc_spt_key=None, fc_spt_value=None, fc_update=None):
        # TODO: Create different modules to separate attention from film aggregation
        super().__init__()
        self.hidden_size = hidden_size

        # Create new layers if None are given to reuse
        if fc_x_query is not None:
            self.fc_x_query = fc_x_query
        else:
            self.fc_x_query = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        if fc_spt_key is not None:
            self.fc_spt_key = fc_spt_key
        else:
            self.fc_spt_key = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        if fc_spt_value is not None:
            self.fc_spt_value = fc_spt_value
        else:
            self.fc_spt_value = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        if fc_update is not None:
            self.fc_update = fc_update
        else:
            self.fc_update = torch.nn.Linear(2 * hidden_size, 2 * hidden_size, bias=True)

    def forward(self, x, proto_spt):
        proto_x = x.mean(axis=3).mean(axis=2)

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
        return x


class AttentionModuleV2(torch.nn.Module):
    def __init__(self, hidden_size, fc_x_query=None, fc_spt_key=None, fc_spt_value=None, fc_x_update=None, fc_update=None,
                 fc_spt_spt_query=None, fc_spt_spt_key=None, fc_spt_spt_value=None):
        # TODO: Create different modules to separate attention from film aggregation
        super().__init__()
        self.hidden_size = hidden_size

        # Create new layers if None are given to reuse
        if fc_x_query is not None:
            self.fc_x_query = fc_x_query
        else:
            self.fc_x_query = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        if fc_spt_key is not None:
            self.fc_spt_key = fc_spt_key
        else:
            self.fc_spt_key = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        if fc_spt_value is not None:
            self.fc_spt_value = fc_spt_value
        else:
            self.fc_spt_value = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        if fc_x_update is not None:
            self.fc_x_update = fc_x_update
        else:
            self.fc_x_update = torch.nn.Linear(2 * hidden_size, hidden_size, bias=True)

        if fc_update is not None:
            self.fc_update = fc_update
        else:
            self.fc_update = torch.nn.Linear(2 * hidden_size, 2 * hidden_size, bias=True)

        if fc_spt_spt_query is not None:
            self.fc_spt_spt_query = fc_spt_spt_query
        else:
            self.fc_spt_spt_query = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        if fc_spt_spt_key is not None:
            self.fc_spt_spt_key = fc_spt_spt_key
        else:
            self.fc_spt_spt_key = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        if fc_spt_spt_value is not None:
            self.fc_spt_spt_value = fc_spt_spt_value
        else:
            self.fc_spt_spt_value = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, proto_spt):
        proto_x = x.mean(axis=3).mean(axis=2)

        # Reshape from [N*K, C] to [N*K, 1, C]
        proto_x = proto_x.unsqueeze(dim=1)

        # Reshape from [N, C] to [1, N, C]
        proto_spt = proto_spt.unsqueeze(dim=0)

        # Self-attention
        query = self.fc_x_query(proto_x)
        key = self.fc_spt_key(proto_spt)
        value = self.fc_spt_value(proto_spt)

        key_t = torch.transpose(key, dim0=1, dim1=2)
        correlation = torch.matmul(query, key_t) / math.sqrt(self.hidden_size)
        correlation = F.softmax(correlation, dim=-1)
        aggregated_messages = torch.matmul(correlation, value)

        # Compute updated proto_x based on current proto_x and context
        proto_x = self.fc_x_update(torch.cat([proto_x, aggregated_messages], dim=-1))

        # Send messages from proto_x to support
        # TODO: experiment with other update methods. Ex: proto_spt = W[proto_x|proto_spt] + proto_spt,
        proto_spt = proto_spt + proto_x

        # Send messages between support prototypes
        # Self-attention
        query = self.fc_spt_spt_query(proto_spt)
        key = self.fc_spt_spt_key(proto_spt)
        value = self.fc_spt_spt_value(proto_spt)

        key_t = torch.transpose(key, dim0=1, dim1=2)
        correlation = torch.matmul(query, key_t) / math.sqrt(self.hidden_size)
        correlation = F.softmax(correlation, dim=-1)
        proto_spt = torch.matmul(correlation, value)

        # Send messages from support to x
        # Self-attention
        query = self.fc_x_query(proto_x)
        key = self.fc_spt_key(proto_spt)
        value = self.fc_spt_value(proto_spt)

        key_t = torch.transpose(key, dim0=1, dim1=2)
        correlation = torch.matmul(query, key_t) / math.sqrt(self.hidden_size)
        correlation = F.softmax(correlation, dim=-1)
        aggregated_messages = torch.matmul(correlation, value)

        # Compute film params based on current x and context
        film_params = self.fc_update(torch.cat([proto_x, aggregated_messages], dim=-1))

        gamma = film_params[:, 0, :self.hidden_size].unsqueeze(dim=2).unsqueeze(dim=3)
        beta = film_params[:, 0, self.hidden_size:].unsqueeze(-1).unsqueeze(-1)

        x = gamma * x + beta

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

        self.fc_spt_spt_query = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_spt_spt_key = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_spt_spt_value = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        self.fc_x_update = torch.nn.Linear(2 * hidden_size, hidden_size, bias=True)
        self.fc_update = torch.nn.Linear(2 * hidden_size, 2 * hidden_size, bias=True)

        # TODO: maybe rename hidden_size to num_filters
        self.block1 = ConvBlock(in_channels, hidden_size)
        self.block2 = ConvBlock(hidden_size, hidden_size)
        self.block3 = ConvBlock(hidden_size, hidden_size)
        self.block4 = ConvBlock(hidden_size, hidden_size)

        # Create attention modules that share the same fully connected layers
        # self.attention1 = AttentionModule(hidden_size, self.fc_x_query, self.fc_spt_key, self.fc_spt_value, self.fc_update)
        # self.attention2 = AttentionModule(hidden_size, self.fc_x_query, self.fc_spt_key, self.fc_spt_value, self.fc_update)
        # self.attention3 = AttentionModule(hidden_size, self.fc_x_query, self.fc_spt_key, self.fc_spt_value, self.fc_update)
        # self.attention4 = AttentionModule(hidden_size, self.fc_x_query, self.fc_spt_key, self.fc_spt_value, self.fc_update)

        self.attention1 = AttentionModuleV2(hidden_size, self.fc_x_query, self.fc_spt_key, self.fc_spt_value, self.fc_x_update, self.fc_update, self.fc_spt_spt_query, self.fc_spt_spt_key, self.fc_spt_spt_value)
        self.attention2 = AttentionModuleV2(hidden_size, self.fc_x_query, self.fc_spt_key, self.fc_spt_value, self.fc_x_update, self.fc_update, self.fc_spt_spt_query, self.fc_spt_spt_key, self.fc_spt_spt_value)
        self.attention3 = AttentionModuleV2(hidden_size, self.fc_x_query, self.fc_spt_key, self.fc_spt_value, self.fc_x_update, self.fc_update, self.fc_spt_spt_query, self.fc_spt_spt_key, self.fc_spt_spt_value)
        self.attention4 = AttentionModuleV2(hidden_size, self.fc_x_query, self.fc_spt_key, self.fc_spt_value, self.fc_x_update, self.fc_update, self.fc_spt_spt_query, self.fc_spt_spt_key, self.fc_spt_spt_value)

        # self.film_fc = torch.nn.Linear(hidden_size, 2 * hidden_size)

        self.fc = torch.nn.Linear(int(in_size / 16) ** 2 * hidden_size, out_channels)
        self.temp = torch.nn.Parameter(torch.tensor(10.0))

    def forward(self, x, is_support=False):
        # TODO: give conv block access to n, k, hidden_size directly so we don't have to pass them
        x = self.block1.forward(x, is_support=is_support)
        x = self.block2.forward(x, is_support=is_support)

        x = self.block3.forward(x, is_support=is_support)
        # x = self.attention3.forward(x, proto_spt=self.aggregate_features(self.block3.cached_support_features))

        x = self.block4.forward(x, is_support=is_support)
        # x = self.attention4.forward(x, proto_spt=self.aggregate_features(self.block4.cached_support_features))

        x = x.reshape(x.size(0), -1)
        # x = self.fc(x)

        return x

    def aggregate_features(self, intermediate_features):
        # Compute global average pooling
        intermediate_features = F.avg_pool2d(intermediate_features.detach(), intermediate_features.size()[-1])
        # Separate features by class changing shape to: [N, K, ...]
        # TODO: hidden size can be determined from intermediate_features shape, remove it from method parameters
        intermediate_features = intermediate_features.view(self.out_channels, self.k_spt, self.hidden_size)
        # Average over all examples in the same class
        intermediate_features = intermediate_features.mean(axis=1)
        return intermediate_features.detach()
