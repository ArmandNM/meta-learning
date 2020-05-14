import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes, track_running_stats=False)


class Block(nn.Module):

    def __init__(self, inplanes, planes, downsample, n_ways, k_spt, graphs):
        super().__init__()

        self.graphs = graphs
        self.n_ways = n_ways
        self.k_spt = k_spt
        self.planes = planes

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.cached_support_features1 = None
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.cached_support_features2 = None
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)
        self.cached_support_features3 = None

        self.downsample = downsample

        self.maxpool = nn.MaxPool2d(2)

        self.cached_support_features4 = None

    def forward(self, x, is_support=False):
        out = self.conv1(x)
        out = self.bn1(out)
        if is_support:
            self.cached_support_features1 = out.detach()
        if self.graphs[0] is not None:
            out = self.graphs[0](out, proto_spt=self.aggregate_features(self.cached_support_features1))
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if is_support:
            self.cached_support_features2 = out.detach()
        if self.graphs[1] is not None:
            out = self.graphs[1](out, proto_spt=self.aggregate_features(self.cached_support_features2))
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if is_support:
            self.cached_support_features3 = out.detach()
        if self.graphs[2] is not None:
            out = self.graphs[2](out, proto_spt=self.aggregate_features(self.cached_support_features3))

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        if is_support:
            self.cached_support_features4 = out.detach()
        if self.graphs[3] is not None:
            out = self.graphs[3](out, proto_spt=self.aggregate_features(self.cached_support_features4))

        return out

    def aggregate_features(self, intermediate_features):
        # Compute global average pooling
        intermediate_features = F.avg_pool2d(intermediate_features.detach(), intermediate_features.size()[-1])
        # Separate features by class changing shape to: [N, K, ...]
        # TODO: hidden size can be determined from intermediate_features shape, remove it from method parameters
        intermediate_features = intermediate_features.view(self.n_ways, self.k_spt, self.planes)
        # Average over all examples in the same class
        intermediate_features = intermediate_features.mean(axis=1)
        return intermediate_features.detach()


class AttentionModule(torch.nn.Module):
    def __init__(self, hidden_size, fc_x_query=None, fc_spt_key=None, fc_spt_value=None, fc_update=None,
                 gamma_scale_gate=None, gamma_bias_gate=None, beta_scale_gate=None):
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

        if gamma_scale_gate is not None:
            self.gamma_scale_gate = gamma_scale_gate
        else:
            self.gamma_scale_gate = torch.nn.Parameter(torch.zeros(size=[1, hidden_size, 1, 1, 1], requires_grad=True))

        if gamma_bias_gate is not None:
            self.gamma_bias_gate = gamma_bias_gate
        else:
            self.gamma_bias_gate = torch.nn.Parameter(torch.ones(size=[1, hidden_size, 1, 1, 1], requires_grad=True))

        if beta_scale_gate is not None:
            self.beta_scale_gate = beta_scale_gate
        else:
            self.beta_scale_gate = torch.nn.Parameter(torch.zeros(size=[1, hidden_size, 1, 1, 1], requires_grad=True))

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
                 fc_spt_spt_query=None, fc_spt_spt_key=None, fc_spt_spt_value=None,
                 gamma_scale_gate=None, gamma_bias_gate=None, beta_scale_gate=None):
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

        if gamma_scale_gate is not None:
            self.gamma_scale_gate = gamma_scale_gate
        else:
            self.gamma_scale_gate = torch.nn.Parameter(torch.zeros(size=[1, hidden_size, 1, 1, 1], requires_grad=True))

        if gamma_bias_gate is not None:
            self.gamma_bias_gate = gamma_bias_gate
        else:
            self.gamma_bias_gate = torch.nn.Parameter(torch.ones(size=[1, hidden_size, 1, 1, 1], requires_grad=True))

        if beta_scale_gate is not None:
            self.beta_scale_gate = beta_scale_gate
        else:
            self.beta_scale_gate = torch.nn.Parameter(torch.zeros(size=[1, hidden_size, 1, 1, 1], requires_grad=True))

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

        gamma = film_params[:, 0, :self.hidden_size].unsqueeze(dim=2).unsqueeze(dim=3).unsqueeze(dim=-1)
        beta = film_params[:, 0, self.hidden_size:].unsqueeze(-1).unsqueeze(-1).unsqueeze(dim=-1)

        # init gamma with ones and beta with zeros
        gamma = gamma * self.gamma_scale_gate + self.gamma_bias_gate
        beta  = beta * self.beta_scale_gate

        x = gamma * x.unsqueeze(dim=-1) + beta
        x = x.squeeze(dim=-1)

        return x


class ResNet12(nn.Module):

    def __init__(self, channels, n_ways, k_spt):
        super().__init__()

        self.channels = channels
        self.n_ways = n_ways
        self.k_spt = k_spt
        self.inplanes = 3

        self.attention1 = AttentionModuleV2(hidden_size=channels[0])

        self.attention2_1 = AttentionModuleV2(hidden_size=channels[1])
        self.attention2_2 = AttentionModuleV2(hidden_size=channels[1], fc_x_query=self.attention2_1.fc_x_query,
                                              fc_spt_key=self.attention2_1.fc_spt_key, fc_spt_value=self.attention2_1.fc_spt_value,
                                              fc_x_update=self.attention2_1.fc_x_update, fc_update=self.attention2_1.fc_update,
                                              fc_spt_spt_query=self.attention2_1.fc_spt_spt_query, fc_spt_spt_key=self.attention2_1.fc_spt_spt_key,
                                              fc_spt_spt_value=self.attention2_1.fc_spt_spt_value)
        self.attention2_3 = AttentionModuleV2(hidden_size=channels[1], fc_x_query=self.attention2_1.fc_x_query,
                                              fc_spt_key=self.attention2_1.fc_spt_key, fc_spt_value=self.attention2_1.fc_spt_value,
                                              fc_x_update=self.attention2_1.fc_x_update, fc_update=self.attention2_1.fc_update,
                                              fc_spt_spt_query=self.attention2_1.fc_spt_spt_query, fc_spt_spt_key=self.attention2_1.fc_spt_spt_key,
                                              fc_spt_spt_value=self.attention2_1.fc_spt_spt_value)

        self.attention3_1 = AttentionModuleV2(hidden_size=channels[2], fc_x_query=self.attention2_1.fc_x_query,
                                              fc_spt_key=self.attention2_1.fc_spt_key, fc_spt_value=self.attention2_1.fc_spt_value,
                                              fc_x_update=self.attention2_1.fc_x_update, fc_update=self.attention2_1.fc_update,
                                              fc_spt_spt_query=self.attention2_1.fc_spt_spt_query, fc_spt_spt_key=self.attention2_1.fc_spt_spt_key,
                                              fc_spt_spt_value=self.attention2_1.fc_spt_spt_value)
        self.attention3_2 = AttentionModuleV2(hidden_size=channels[2], fc_x_query=self.attention2_1.fc_x_query,
                                              fc_spt_key=self.attention2_1.fc_spt_key, fc_spt_value=self.attention2_1.fc_spt_value,
                                              fc_x_update=self.attention2_1.fc_x_update, fc_update=self.attention2_1.fc_update,
                                              fc_spt_spt_query=self.attention2_1.fc_spt_spt_query, fc_spt_spt_key=self.attention2_1.fc_spt_spt_key,
                                              fc_spt_spt_value=self.attention2_1.fc_spt_spt_value)
        self.attention3_3 = AttentionModuleV2(hidden_size=channels[2], fc_x_query=self.attention2_1.fc_x_query,
                                              fc_spt_key=self.attention2_1.fc_spt_key, fc_spt_value=self.attention2_1.fc_spt_value,
                                              fc_x_update=self.attention2_1.fc_x_update, fc_update=self.attention2_1.fc_update,
                                              fc_spt_spt_query=self.attention2_1.fc_spt_spt_query, fc_spt_spt_key=self.attention2_1.fc_spt_spt_key,
                                              fc_spt_spt_value=self.attention2_1.fc_spt_spt_value)
        
        self.attention4_1 = AttentionModuleV2(hidden_size=channels[3], fc_x_query=self.attention2_1.fc_x_query,
                                              fc_spt_key=self.attention2_1.fc_spt_key, fc_spt_value=self.attention2_1.fc_spt_value,
                                              fc_x_update=self.attention2_1.fc_x_update, fc_update=self.attention2_1.fc_update,
                                              fc_spt_spt_query=self.attention2_1.fc_spt_spt_query, fc_spt_spt_key=self.attention2_1.fc_spt_spt_key,
                                              fc_spt_spt_value=self.attention2_1.fc_spt_spt_value)
        self.attention4_2 = AttentionModuleV2(hidden_size=channels[3], fc_x_query=self.attention2_1.fc_x_query,
                                              fc_spt_key=self.attention2_1.fc_spt_key, fc_spt_value=self.attention2_1.fc_spt_value,
                                              fc_x_update=self.attention2_1.fc_x_update, fc_update=self.attention2_1.fc_update,
                                              fc_spt_spt_query=self.attention2_1.fc_spt_spt_query, fc_spt_spt_key=self.attention2_1.fc_spt_spt_key,
                                              fc_spt_spt_value=self.attention2_1.fc_spt_spt_value)
        self.attention4_3 = AttentionModuleV2(hidden_size=channels[3], fc_x_query=self.attention2_1.fc_x_query,
                                              fc_spt_key=self.attention2_1.fc_spt_key, fc_spt_value=self.attention2_1.fc_spt_value,
                                              fc_x_update=self.attention2_1.fc_x_update, fc_update=self.attention2_1.fc_update,
                                              fc_spt_spt_query=self.attention2_1.fc_spt_spt_query, fc_spt_spt_key=self.attention2_1.fc_spt_spt_key,
                                              fc_spt_spt_value=self.attention2_1.fc_spt_spt_value)

        self.layer1 = self._make_layer(channels[0], n_ways, k_spt, [None, None, None, None])
        self.layer2 = self._make_layer(channels[1], n_ways, k_spt, [self.attention2_1, self.attention2_2, self.attention2_3, None])
        self.layer3 = self._make_layer(channels[2], n_ways, k_spt, [self.attention3_1, self.attention3_2, self.attention3_3, None])
        self.layer4 = self._make_layer(channels[3], n_ways, k_spt, [self.attention4_1, self.attention4_2, self.attention4_3, None])

        self.out_dim = channels[3]

        self.fc = torch.nn.Linear(self.out_dim, self.n_ways)
        self.temp = torch.nn.Parameter(torch.tensor(10.0))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, n_ways, k_spt, graphs):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes),
        )
        block = Block(self.inplanes, planes, downsample, n_ways, k_spt, graphs)
        self.inplanes = planes
        return block

    def forward(self, x, is_support=False):
        x = self.layer1(x, is_support=is_support)
        x = self.layer2(x, is_support=is_support)
        x = self.layer3(x, is_support=is_support)
        x = self.layer4(x, is_support=is_support)
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        return x

    def predict(self, x_embeddings):
        return self.fc(x_embeddings)

# @register('resnet12')
def resnet12(n_ways, k_spt):
    return ResNet12(channels=[64, 128, 256, 512], n_ways=n_ways, k_spt=k_spt)

# @register('resnet12-narrow128')
def resnet12_narrow128(n_ways, k_spt):
    return ResNet12(channels=[64, 128, 128, 128], n_ways=n_ways, k_spt=k_spt)



# @register('resnet12-wide')
def resnet12_wide():
    return ResNet12([64, 160, 320, 640])
