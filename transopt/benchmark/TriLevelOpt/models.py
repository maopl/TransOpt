import torch
import torch.nn as nn
import torch.nn.functional as F


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=True))

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class WideResNet(nn.Module):
    """WideResNet with the softmax layer removed"""
    def __init__(self, input_shape, num_classes, model_size, dropout_rate=0.0):
        super(WideResNet, self).__init__()
        
        # Define configurations for different model sizes
        configs = {
            28: (28, 10),  # WRN-28-10
            16: (16, 8),   # WRN-16-8
            40: (40, 2),   # WRN-40-2
            22: (22, 2)    # WRN-22-2
        }
        
        if model_size not in configs:
            raise ValueError(f"Unsupported model size: {model_size}. Choose from {list(configs.keys())}")
        
        self.depth, self.widen_factor = configs[model_size]
        self.nChannels = [16, 16*self.widen_factor, 32*self.widen_factor, 64*self.widen_factor]
        self.in_planes = 16

        assert ((self.depth-4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (self.depth-4) // 6
        self.n_outputs = self.nChannels[3]
        self.dropout = dropout_rate

        self.conv1 = nn.Conv2d(input_shape[0], self.nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(wide_basic, self.nChannels[1], n, self.dropout, stride=1)
        self.layer2 = self._wide_layer(wide_basic, self.nChannels[2], n, self.dropout, stride=2)
        self.layer3 = self._wide_layer(wide_basic, self.nChannels[3], n, self.dropout, stride=2)
        self.bn1 = nn.BatchNorm2d(self.nChannels[3], momentum=0.9)
        
        # Add classification layers
        self.linear = nn.Linear(self.nChannels[3], num_classes)
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def _wide_layer(self, block, planes, num_blocks, dropout, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class Hyperparameters(nn.Module):
    def __init__(self):
        super(Hyperparameters, self).__init__()

        self.lr = nn.Parameter(torch.randn(1))
        self.weight_decay = nn.Parameter(torch.randn(1))
        self.momentum = nn.Parameter(torch.randn(1))
        self.batch_size = nn.Parameter(torch.randn(1))
        self.dropout_rate = nn.Parameter(torch.randn(1)) 
        
        with torch.no_grad():
            # initialize to smaller value
            self.lr.mul_(1e-3)
            self.weight_decay.mul_(1e-4)
            self.momentum.mul_(1e-1)
            self.batch_size.mul_(1e-1) # Initialize to 0-1 range
            self.dropout_rate.mul_(1e-1)

    def forward(self):
        return [
            self.lr,
            self.weight_decay,
            self.momentum,
            torch.tensor([32, 64, 128, 196])[torch.argmax(F.softmax(self.batch_size))],
            self.dropout_rate 
        ]
