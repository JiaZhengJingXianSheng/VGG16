import torch.nn as nn
import torch
from typing import Union, List, Dict, Any, cast


class VggNet(nn.Module):
    def __init__(self, features: nn.Module, num_classes=1000, init_weights=True):
        super(VggNet, self).__init__()
        self.fetures = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._intialize_weights()

    def forward(self, x: torch.Tensor):
        x = self.fetures(x)
        x = self.avgpool(x)
        x = x.view(-1, 512 * 7 *7)
        x = self.classifier(x)
        return x

    def _intialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 会认为子类继承父类的属性来判断是否是一个已知类型

                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # 这是往列表里添加元素的操作
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'Vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'Vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'Vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'Vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(cfg, batch_norm=True, num_classes=1000, init_weights=True):
    model = VggNet(make_layers(cfgs[cfg], batch_norm=batch_norm), num_classes=num_classes, init_weights=init_weights)
    return model


def vgg16(batch_norm=True, num_classes=1000, init_weights=True):
    return _vgg('Vgg16', batch_norm=batch_norm, num_classes=num_classes, init_weights=init_weights)


def vgg19(batch_norm=True, num_classes=1000, init_weights=True):
    return _vgg('Vgg19', batch_norm=batch_norm, num_classes=num_classes, init_weights=init_weights)
