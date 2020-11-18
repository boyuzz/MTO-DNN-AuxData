import torch
from torchvision import models
import torch.nn as nn


class ResNet18(models.ResNet):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__(models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.conv1 = torch.nn.Conv2d(1, 64,
                                     kernel_size=7,
                                     stride=2,
                                     padding=3, bias=False)
        self.avgpool = nn.AvgPool2d(1, stride=1)

    def forward(self, x):
        return super(ResNet18, self).forward(x)


class ResNet34(models.ResNet):
    def __init__(self, num_classes):
        super(ResNet34, self).__init__(models.resnet.BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
        self.conv1 = torch.nn.Conv2d(1, 64,
                                     kernel_size=7,
                                     stride=2,
                                     padding=3, bias=False)
        self.avgpool = nn.AvgPool2d(1, stride=1)

    def forward(self, x):
        return super(ResNet34, self).forward(x)


class ResNet50(models.ResNet):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__(models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
        self.conv1 = torch.nn.Conv2d(1, 64,
                                     kernel_size=7,
                                     stride=2,
                                     padding=3, bias=False)
        self.avgpool = nn.AvgPool2d(1, stride=1)

    def forward(self, x):
        return super(ResNet50, self).forward(x)