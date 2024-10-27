import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, num_classes=10, use_batch_norm=True):
        super(LeNet5, self).__init__()
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        self.feature = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        return self.classifier(x)


class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet9, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            self.resnet_module(3, 64),
            self.resnet_module(64, 128),
        )
        self.conv2 = nn.Sequential(
            self.resnet_module(128, 128),
            self.resnet_module(128, 128),
        )
        self.conv3 = nn.Sequential(
            self.resnet_module(128, 256),
            self.resnet_module(256, 512),
        )
        self.conv4 = nn.Sequential(
            self.resnet_module(512, 512),
            self.resnet_module(512, 512),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.num_classes),
        )

    @staticmethod
    def resnet_module(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x) + x
        x = self.conv4(x) + x
        x = self.classifier(x)
        return x
