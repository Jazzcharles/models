from torch import nn
import torch as t
import torchvision.models as models
from torch.nn import functional as F

res34 = models.resnet34()
#print(res34)


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        self.right = shortcut

    def forward(self, x):
        out_left = self.left(x)
        out_right = x
        if self.right:
            out_right = self.right(x)

        out_left += out_right
        return F.relu(out_left)


class ResNet34(nn.Module):
    def __init__(self, num_classes = 1000):
        super(ResNet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.layer1 = self.make_layer(64, 128, block_num=3)
        self.layer2 = self.make_layer(128, 256, block_num=4, stride=2)
        self.layer3 = self.make_layer(256, 512, block_num=6, stride=2)
        self.layer4 = self.make_layer(512, 512, block_num=3, stride=2)

        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_channel, out_channel, block_num=1, stride=1):

        shortcut = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        layers = list()
        layers.append(ResidualBlock(in_channel, out_channel, stride, shortcut))

        for i in range(block_num -1):
            layers.append(ResidualBlock(out_channel, out_channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


res34_2 = ResNet34()
print(res34_2)
ipt = t.randn(1, 3, 224, 224)
out = res34_2(ipt)
print(out.size())
