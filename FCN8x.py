'''

Semantic Segmentation Preparation--- Fully Convolutional Networks

Author: JazzCharles
link: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

'''


import torch as t
import torch.nn as nn
import torch.functional as F
import torchvision.models as models


#alexnet = models.alexnet(pretrained=True)
#print(alexnet.features)
# size
# 224, 27, 13, 13, 6


class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)
        )

        self.score1 = nn.Conv2d(128, num_classes, 1)
        self.score2 = nn.Conv2d(256, num_classes, 1)
        self.score3 = nn.Conv2d(512, num_classes, 1)

        #8x restore size
        self.conv_trans1 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, padding=4)
        #2x
        self.conv_trans2 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1)
        #2x
        self.conv_trans3 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1)

    def forward(self, x):
        print('Input ', x.shape)
        x = self.features[:9](x)
        # downsize = 1/8
        stage1 = x
        print('Downsampled_Stage1  ', stage1.shape)

        x = self.features[9:12](x)
        # 1/16
        stage2 = x
        print('Downsampled_Stage2 ', stage2.shape)

        x = self.features[12:15](x)
        # 1/32
        stage3 = x
        print('Downsampled_Stage3 ', stage3.shape)


        stage3 = self.conv_trans3(self.score3(stage3))
        print('Upsampled_Stage3 ', stage3.shape)

        stage2 = self.conv_trans2(self.score2(stage2) + stage3)
        print('Upsampled_Stage2 ', stage2.shape)

        stage1 = self.conv_trans1(self.score1(stage1) + stage2)
        print('Upsampled_Stage1 ', stage1.shape)

        return stage1


fcn = FCN(num_classes=10)
print(fcn)

ipt = t.randn(1, 3, 64, 64)
out = fcn(ipt)
print(out.shape)

