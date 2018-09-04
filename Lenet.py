'''
The lenet network
Arxiv: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

'''

import torch as t
import numpy as np
import torch.nn as nn
import torch.functional as F

class lenet5(nn.Module):
    def __init__(self, num_classes):
        super(lenet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=(5, 5)),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(5, 16, kernel_size=(5, 5)),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        print(x.shape)
        x = self.classifier(x)
        return x


net = lenet5(10)
ipt = t.randn(1, 3, 32, 32)
out = net(ipt)
print(net)
print(ipt)
print(out)
