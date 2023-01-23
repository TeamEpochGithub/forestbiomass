import torch
import torch.nn as nn
from models.swin.efficientnet_swin import Efficient_Swin
from res_swin_v2 import Res_Swin

class Conv_3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding):
        super(Conv_3, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv3(x)

class Ensemble_Model(nn.Module):
    def __init__(self, path_efficient, path_resnet):
        super().__init__()
        self.model1 = Efficient_Swin()
        checkpoint1 = torch.load(str(path_efficient), map_location=torch.device('cpu'))
        self.model1.load_state_dict(checkpoint1["state_dict"])
        self.model2 = Res_Swin()
        checkpoint2 = torch.load(str(path_resnet), map_location=torch.device('cpu'))
        self.model2.load_state_dict(checkpoint2["state_dict"])
        self.conv1 = Conv_3(2, 1, 1, 1, 0)
        self.ff = nn.Conv2d(1, 1, 1, 1, 0)

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        xx = torch.cat((x1, x2), axis=1)
        xx = self.conv1(xx)
        return self.ff(xx)