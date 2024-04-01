import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self):
        super(ResNetBlock, self).__init__()
        self.bottleneck =nn.Sequential(
            nn.Conv2d(256,64,1,padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,256,1,padding="same")
        )

        def forward(self,x):
            residual=self.bottleneck(x)
            outputs=x+residual
            return outputs

