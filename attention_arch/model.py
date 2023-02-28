import torch.nn as nn
import torch.nn.functional as F
from utils import weights_init_normal
import torch
from custom_utils.restormer import Restormer

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator_S2F(nn.Module):
    def __init__(self,init_weights=False):
        super(Generator_S2F, self).__init__()

        self.restormer = Restormer(num_blocks=[2,2,2,2],num_heads=[1,2,4,8],channels=[12,24,48,96],num_refinement=1)
        
        if init_weights:
            self.apply(weights_init_normal)
    
    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_S2F(init_weights=True)
        return model


    def forward(self,xin):
        xout = self.restormer(xin)
        return xout.tanh()

class Generator_F2S(nn.Module):
    def __init__(self,init_weights=False):
        super(Generator_F2S, self).__init__()

        self.restormer = Restormer(in_channel=4,num_blocks=[2,2,2,2],num_heads=[1,2,4,8],channels=[12,24,48,96],num_refinement=1)
        
        if init_weights:
            self.apply(weights_init_normal)
    
    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_F2S(init_weights=True)
        return model

    def forward(self,xin,mask):
        x=torch.cat((xin,mask),1)
        xout=self.restormer(x)
        return xout.tanh()

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(3, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self,x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1).squeeze() #global avg pool