# coding: utf-8
import numpy as np
import os

from cdnet.vgg16 import encoder
from cdnet.ops import decoder

import torch
import torch.nn as nn

class CDNet(nn.Module):
    def __init__(self):
        super(CDNet,self).__init__()

        self.encoder0 = encoder()
        for name,param in self.encoder0.named_parameters():
            param.requires_grad = False

        self.encoder1 = encoder()
        for name,param in self.encoder1.named_parameters():
            param.requires_grad = False

        self.decoder  = decoder()

    def forward(self,x0,x1):

        f0_1,f0_2,f0_3,f0_4 = self.encoder0(x0)
        f1_1,f1_2,f1_3,f1_4 = self.encoder1(x1)

        p, c = self.decoder(f0_1,f0_2,f0_3,f0_4,
                            f1_1,f1_2,f1_3,f1_4)

        return p,c


if __name__ == "__main__":

    nseed = 42
    torch.manual_seed(nseed)
    torch.cuda.manual_seed(nseed)
    np.random.seed(nseed)

    m = CDNet()
    print(m)
    for name,param in m.named_parameters():
        print(name, param.size())
