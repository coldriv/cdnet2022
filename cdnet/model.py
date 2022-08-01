# coding: utf-8
import numpy as np
import os

# from ops import ce_with_c
from cdnet.cdnet import CDNet

import torch
import torch.nn as nn
# import torch.optim as optim

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print("Device: ", self.device)

        self.model   = CDNet()

    def save(self,path="./weights/model.pth",check=True):
        if check:
            print("saveing ", path)
        torch.save(self.model.cpu().state_dict(), path)
        if check:
            print("saved")
        return

    def load(self,path="./weights/model.pth",check=True):
        if check:
            print("loading ", path)
        self.model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        if check:
            print("loaded")
        return

    def prediction(self,x0,x1,device=""):
        if ( device=="" ): device = self.device
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            x0 = torch.from_numpy(x0).clone().to(device)
            x1 = torch.from_numpy(x1).clone().to(device)
            p, c = self.model(x0,x1)
        return p.cpu().detach().numpy().copy(), c.cpu().detach().numpy().copy()


if __name__ == "__main__":

    m = net().to("cpu")
    m.load(path="./weights/model.pth")

    x0 = np.ones(shape=(1,3,512,640),dtype=np.float32)
    x1 = np.ones(shape=(1,3,512,640),dtype=np.float32)
    y,c = m.prediction(x0,x1,device="cpu")
    print(c)
    print(y.shape)
    print(np.mean(y[:,0,:,:]),np.mean(y[:,1,:,:]))
    print(c.shape)
    print(np.mean(c))
