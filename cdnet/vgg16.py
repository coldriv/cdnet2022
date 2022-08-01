# coding: utf-8
import os
import numpy as np
import torch
import torch.nn as nn

vgg_mean = torch.tensor([103.939, 116.779, 123.68],dtype=torch.float32)
vgg16_npy_path = "vgg16npy"

def conv_layer(bottom, name):
    filt = get_conv_filter(vgg16_npy_path+os.sep+name+"_W_1.npy")
    conv_bias = get_bias(vgg16_npy_path+os.sep+name+"_b_1.npy")

    with torch.no_grad():
        bottom.weight.copy_(filt)
        bottom.bias.copy_(conv_bias)

    return bottom

def get_conv_filter(path):
    return torch.from_numpy(np.load(path).transpose([3, 2, 0, 1]))

def get_bias(path):
    return torch.from_numpy(np.load(path))

class encoder(nn.Module):
    def __init__(self, nc=3):
        super(encoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d( nc, 64,kernel_size=(3,3),stride=(1,1),padding=(1,1),dilation=(1,1),bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d( 64, 64,kernel_size=(3,3),stride=(1,1),padding=(1,1),dilation=(1,1),bias=True),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2),padding=(0,0),dilation=(1,1))

        self.conv2 = nn.Sequential(
            nn.Conv2d( 64,128,kernel_size=(3,3),stride=(1,1),padding=(1,1),dilation=(1,1),bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=(3,3),stride=(1,1),padding=(1,1),dilation=(1,1),bias=True),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2),padding=(0,0),dilation=(1,1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=(3,3),stride=(1,1),padding=(1,1),dilation=(1,1),bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=(3,3),stride=(1,1),padding=(1,1),dilation=(1,1),bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=(3,3),stride=(1,1),padding=(1,1),dilation=(1,1),bias=True),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2),padding=(0,0),dilation=(1,1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(256,512,kernel_size=(3,3),stride=(1,1),padding=(1,1),dilation=(1,1),bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=(1,1),dilation=(1,1),bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=(1,1),dilation=(1,1),bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self,rgb):
        rgb_scaled = rgb * 255.0
        blue, green, red  = torch.chunk(input=rgb_scaled, chunks=3, dim=1)
        bgr = torch.cat( [blue  - vgg_mean[0],
                          green - vgg_mean[1],
                          red   - vgg_mean[2]], dim=1)

        conv1_2 = self.conv1(bgr)
        bgr = self.pool1(conv1_2)

        conv2_2 = self.conv2(bgr)
        bgr = self.pool2(conv2_2)

        conv3_3 = self.conv3(bgr)
        bgr = self.pool3(conv3_3)

        conv4_3 = self.conv4(bgr)

        return conv1_2,conv2_2,conv3_3,conv4_3

def readimg(path, h=512):
    import cv2
    img = cv2.imread(path)
    img = img.astype(np.float32)/255.0
    img = img[np.newaxis]
    return img

if __name__ == "__main__":

    m = encoder()

    rgb = readimg("./_tmp/202004281727.jpg").transpose([0,3,1,2])
    print(rgb.shape)
    rgb = torch.from_numpy(rgb).clone()
    with torch.no_grad():
        f1,f2,f3,f4 = m(rgb)
    f4 = f4.detach().numpy().transpose([0,2,3,1])
    print(np.mean(f4))
