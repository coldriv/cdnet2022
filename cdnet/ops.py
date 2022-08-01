# coding: utf-8
import torch
import torch.nn as nn

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()

        self.h3_dconv = nn.Sequential(
            nn.ConvTranspose2d(512,256,kernel_size=(5,5),stride=(2,2),padding=(2,2),output_padding=(1,1),dilation=(1,1),bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.h3_conv = nn.Sequential(
            nn.Conv2d(512,256, kernel_size=(1,1),stride=(1,1),padding=(0,0),dilation=(1,1),bias=True),
            nn.ReLU(),
        )

        self.h2_dconv = nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=(5,5),stride=(2,2),padding=(2,2),output_padding=(1,1),dilation=(1,1),bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.h2_conv = nn.Sequential(
            nn.Conv2d(256,128, kernel_size=(1,1),stride=(1,1),padding=(0,0),dilation=(1,1),bias=True),
            nn.ReLU(),
        )

        self.tile_conv = nn.Sequential(
            nn.Conv2d(128,512, kernel_size=(5,5),stride=(1,1),padding=(2,2),dilation=(1,1),bias=True),
            nn.ReLU(),

            nn.Conv2d(512,512, kernel_size=(5,5),stride=(1,1),padding=(2,2),dilation=(1,1),bias=True),
            nn.ReLU(),

            nn.Conv2d(512,128, kernel_size=(1,1),stride=(1,1),padding=(0,0),dilation=(1,1),bias=True),
            nn.ReLU(),
        )

        self.h1_dconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64,kernel_size=(5,5),stride=(2,2),padding=(2,2),output_padding=(1,1),dilation=(1,1),bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.h0y = nn.Sequential(
            nn.ConvTranspose2d( 64,  2,kernel_size=(5,5),stride=(1,1),padding=(2,2),output_padding=(0,0),dilation=(1,1),bias=True),
            nn.Softmax(dim=1),
        )
        self.h0c = nn.Sequential(
            nn.ConvTranspose2d( 64,  1,kernel_size=(5,5),stride=(1,1),padding=(2,2),output_padding=(0,0),dilation=(1,1),bias=True),
            nn.Sigmoid(),
        )


    def forward(self,
                h0_1,h0_2,h0_3,h0_4,
                h1_1,h1_2,h1_3,h1_4):

        h1  = torch.abs(h0_1 - h1_1) # (None, 64,256,320)
        h2  = torch.abs(h0_2 - h1_2) # (None,128,128,160)
        h3  = torch.abs(h0_3 - h1_3) # (None,256, 64, 80)
        h4  = torch.abs(h0_4 - h1_4) # (None,512, 32, 40)

        h3_ = self.h3_dconv(h4)
        h3d = torch.cat([h3,h3_],dim=1) # (None,256, 64, 80)
        h3d = self.h3_conv(h3d)

        h2_ = self.h2_dconv(h3d)
        h2d = torch.cat([h2,h2_],dim=1) # (None,128,128,160)
        h2d = self.h2_conv(h2d)

        h2d = self.tile_conv(h2d)

        h1d = self.h1_dconv(h2d) # (None, 64,256,320)

        y = self.h0y(h1d)
        c = self.h0c(h1d)

        # print("h1     :",   h1.size())  #
        # print("h2     :",   h2.size())  #
        # print("h3     :",   h3.size())  #
        # print("h4     :",   h4.size())  #
        # print("h3d    :",  h3d.size())  #
        # print("h2d    :",  h2d.size())  #
        # print("h1d    :",  h1d.size())  #
        # print("y      :",    y.size())  # 
        # print("c      :",    c.size())  # 

        return y,c

def ce_with_c(logits, labels, confidence):
    flatten = nn.flatten()
    logits     = flatten(logits)
    labels     = flatten(labels)
    confidence = flatten(confidence)
    losst      = labels * torch.log(logits    +1e-8)
    lossc      =          torch.log(confidence+1e-8)
    losst      = -torch.mean(losst)
    lossc      = -torch.mean(lossc)
    return losst,lossc

if __name__ == "__main__":
    import numpy as np

    x = torch.ones(size=(1,3,256,320),dtype=torch.float32)

    from vgg16 import encoder
    enc0 = encoder()
    enc1 = encoder()
    dec0 = decoder()
    with torch.no_grad():
        f0_1,f0_2,f0_3,f0_4 = enc0(x)
        f1_1,f1_2,f1_3,f1_4 = enc1(x)
        y,c = dec0(f0_1,f0_2,f0_3,f0_4,
                   f1_1,f1_2,f1_3,f1_4)
    print(torch.mean(y[:,0,:,:]),torch.mean(y[:,1,:,:]))
    print(torch.mean(c))
    # np.savetxt(fname="pt_f01.csv",X=f0_1[0,:,:,5],delimiter=",")
    # np.savetxt(fname="pt_f02.csv",X=f0_2[0,:,:,5],delimiter=",")
    # np.savetxt(fname="pt_f03.csv",X=f0_3[0,:,:,5],delimiter=",")
    # np.savetxt(fname="pt_f04.csv",X=f0_4[0,:,:,5],delimiter=",")
    # np.savetxt(fname="pt_f11.csv",X=f1_1[0,:,:,5],delimiter=",")
    # np.savetxt(fname="pt_f12.csv",X=f1_2[0,:,:,5],delimiter=",")
    # np.savetxt(fname="pt_f13.csv",X=f1_3[0,:,:,5],delimiter=",")
    # np.savetxt(fname="pt_f14.csv",X=f1_4[0,:,:,5],delimiter=",")
