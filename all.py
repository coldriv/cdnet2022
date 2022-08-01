# coding: utf-8
import os, time, cv2
import numpy as np
from cdnet.model import net
from kym import run_yolov5, load_yolov5
from akh import Depth

def list2mask(list_detect,img):
    height, width, ch = img.shape
    img = np.ones_like(img)
    for j in range(len(list_detect)):
        npd = np.array(list_detect[j],dtype=np.float32)
        x_c = npd[1]*width
        y_c = npd[2]*height
        w   = npd[3]*width
        h   = npd[4]*height
        x1  = round(x_c-w*0.5)
        y1  = round(y_c-h*0.5)
        x2  = round(x_c+w*0.5)
        y2  = round(y_c+h*0.5)
        img[y1:y2,x1:x2] = 0
    return img

def maket0(yolo, device, t0s):
    imgs = []
    msks = []
    for t0 in t0s:
        list_detect, _ = run_yolov5(yolo, t0, device=device)
        img = cv2.imread(t0)
        msk = list2mask(list_detect, img) #[0-1]
        imgs.append(img)
        msks.append(msk)
    imgs = np.array(imgs, dtype=np.float32)
    msks = np.array(msks, dtype=np.float32)
    bs,h,w,ch = imgs.shape
    imgs = imgs.reshape([bs,-1])
    msks = msks.reshape([bs,-1])

    avg = np.mean(imgs,     axis=0)
    sum =  np.sum(imgs*msks,axis=0)
    cnt =  np.sum(     msks,axis=0)
    ind = np.where(cnt>0)[0]
    avg[ind] = sum[ind]/cnt[ind]

    avg = avg.reshape([h,w,ch])
    return avg.astype(np.uint8)

class predictor:
    def __init__(self, device="cuda",
                       weight_cdnet="../weights/model.pth",
                       weight_yolo5="../weights/yolov5x6.pt",
                       weight_depth="../weights/depth.pt",
                       is_and=True, is_or=False):
        self.device = device
        self.is_and = is_and
        self.is_or  = is_or

        ### Load CDNet
        s = time.time()
        self.model_CDNet = net().to(device)
        self.model_CDNet.load(path=weight_cdnet)
        print("CDNet load:", time.time()-s)

        ### Load YOLOv5
        s = time.time()
        self.model_yolov5 = load_yolov5(device, weights=weight_yolo5)
        print("YOLOv5 load:", time.time()-s)

        ### Load MiDas
        s = time.time()
        self.model_MiDas = Depth(model_path=weight_depth)
        print("MiDas load:", time.time()-s)

    def pre(self, t0s): # list of path only
        return maket0(self.model_yolov5, self.device, t0s)

    def readimg(self, path):
        img = cv2.imread(path)
        img = img.astype(np.float32)/255.0
        img = img[np.newaxis]
        return img

    def writeimg(self, path, img):
        if img.ndim == 4: img = img[0]
        cv2.imwrite(path, img)

    def getred(self, p, plim=128, c=None):
        p  = p[:,:,:, 1:2]

        if self.is_and:
            p = p*c
        elif self.is_or:
            p = np.maximum(p,1.0-c)

        p  = np.uint8(255* p)
        ip = np.where(p.reshape([-1])>plim)[0]
        rgb = np.zeros([p.shape[1],p.shape[2],3], dtype=np.uint8)
        h,w,ch = rgb.shape
        rgb = rgb.reshape([-1,3])
        rgb[ip] = [0,0,255]
        return rgb.reshape([h,w,ch])

    def inpaint(self, img, msk):
        img = np.uint8(255*img)[0]
        img = cv2.inpaint(img, msk, 3, cv2.INPAINT_TELEA)
        return img[np.newaxis].astype(np.float32)/255

    def overlay_mask(self,list_detect, img):
        height, width, ch = img.shape
        img = img.copy()
        for j in range(len(list_detect)):
            npd = np.array(list_detect[j],dtype=np.float32)
            x_c = npd[1]*width
            y_c = npd[2]*height
            w   = npd[3]*width
            h   = npd[4]*height
            x1  = round(x_c-w*0.5)
            y1  = round(y_c-h*0.5)
            x2  = round(x_c+w*0.5)
            y2  = round(y_c+h*0.5)
            img[y1:y2,x1:x2] = 0
        return img

    def __call__(self, base_img, targ_img, path_msk=None):
        s = time.time()
        t0 = self.readimg(base_img)
        t1 = self.readimg(targ_img)

        ### CDNet
        t0_pt = t0.transpose([0,3,1,2])
        t1_pt = t1.transpose([0,3,1,2])
        p,c = self.model_CDNet.prediction(t0_pt, t1_pt, device=self.device)
        p = p.transpose([0,2,3,1])
        c = c.transpose([0,2,3,1])
        npred = self.getred(p, c=c)

        ### YOLOv5
        list_detect, npimg_yolo0 = run_yolov5(self.model_yolov5, base_img, device=self.device)
        npredY = self.overlay_mask(list_detect, npred)

        list_detect, npimg_yolo1 = run_yolov5(self.model_yolov5, targ_img, device=self.device)
        npredY = self.overlay_mask(list_detect, npredY)

        ### MiDas
        if path_msk is not None:
            t1 = self.inpaint(t1, cv2.imread(path_msk,0))
        dep = self.model_MiDas(t1[0,:,:,::-1])
        dep = self.model_MiDas.get_depth(dep)

        print("process:", time.time()-s)
        return np.uint8(255*t1), npred, npimg_yolo0, npimg_yolo1, npredY, dep


if __name__ == '__main__':

    p = predictor()

    base_img = "_tmp/base1_202004281717.jpg"
    targ_img = "_tmp/targ1_202004281727.jpg"
    path_msk = "mask/mask.jpg"

    t1, npred, npimg_yolo0, npimg_yolo1, npredY, dep = p(base_img,targ_img,path_msk)

    p.writeimg(targ_img.replace(".jpg","_result1.jpg"), npred)
    p.writeimg(targ_img.replace(".jpg","_result2.jpg"), npimg_yolo1)
    p.writeimg(targ_img.replace(".jpg","_result3.jpg"), npredY)
    p.writeimg(targ_img.replace(".jpg","_result4.jpg"), dep)
