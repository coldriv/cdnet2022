
import os,time,cv2
from glob import glob
import numpy as np
#from kym import run_yolov5, load_yolov5 # for test yolo assist

def norm(img):
    h,w = img.shape
    img = img.astype(np.float32)
    img = img.reshape([-1])
    avg = np.mean(img, axis=0)
    img = img - avg + 128.0
    img = np.clip(img,0,255)
    img = img.astype(np.uint8)
    return img.reshape([h,w])

def edge(path, nblur=5, ksize=3, limit=96):
    img    = cv2.imread(path, 0)[100:-100]
    img    = cv2.blur(img, (nblur,nblur))
    img    = norm(img)
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
    sobel  = np.maximum(sobelx,sobely)
    sobel  = np.where(sobel>limit, 255, 0)
    return sobel.astype(np.uint8)

def edge_iou(path0, path1, limit=5.0):
    edge0 = edge(path0)
    edge1 = edge(path1)

    _and  = edge0.astype(np.float32)*edge1.astype(np.float32)/255
    _and  = _and.astype(np.uint8)
    _or   = np.maximum(edge0,edge1)
    iou   = 100*np.sum(_and)/np.sum(_or)
    return iou

def select_t0s(t0s, limit=5):
    new = [t0s[-1]]
    t0s = t0s[:-1]
    t0s = t0s[::-1]
    for t0 in t0s:
        if edge_iou(new[-1], t0):
            new.append(t0)
        else:
            break
    return new[::-1]

#def list2mask(list_detect,img):
#    height, width, ch = img.shape
#    img = np.ones_like(img)
#    for j in range(len(list_detect)):
#        npd = np.array(list_detect[j],dtype=np.float32)
#        x_c = npd[1]*width
#        y_c = npd[2]*height
#        w   = npd[3]*width
#        h   = npd[4]*height
#        x1  = round(x_c-w*0.5)
#        y1  = round(y_c-h*0.5)
#        x2  = round(x_c+w*0.5)
#        y2  = round(y_c+h*0.5)
#        img[y1:y2,x1:x2] = 0
#    return img
#
#def maket0(yolo, device, t0s):
#    imgs = []
#    msks = []
#    for t0 in t0s:
#        list_detect, _ = run_yolov5(yolo, t0, device=device)
#        img = cv2.imread(t0)
#        msk = list2mask(list_detect, img) #[0-1]
#        imgs.append(img)
#        msks.append(msk)
#    imgs = np.array(imgs, dtype=np.float32)
#    msks = np.array(msks, dtype=np.float32)
#    bs,h,w,ch = imgs.shape
#    imgs = imgs.reshape([bs,-1])
#    msks = msks.reshape([bs,-1])
#
#    avg = np.mean(imgs,     axis=0)
#    sum =  np.sum(imgs*msks,axis=0)
#    cnt =  np.sum(     msks,axis=0)
#    ind = np.where(cnt>0)[0]
#    avg[ind] = sum[ind]/cnt[ind]
#
#    avg = avg.reshape([h,w,ch])
#    return avg.astype(np.uint8)

if __name__ == "__main__":

    t0s = ["../_tmp/20210726145000.jpg",
           "../_tmp/20210726150001.jpg",
           "../_tmp/20210726151002.jpg",
           "../_tmp/20210726152002.jpg"]
    t1  = ["../_tmp/20210726153001.jpg"]

    t0s = select_t0s(t0s)
    print(t0s)

    #device="cuda"
    #s = time.time()
    #yolo = load_yolov5(device, weights="../weights/yolov5x6.pt")
    #print("YOLOv5 load:", time.time()-s)

    #t0 = maket0(yolo, device, t0s)
    #cv2.imwrite("t0.jpg", t0)

    #imgs = []
    #for t0 in t0s:
    #    imgs.append(cv2.imread(t0))
    #img = np.mean(imgs, axis=0)
    #print(img.shape)
    #cv2.imwrite("t0_brefore.jpg", img.astype(np.uint8))
