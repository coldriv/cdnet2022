# coding: utf-8
import cv2
from  pre import edge_iou, select_t0s
from  all import predictor
from post import postprocess

def allrun(m, p0s, p1, pm=None,
           p0 ="_tmp/t0.jpg",
           r1 ="_tmp/result1.jpg",
           r2 ="_tmp/result2.jpg",
           r3 ="_tmp/result3.jpg",
           r4 ="_tmp/result4.jpg",
           out="_tmp/score.jpg",
           log="_tmp/log.txt",
           a  = 0.5):

    p0s = select_t0s(p0s)
    t0 = m.pre(p0s)
    cv2.imwrite(p0, t0)

    if edge_iou(p0,p1):
        npimg_targ, npimg_red, npimg_yolo0, npimg_yolo1, npimg_red_withYolo, npimg_dep = m(p0,p1,path_msk=pm)
        cv2.imwrite(r1, npimg_red)
        cv2.imwrite(r2, npimg_yolo1)
        cv2.imwrite(r3, npimg_red_withYolo)
        cv2.imwrite(r4, npimg_dep)
    else:
        print("[do nothing!] edge_iou is lower limit:", edge_iou(p0,p1))

    p = postprocess(a)
    npimg_out,results,logs = p(npimg_targ[0], npimg_red_withYolo, npimg_dep)
    cv2.imwrite(out, npimg_out)
    with open(log,"w") as f:
        f.write("i,score,depth\n")
        for log in logs: f.write("{},{},{}\n".format(*log))

if __name__ == '__main__':

    m = predictor(device="cuda",
                  weight_cdnet="../weights/model.pth",
                  weight_yolo5="../weights/yolov5x6.pt",
                  weight_depth="../weights/depth.pt")

    allrun(m,
           p0s = ["../_tmp/20210726145000.jpg",
                  "../_tmp/20210726150001.jpg",
                  "../_tmp/20210726151002.jpg",
                  "../_tmp/20210726152002.jpg"],
           p1  =  "../_tmp/20210726153001.jpg",
           pm  =  "mask/mask_004.jpg")

#loading  ../weights/model.pth
#loaded
#CDNet load: 3.972203016281128
#YOLOv5 load: 0.33254575729370117
#MiDas load: 5.659988880157471
#image 1/1 /mnt/synology/DIGITS_DATA/abe21107/IAIR21/cctv/11.sys/_tmp/20210726145000.jpg: 896x1280 3 cars, Done. (0.059s)
#Speed: 0.8ms pre-process, 58.9ms inference, 1.1ms NMS per image at shape (1, 3, 1280, 1280)
#image 1/1 /mnt/synology/DIGITS_DATA/abe21107/IAIR21/cctv/11.sys/_tmp/20210726150001.jpg: 896x1280 1 truck, Done. (0.059s)
#Speed: 0.7ms pre-process, 58.7ms inference, 0.8ms NMS per image at shape (1, 3, 1280, 1280)
#image 1/1 /mnt/synology/DIGITS_DATA/abe21107/IAIR21/cctv/11.sys/_tmp/20210726151002.jpg: 896x1280 Done. (0.059s)
#Speed: 0.7ms pre-process, 59.2ms inference, 0.6ms NMS per image at shape (1, 3, 1280, 1280)
#image 1/1 /mnt/synology/DIGITS_DATA/abe21107/IAIR21/cctv/11.sys/_tmp/20210726152002.jpg: 896x1280 1 truck, Done. (0.059s)
#Speed: 0.7ms pre-process, 58.7ms inference, 0.8ms NMS per image at shape (1, 3, 1280, 1280)
#image 1/1 /mnt/synology/DIGITS_DATA/abe21107/IAIR21/cctv/11.sys/pytorch_yolo_depth/_tmp/t0.jpg: 896x1280 1 bench, Done. (0.059s)
#Speed: 0.7ms pre-process, 58.7ms inference, 0.9ms NMS per image at shape (1, 3, 1280, 1280)
#image 1/1 /mnt/synology/DIGITS_DATA/abe21107/IAIR21/cctv/11.sys/_tmp/20210726153001.jpg: 896x1280 3 cars, 1 truck, Done. (0.058s)
#Speed: 0.7ms pre-process, 57.7ms inference, 0.8ms NMS per image at shape (1, 3, 1280, 1280)
#/home/abe21107/anaconda3/lib/python3.7/site-packages/torch/nn/functional.py:3458: UserWarning: Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0. Please specify align_corners=True if the old behavior is desired. See the documentation of nn.Upsample for details.
#  "See the documentation of nn.Upsample for details.".format(mode)
#process: 0.7030196189880371

