# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import os
import sys
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolo.models.experimental import attempt_load
from yolo.utils.datasets import LoadImages
from yolo.utils.general import check_img_size, colorstr, non_max_suppression, scale_coords, set_logging, xyxy2xywh
from yolo.utils.plots import Annotator, colors
from yolo.utils.torch_utils import time_sync

def load_yolov5(device, weights = "./weights/yolov5x6.pt"):
    w = str(weights[0] if isinstance(weights, list) else weights)
    model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
    return model

@torch.no_grad()
def run(model,
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):
    list_out=[]  ##### for return #####

    source = str(source)

    # Initialize
    set_logging()
    pt = True
    half = False
    device = torch.device(device)

    ###
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    ###

    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once

    dt, seen = [0.0, 0.0, 0.0], 0
    for path, img, im0s, vid_cap in dataset:
        t1 = time_sync()

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32

        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        if pt:
            pred = model(img, augment=augment, visualize=visualize)[0]

        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1

            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            if ( not nosave ):
                p = Path(p)  # to Path

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    ##### for return #####
                    _xywh = (xyxy2xywh(torch.tensor(xyxy).view(-1, 4)))            # boxes
                    gain = 1.02 ; pad = 10                                         # from [utils.general, save_one_box]
                    _xywh[:, 2:] = _xywh[:, 2:] * gain + pad                       #   box wh * gain + pad
                    _xywh = (_xywh/gn).view(-1).tolist()                           # normalized xywh
                    list_out.append([cls.cpu().item(), *_xywh, conf.cpu().item()]) # label format
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Print time (inference-only)
            print(f'{s}Done. ({t3 - t2:.3f}s)')

            im0 = annotator.result()

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    # print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    print(f'{colorstr("bold","Speed:")} %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    return list_out, im0


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # print_args(FILE.stem, opt)
    return opt

def run_yolov5(model,img_path:str,device):
    opt = parse_opt()
    opt.device    = device
    opt.source    = img_path
    # opt.weights   = "./weights/yolov5x6.pt"
    # opt.imgsz     = [1280,1280]
    # opt.nosave = True
    list_out, img = run(model,**vars(opt))
    return list_out, img

if __name__ == "__main__":
    pass
