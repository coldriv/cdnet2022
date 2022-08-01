
import numpy as np
import cv2, time
import torch
from torchvision.transforms import Compose
from depth.midas.dpt_depth import DPTDepthModel
from depth.midas.midas_net import MidasNet
from depth.midas.midas_net_custom import MidasNet_small
from depth.midas.transforms import Resize, NormalizeImage, PrepareForNet

class Depth:
    def __init__(self, model_path="weights/depth.pt", model_type="dpt_large", optimize=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == "dpt_large": # DPT-Large
            model = DPTDepthModel(
                path=model_path,
                backbone="vitl16_384",
                non_negative=True,
            )
            net_w, net_h = 384, 384
            resize_mode = "minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            print(f"model_type '{model_type}' not implemented, use: --model_type large")
            assert False

        transform = Compose( [
                Resize( net_w, net_h,
                        resize_target=None,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        resize_method=resize_mode,
                        image_interpolation_method=cv2.INTER_CUBIC ),
                normalization,
                PrepareForNet() ] )

        model.eval()

        if optimize==True:
            if device == torch.device("cuda"):
                model = model.to(memory_format=torch.channels_last)
                model = model.half()
        model.to(device)

        self.device = device
        self.optimize = optimize
        self.transform = transform
        self.model = model

    def read_image(self, path):
        img = cv2.imread(path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        return img

    def get_depth(self, depth, bits=1):
        depth_min = depth.min()
        depth_max = depth.max()
        max_val = (2**(8*bits))-1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.type)

        if bits == 1:
            out = out.astype("uint8")
        elif bits == 2:
            out = out.astype("uint16")
        return out

    def write_depth(self, path, depth):
        cv2.imwrite(path, self.get_depth(depth))

    def __call__(self, img): #(h,w,ch),[0.0-1.0],[RGB]
        img_input = self.transform({"image": img})["image"]
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
            if self.optimize==True and self.device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()
            prediction = self.model.forward(sample)
            prediction = ( torch.nn.functional.interpolate(prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False).squeeze().cpu().numpy() )
        return prediction

if __name__ == "__main__":

    s = time.time()
    d = Depth()
    print("over head", time.time()-s)

    i = d.read_image("_tmp/202004281727_intpl.jpg")

    s = time.time()
    o = d(i)
    print("prediction:", time.time()-s)

    d.write_depth("_tmp/202004281727_intpl_abebe.png", o)
