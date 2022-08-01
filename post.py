
import os, cv2, time
from PIL import Image
import numpy as np

class postprocess:
    def __init__(self, a=0.5, limit=100.0, dscale=100):
        self.a      = a
        self.limit  = limit
        self.dscale = dscale

    def mor(self, red, redlimit=10, nmor=3, imor=1):
        ret,red = cv2.threshold(red,redlimit,255,cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(nmor,nmor))
        #red =  cv2.erode(red,kernel,iterations=imor)
        red = cv2.dilate(red,kernel,iterations=imor)
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)
        return red

    def depth_avg(self, dep, msk):
        dep = dep.reshape([-1])
        msk = msk.reshape([-1])
        ind = np.where(msk>128)[0]
        return np.mean(dep[ind])

    def paste(self, img, results):
        img = Image.fromarray(img)
        for result in results:
            msk  = result["msk"].astype(np.float32)

            red  = np.zeros_like(img)
            red[:,:,2] = 255
            red  = Image.fromarray(red.astype(np.uint8))

            msk *= 0.5
            msk  = Image.fromarray(msk.astype(np.uint8))

            img.paste(red,(0,0),msk)
        img = np.asarray(img)
        for result in results:
            scr  = result["score"]
            bbx  = result["bbox"]

            img = cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[2], bbx[3]), (0,0,255), 2)
            cv2.putText(img, "{:4.1f}".format(scr), (bbx[2], bbx[3]),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.0,
                        color=(0, 255, 0),thickness=2,lineType=cv2.LINE_4)

        return img

    def __call__(self, org, red, dep):
        if dep.ndim == 3: dep = dep[:,:,-1]
        if red.ndim == 3: red = red[:,:,-1]
        dep = 1.0 - dep/255.0
        red = self.mor(red)

        logs = []
        results = []
        contours, hierarchy = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i,contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > self.limit:
                xx,yy,ww,hh = cv2.boundingRect(contour)
                new = np.zeros_like(red)
                new = cv2.drawContours(new, [contour], -1, color=(255), thickness=-1)

                depth_avg = self.depth_avg(dep, new)
                if depth_avg > self.a:
                    rate  = (depth_avg-self.a)/(1.0-self.a)
                    score = area/self.dscale*rate
                    logs.append([i,score,rate,])
                    results.append({"msk":new,"score":score,"bbox":[xx,yy,xx+ww,yy+hh]})

        if len(results) > 0: org = self.paste(org.copy(), results)
        return org, results, logs


if __name__ == '__main__':

    p = postprocess()

    org = cv2.imread("_tmp/targ1_202004281727.jpg")
    red = cv2.imread("_tmp/targ1_202004281727_result1.jpg")
    dep = cv2.imread("_tmp/targ1_202004281727_result4.jpg")
    s = time.time()
    out,results = p(org, red, dep)
    print(time.time()-s) #0.01184844970703125
    cv2.imwrite("_tmp/targ1_202004281727_result5.jpg", out)
