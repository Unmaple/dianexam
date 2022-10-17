import cv2
import os
import numpy as np
from PIL import Image
# fname = r'F:\steet_character_detector\data\01'
# key=fname
# img = cv2.imread(fname+".png")
# width=img.shape[1]
# height=img.shape[0]
# f=open(fname+".txt")
# lines=f.readlines()
# for line in lines :
#     para=list(map(float,line.split()))
#     ptLeftTop=(int((para[1]-para[3]/2)*width),int((para[2]-para[4]/2)*height))
#     ptRightBottom=(int((para[1]+para[3]/2)*width),int((para[2]+para[4]/2)*height))
#     cv2.putText(img,str(para[0]),ptLeftTop,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),thickness=1)
#     point_color=(0,255,0)
#     thickness= 1
#     cv2.rectangle(img, ptLeftTop,ptRightBottom,point_color,thickness=1)
# f.close
# cv2.namedWindow("result",0)
# cv2.resizeWindow("result",800,600)
# cv2.imshow("result",img)
# cv2.waitKey(0)
def showing(lines,Img):
    img = cv2.cvtColor(np.asarray(Img), cv2.COLOR_RGB2BGR)
    width=img.shape[1]
    height=img.shape[0]
    for line in lines:
        para = line
        ptLeftTop = (int((para[1] - para[3] / 2) * width), int((para[2] - para[4] / 2) * height))
        ptRightBottom = (int((para[1] + para[3] / 2) * width), int((para[2] + para[4] / 2) * height))
        cv2.putText(img, str(para[0]), ptLeftTop, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=1)
        point_color = (0, 255, 0)
        thickness = 1
        cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness=1)
    #f.close
    cv2.namedWindow("result", 0)
    cv2.resizeWindow("result", 800, 600)
    cv2.imshow("result", img)
    cv2.waitKey(0)
