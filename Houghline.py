import cv2
import numpy as np
import math

def TransHough(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray,100,10)
    # cv2.imshow("Canny",edges)
    lines = cv2.HoughLines(edges,1,np.pi/180,120)
    min_theta = np.pi/2
    if lines is not None:
        for line in lines:
            r,theta = line[0]

            if (theta < min_theta and theta>0 ):
                min_theta = theta

    ver, hor = img_gray.shape

    center = int(hor/2),int(ver/2)
    degree = -math.degrees((np.pi/2)-min_theta)
    rotate = cv2.getRotationMatrix2D(center,degree,1)
    res_rotate = cv2.warpAffine(img_gray,rotate,(hor,ver))
    # while True:
    #     cv2.imshow("original",img)
    #     cv2.imshow("rotate", res_rotate)
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         break
    return res_rotate

if __name__ == "__main__":
    # img_path="runs/number_plate/crop/20_0.jpg"
    img_path="runs/number_plate/crop/226_0.jpg"
    img = cv2.imread(img_path)
    TransHough(img)