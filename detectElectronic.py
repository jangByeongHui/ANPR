import cv2
import numpy as np

LowHue=0
LowSat=0
LowVal=0

HighHue=0
HighSat=0
HighVal=0
def isElectronic(img):
    cv2.imshow('Test',img)
    return True

def findElectronic(img):
    global LowHue,LowSat,LowVal,HighHue,HighSat,HighVal

    cv2.namedWindow('findElectronic',flags=cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar('LOW HUE','findElectronic',0,180,onChange)
    cv2.createTrackbar('HIGH HUE', 'findElectronic', 0, 180, onChange)

    cv2.createTrackbar('LOW Saturation', 'findElectronic', 0, 255,onChange)
    cv2.createTrackbar('HIGH Saturation', 'findElectronic', 0, 255, onChange)

    cv2.createTrackbar('LOW Value', 'findElectronic', 0, 255,onChange)
    cv2.createTrackbar('HIGH Value', 'findElectronic', 0, 255,onChange)

    cv2.setTrackbarPos('LOW HUE', 'findElectronic',94)
    cv2.setTrackbarPos('LOW Saturation', 'findElectronic',56)
    cv2.setTrackbarPos('LOW Value', 'findElectronic',70)

    cv2.setTrackbarPos('HIGH HUE', 'findElectronic',104)
    cv2.setTrackbarPos('HIGH Saturation', 'findElectronic',233)
    cv2.setTrackbarPos('HIGH Value', 'findElectronic',255)


    while True:
        LowHue = cv2.getTrackbarPos('LOW HUE','findElectronic')
        LowSat = cv2.getTrackbarPos('LOW Saturation','findElectronic')
        LowVal = cv2.getTrackbarPos('LOW Value','findElectronic')

        HighHue = cv2.getTrackbarPos('HIGH HUE','findElectronic')
        HighSat = cv2.getTrackbarPos('HIGH Saturation','findElectronic')
        HighVal = cv2.getTrackbarPos('HIGH Value','findElectronic')

        # HSV로 색 추출
        hsvLower = np.array([LowHue, LowSat, LowVal]) # 추출할 색의 하한(HSV)
        hsvUpper = np.array([HighHue, HighSat, HighVal]) # 추출할 색의 상한(HSV)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # 이미지를 HSV으로

        hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)  #HSV에서 마스크를 작성

        result = cv2.bitwise_and(img, img, mask=hsv_mask)  # 원래 이미지와 마스크를 합성

        Show_img = cv2.hconcat([img,result])
        cv2.imshow('findElectronic',Show_img)

        status = cv2.waitKey(2000)
        if status == 27:
            print("-------Final---------")
            print(
                f'LOW HUE :{LowHue}\nLOW Saturation : {LowSat}\nLOW Value : {LowVal}\nHIGH HUE :{HighHue}\nHIGH Saturation : {HighSat}\nHIGH Value : {HighVal}')
            print("-----------------------\n")
            break


def onChange(pos):
    print("-----------------------")
    print(f'LOW HUE :{LowHue}\nLOW Saturation : {LowSat}\nLOW Value : {LowVal}\nHIGH HUE :{HighHue}\nHIGH Saturation : {HighSat}\nHIGH Value : {HighVal}')
    print("-----------------------\n")




if __name__ == '__main__':
    test_img_path ="runs/test.jpg_resized.jpeg"

    test_img = cv2.imread(test_img_path)

    findElectronic(test_img)