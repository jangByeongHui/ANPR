import cv2
import pytesseract

block_size=3
maxValue=0
C=0

def maxValue_change(pos):
    global maxValue
    maxValue=pos
    print(f'maxvalue:{maxValue} block_size : {block_size} C : {C}')

def block_size_change(pos):
    global block_size
    if pos>2:
        if pos%2==0:
            block_size=pos+1
        else:
            block_size=pos
    else:
        pass
    print(f'maxvalue:{maxValue} block_size : {block_size} C : {C}')

def C_change(pos):
    global C
    C=pos
    print(f'maxvalue:{maxValue} block_size : {block_size} C : {C}')

def Image2String(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_blurred = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=2)
    # while True:
    #     cv2.imshow('img_blurred',img_blurred)
    #     key = cv2.waitKey(1)
    #     if key ==27:
    #         break
    img_thresh = cv2.adaptiveThreshold(
        gray,
        maxValue=maxValue,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=C
    )
    text = pytesseract.image_to_string(img_thresh,lang='kor')
    print(text)
    cv2.namedWindow('imgThresh')
    cv2.createTrackbar("maxValue", "imgThresh", 0, 255, maxValue_change)
    cv2.createTrackbar("block_size", "imgThresh", 3, 100,block_size_change)
    cv2.createTrackbar("C", "imgThresh", 0, 100, C_change)

    cv2.setTrackbarPos("maxValue", "imgThresh", 0)
    cv2.setTrackbarPos("block_size", "imgThresh", 3)
    cv2.setTrackbarPos("C", "imgThresh", 0)

    while True:
        img_thresh = cv2.adaptiveThreshold(
            gray,
            maxValue=maxValue,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=block_size,
            C=C
        )
        cv2.imshow('imgThresh',img_thresh)
        key = cv2.waitKey(1)
        if key ==27:
            break

    return text

if __name__ == "__main__":
    image = cv2.imread("data/black.png")
    Image2String(image)