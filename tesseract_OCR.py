import cv2
import pytesseract


def Image2String(img):

    # text = pytesseract.image_to_string(img,lang='kor')
    text = pytesseract.image_to_string(img)
    print(text)
    return text

if __name__ == "__main__":
    image = cv2.imread("runs/crop/kor.jpeg")
    Image2String(image)