import cv2
import pytesseract

def Image2String(img):
    gray
    text = pytesseract.image_to_string(img)
    print(text)
    return text

if __name__ == "__main__":
    image = cv2.imread("data/TCmbJ.png")
    Image2String(image)