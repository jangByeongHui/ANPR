import cv2
import easyocr

def EasyOCR(img):
    reader = easyocr.Reader(['ko','en'],gpu=False)
    result = reader.readtext(img)
    text=""
    for sentence in result:
        text += sentence[-2]
    return text

if __name__ == "__main__":
    img_path="runs/crop/kor_crop.png"
    img = cv2.imread(img_path)
    print(EasyOCR(img))