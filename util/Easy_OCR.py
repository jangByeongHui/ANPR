import cv2
import easyocr

def EasyOCR(img_path):
    reader = easyocr.Reader(['ko','en'],gpu=True)
    result = reader.readtext(img_path)
    text=""
    for sentence in result:
        text += sentence[-2]
    return text

if __name__ == "__main__":
    img_path="runs/crop/kor_crop.png"
    img = cv2.imread(img_path)
    print(EasyOCR(img))