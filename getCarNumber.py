import torch
import cv2
from KaKaoOCR import kakao_ocr
# from tesseract_OCR import Image2String
import easyocr
from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt
import re
import numpy as np
import random
from Houghline import TransHough

COLORS = [(218,229,0),(173,0,186),(113,206,0)]

# 영상 폰트
# font = cv2.FONT_HERSHEY_SIMPLEX  # 글씨 폰트

#차량번호 정규표현식
re_car_num = re.compile('/^\D{2}\d{2}\D\d{4}$/')

#yolov5 모델 로드
model = torch.hub.load('yolov5','custom',path='ANPR_V2.pt',source='local') #yolov5 모델 load
# model.classes=[2] # 차량만 검출
# model.conf=0.5
reader = easyocr.Reader(['ko'],gpu=True)
def EasyOCR(img):
    result = reader.readtext(img)
    text=""
    for sentence in result:
        text += sentence[-2]
    return text

def detect(img):
    detects = model(img,size=640)
    
    crop_image=[]
    font = ImageFont.truetype("/usr/share/fonts/nanum/NanumBrush.ttf",20)
    
    for num,det in enumerate(detects.pandas().xyxy[0].values.tolist()):
        #Detect 결과
        [x1,y1,x2,y2,conf,cls,name] = det
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cls=int(cls)
        
        if cls == 1:
            car_plate = img[y1:y2, x1:x2]
            crop_image.append(car_plate)
            # car_plate=TransHough(car_plate)
            CarNumber = EasyOCR(car_plate)

            if CarNumber !="":
                print("OCR_RESULT",CarNumber)
                img=Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                draw.rectangle(((x1, y1), (x2, y2)), outline=COLORS[cls], width=2)  # bounding box
                draw.text((x1, y1-20), CarNumber , font=font, fill=COLORS[cls],stroke_width=1)
                 
        else:
            img=Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            draw.rectangle(((x1, y1), (x2, y2)), outline=COLORS[cls], width=2)  # bounding box
            draw.text((x1, y1 - 20), name , font=font, fill=COLORS[cls])  # 정확도
           
        # img=Image.fromarray(img)
        # draw = ImageDraw.Draw(img)
        
        
        img=np.array(img)

    return img,crop_image



if __name__ == '__main__':
    video_path="data/test3.mp4"
    cap = cv2.VideoCapture(video_path)
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./runs/output.mp4', fourcc, fps, (w, h))

    count=0
    while True:
        ret, frame = cap.read()

        if ret:
            view_img,crop_image = detect(frame)
            # for num,i in enumerate(crop_image):
            #     cv2.imwrite(f'runs/crop/{count}_{num}.jpg',i)
            # cv2.imshow("View_img",view_img)
            out.write(view_img)
            cv2.imwrite(f'runs/view/{count}.jpg',view_img)
            count+=1
            # key = cv2.waitKey(15)
            # if key ==27:
            #     break
        else:
            cap.release()
            out.release()
            break
