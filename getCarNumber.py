import torch
import cv2
from KaKaoOCR import kakao_ocr
from tesseract_OCR import Image2String

import re



# 영상 폰트
font = cv2.FONT_HERSHEY_SIMPLEX  # 글씨 폰트

#차량번호 정규표현식
re_car_num = re.compile('/^\D{2}\d{2}\D\d{4}$/')

#yolov5 모델 로드
model = torch.hub.load('yolov5','custom',path='car_plate_v1.pt',source='local') #yolov5 모델 load
# model.classes=[2] # 차량만 검출
# model.conf=0.5

def detect(img):
    detects = model(img,size=640)
    crop_image=[]
    for num,det in enumerate(detects.pandas().xyxy[0].values.tolist()):
        #Detect 결과
        [x1,y1,x2,y2,conf,cls,name] = det
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        crop_image.append(img[y1:y2, x1:x2])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # bounding box
        CarNumber = kakao_ocr(img[y1:y2,x1:x2])
        if CarNumber is not None:
             cv2.putText(img,CarNumber, (x2, y1 - 5), font, 0.5, (0, 255, 0), 2)  # 차량번호
        cv2.putText(img, "{:.2f}".format(conf), (x1, y1 - 5), font, 0.5, (255, 0, 0), 2)  # 정확도
    return img,crop_image



if __name__ == '__main__':
    video_path="data/test.mp4"
    cap = cv2.VideoCapture(video_path)
    count=0
    while True:
        ret, frame = cap.read()

        if ret:
            view_img,crop_image = detect(frame)
            for num,i in enumerate(crop_image):
                cv2.imshow(f'{num}', i)
                cv2.imwrite(f'runs/crop/{count}_{num}.jpg',i)
            cv2.imshow('Temp_image',  view_img)
            cv2.imwrite(f'runs/view/{count}.jpg',view_img)
            count+=1
            key = cv2.waitKey(1)
            if key ==27:
                break

        else:
            cap.release()
            break
