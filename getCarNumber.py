import torch
import cv2
import requests
import re
import copy
import json

#카카오 API
appkey = "71234b5024f98714a62dfb31d7c988c9"
API_URL = 'https://dapi.kakao.com/v2/vision/text/ocr'
headers = {'Authorization': 'KakaoAK {}'.format(appkey)}

# 영상 폰트
font = cv2.FONT_HERSHEY_SIMPLEX  # 글씨 폰트

#차량번호 정규표현식
re_car_num = re.compile('/^\D{2}\d{2}\D\d{4}$/')

#yolov5 모델 로드
model = torch.hub.load('yolov5','custom',path='yolov5s.pt',source='local') #yolov5 모델 load
model.classes=[2] # 차량만 검출
model.conf=0.5

def OCR(img):
    jpeg_image = cv2.imencode(".jpg", img)[1]
    data = jpeg_image.tobytes()

    response = requests.post(API_URL, headers=headers, files={"image": data}).json()

    for words in response['result']:
        print(words)
        # if re_car_num is not None: # 차량 정규 표현식
        return words['recognition_words'][0]
    else:
        return None


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
        # CarNumber = OCR(img[y1:y2,x1:x2])
        # if CarNumber is not None:
        #     cv2.putText(img,CarNumber, ((x1+x2)//2, y1 - 5), font, 0.5, (0, 255, 0), 2)  # 차량번호
        cv2.putText(img, "{:.2f}".format(conf), (x2, y1 - 5), font, 0.5, (255, 0, 0), 2)  # 정확도
    return img,crop_image



if __name__ == '__main__':
    video_path="data/test.mp4"
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()

        if ret:
            view_img,crop_image = detect(frame)
            for num,i in enumerate(crop_image):
                cv2.imshow(f'{num}', i)
                cv2.imshow('Temp_image',  view_img)
            key = cv2.waitKey(1)
            if key ==27:
                break

        else:
            cap.release()
            break
