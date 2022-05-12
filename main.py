import torch
import cv2
from detectElectronic import isElectronic

COLORS = [(218,229,0),(173,0,186),(113,206,0)]


#yolov5 모델 로드
model = torch.hub.load('ultralytics/yolov5','custom',path = 'ALPR_V1.pt',force_reload=True) #yolov5 모델 load


def detect(img):

    detects = model(img)

    for num,det in enumerate(detects.pandas().xyxy[0].values.tolist()):

        #Detect 결과
        [x1,y1,x2,y2,conf,cls,name] = det

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cls=int(cls)
        if isElectronic(img[y1:y2,x1:x2],2000):
            # 친환경 전기차 인경우
            print("친환경 자동차 OCR 실행")
        else:
            # 친환경 자동차 아님
            print("OCR 미실행")
        img = cv2.rectangle(img,(x1,y1),(x2,y2),COLORS[cls],2)


    return img




if __name__ == '__main__':
    img_path = 'runs/test.jpg'
    Img = cv2.imread(img_path)

    view_img = detect(Img)

    cv2.imshow('View',view_img)
    while True:
        status = cv2.waitKey(1)

        if status==27:
            break
