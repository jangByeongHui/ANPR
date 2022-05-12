import torch
import cv2

COLORS = [(218,229,0),(173,0,186),(113,206,0)]


#yolov5 모델 로드
model = torch.hub.load('yolov5','custom',path='ANPR_V2.pt',source='local') #yolov5 모델 load


def detect(img):
    detects = model(img,size=416)
    for num,det in enumerate(detects.pandas().xyxy[0].values.tolist()):

        #Detect 결과
        [x1,y1,x2,y2,conf,cls,name] = det

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cls=int(cls)

    return img,crop_image[y1:y2,x1:x2]



if __name__ == '__main__':
    img_path = 'runs/test.jpg'

    view_img,crop_image = detect(cv2.imread(img_path))

    cv2.imshow('View',crop_image)
    while True:
        status = cv2.waitKey(1)

        if status==27:
            break
