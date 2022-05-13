import torch
import cv2
from util.detectElectronic import isElectronic
from util.Easy_OCR import EasyOCR
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

        cv2.imwrite('crop_img.jpg', img[y1:y2, x1:x2])
        OCR_result = EasyOCR('crop_img.jpg')
        if isElectronic(img[y1:y2,x1:x2],2000):
            # 친환경 전기차 인경우
            print(f'친환경 자동차 OCR : {OCR_result}')
            # Using cv2.putText() method
            img = cv2.putText(img, 'Electronic', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                0.3, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            # 친환경 자동차 아님
            print(f'OCR 미실행 : {OCR_result}')
            img = cv2.putText(img, 'Non-Electronic', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                              0.3, (0, 255, 0), 2, cv2.LINE_AA)

        img = cv2.rectangle(img,(x1,y1),(x2,y2),COLORS[cls],2)

    return img




if __name__ == '__main__':
    img_path = ['runs/test/0.png','runs/test/1.png','runs/test/2.png','runs/test/3.png']

    for img in img_path:
        Img = cv2.imread(img_path)
        view_img = detect(Img)
        cv2.imwrite(f'result.jpg',view_img)