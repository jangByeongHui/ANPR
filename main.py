import torch
import cv2
import easyocr
from util.detectElectronic import isElectronic
from util.KaKaoOCR import kakao_ocr

COLORS = [(218,229,0),(173,0,186),(113,206,0)]


#yolov5 모델 로드
model = torch.hub.load('ultralytics/yolov5','custom',path = 'ALPR_V1.pt',force_reload=True) #yolov5 모델 load
reader = easyocr.Reader(['ko', 'en'], gpu=True)

def EasyOCR(img_path):
    result = reader.readtext(img_path)
    text = ""
    for sentence in result:
        text += sentence[-2]
    return text

def detect(img,img_path):

    detects = model(img)

    for num,det in enumerate(detects.pandas().xyxy[0].values.tolist()):

        #Detect 결과
        [x1,y1,x2,y2,conf,cls,name] = det

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cls=int(cls)

        cv2.imwrite(f'{img_path}_crop_img.png', img[y1:y2, x1:x2])
        OCR_result = EasyOCR(f'{img_path}_crop_img.png')
        # OCR_result = kakao_ocr(img[y1:y2, x1:x2])
        if isElectronic(img[y1:y2,x1:x2],100)[0]:
            # 친환경 전기차 인경우
            print(f'친환경 자동차 OCR : {OCR_result}')
            # Using cv2.putText() method
            # img = cv2.putText(img, 'Electronic', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
            #                     0.3, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            # 친환경 자동차 아님
            print(f'일반 자동차 OCR : {OCR_result}')

        img = cv2.putText(img,f'{name} {conf}', (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.5, COLORS[cls], 2, cv2.LINE_AA)
        img = cv2.rectangle(img,(x1,y1),(x2,y2),COLORS[cls],2)

    return img




if __name__ == '__main__':

    # 이미지
    img_path_LIST = ['runs/test/0.png','runs/test/1.png','runs/test/2.png','runs/test/3.png','runs/test/4.png']

    for img_path in img_path_LIST:
        Img = cv2.imread(img_path)
        view_img = detect(Img,img_path)
        cv2.imwrite(f'{img_path}_result.jpg',view_img)


    # 비디오
    vid_path="data/test.mp4"

    cap = cv2.VideoCapture(vid_path)
    frame_count = 0
    while True:
        ret,frame = cap.read()

        if ret:
            view_img = detect(frame,f'{vid_path}_{frame_count}')
            cv2.imwrite(f'{vid_path}_{frame_count}_result.jpg', view_img)
            frame_count += 1
        else:
            cap.release()
            break