# 배경 제거

import cv2
import numpy as np

# 초기설정
# cap = cv2.VideoCapture('../img/walking.avi')
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)

# 배경 제거 객체 생성
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

# createBackgroundSubtractorMOG2()의 매개변수
# history: 과거 프레임의 개수, 배경을 학습하는데 얼마나 많은 프레임을 기억할지
# varThreshold: 픽셀이 배경인지 객체인지 구분할 기준값
fgbg2 = cv2.createBackgroundSubtractorMOG2(500, detectShadows=False)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 배경 제거 마스크 계산
    fgmask = fgbg.apply(frame)
    fgmask2 = fgbg2.apply(frame)
    cv2.imshow('frame', frame)
    cv2.imshow('bgsub', fgmask)
    cv2.imshow('bgsub2', fgmask2)
    if cv2.waitKey(1) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()