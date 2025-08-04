# 배경 제거

import cv2
import numpy as np

# 초기설정
cap = cv2.VideoCapture('../img/walking.avi')
#cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)

color = np.random.randint(0,255,(200,3))
lines = None # 추적 선을 그릴 이미지 저장 변수
previmg = None # 이전 이미지 저장 변수
# calcOpticalFlowPyrLK 중지 요건 설정
termcriteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    img_draw = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 최초 프레임인 경우
    if previmg is None:
        previmg = gray
        # 추적 선을 그릴 이미지를 프레임 크기에 맞게 생성
        lines = np.zeros_like(frame)
        # 추적을 위한 코너 검출
        prevPt = cv2.goodFeaturesToTrack(previmg, 200, 0.01, 10)
    else: 
        nextimg = gray
        # 옵티컬 플로우로 다음 프레임의 코너 찾기
        nextPt, status, err = cv2.calcOpticalFlowPyrLK(previmg, nextimg, \
                                                       prevPt, None, criteria = termcriteria)
        # 대응점이 있는 코너와 움직인 코너 선별
        prevMv = prevPt[status==1]
        nextMv = nextPt[status==1]
        for i, (p, n) in enumerate(zip(prevMv, nextMv)):
            px, py = int(p.ravel()[0]), int(p.ravel()[1])
            nx, ny = int(n.ravel()[0]), int(n.ravel()[1])
            # 이전 코너와 새 코너에 선 그리기
            cv2.line(lines, (px, py), (nx, ny), color[i].tolist(), 2)
            # 새로운 코너에 점 그리기
            cv2.circle(img_draw, (nx, ny), 2, color[i].tolist(), -1)
        # 누적된 추적 선을 출력 이미지에 합성
        img_draw = cv2.add(img_draw, lines)
        # 다음 프레임을 위한 프레임과 코너 이월
        previmg = nextimg
        prevPt = nextMv.reshape(-1, 1, 2)

    cv2.imshow('OpticalFlow-LK', img_draw)
    key = cv2.waitKey(delay)

    if cv2.waitKey(1) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()