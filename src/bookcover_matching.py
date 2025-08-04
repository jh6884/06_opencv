import cv2
import numpy as np
import glob

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

win_name = 'Book cover matching'
search_dir = '../img/books/'
search_path = glob.glob(search_dir + '*.jpg')
roi = None
roi_on = False
MIN_MATCH = 10
result = []

detector = cv2.ORB_create(1000)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                   table_number=6,
                   key_size=12,
                   multi_probe_level=1)
search_params = dict(checks=32)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

end_switch = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('Camera required')
        break
    
    if roi is None and end_switch is not True:
        res = frame
        cv2.imshow(win_name, res)
    elif end_switch is True:
        print(f"검색이 완료되었습니다. \n추정되는 책 파일명 : ", end='')
        for i in result:
            print(i[1:], end = ', ')
        print("\n종료하려면 아무 키나 누르십시오...")
        cv2.waitKey(0)
        break
    else:
        for path in search_path:
            img = cv2.imread(path)
            cv2.imshow('searching...', img)
            cv2.waitKey(5)
            grayROI = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            grayIMG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kpR, descR = detector.detectAndCompute(grayROI, None) # ROI의 특징점
            kpI, descI = detector.detectAndCompute(img, None) # 대조 이미지의 특징점

            if descR is None or descI is None or len(descR) < 2 or len(descI) < 2:
                print("특징점이 부족합니다")
                break
            else:
                # k=2로 knnMatch : 각 특징점마다 가장 유사한 2개의 후보를 찾음
                matches = matcher.knnMatch(descR, descI, 2)
                
                # 이웃 거리의 75%로 좋은 매칭점 추출
                ratio = 0.75
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair # 1, 2위로 유사한 후보를 m, n에 배치
                        if m.distance < n.distance * ratio:
                            good_matches.append(m)
                
                print('good matches:%d/%d' % (len(good_matches), len(matches)))
                cv2.destroyWindow('searching...')
                # 좋은 매칭점 최소 갯수 이상인 경우
                if len(good_matches) > MIN_MATCH: 
                    cv2.imshow(path[12:], img)
                    result.append(path[12:])
            end_switch = True

    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:    # Esc, 종료
        break          
    elif key == ord(' '):  # 스페이스바를 누르면 ROI를 설정
        x, y, w, h = cv2.selectROI(win_name, frame, False)
        if w and h:
            roi = frame[y:y+h, x:x+w]
            print("ROI 선택됨: (%d, %d, %d, %d)" % (x, y, w, h))
            cv2.imshow('query', roi)
            cv2.destroyWindow(win_name)

cap.release()                          
cv2.destroyAllWindows()
