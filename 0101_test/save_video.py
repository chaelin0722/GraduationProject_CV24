#save webcam video and capture last frame

import cv2
cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('save.avi', fourcc, 25.0, (640, 480))

while(True):
    ret, frame = cap.read()    # Read 결과와 frame

    if not ret :
        print("no video available")
        break

    cv2.imshow('video_Save', frame)    # 컬러 화면 출력
    #프레임 좌우반전
    frame = cv2.flip(frame, 1)
    #프레임 저장
    out.write(frame)
    # q (화면종료) 한 순간의 frame이 캡쳐되어 저장된다.
    #capture last frame
    cv2.imwrite('video_Save.png', frame, params=[cv2.IMWRITE_PNG_COMPRESSION,0])

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()