import cv2
import requests
import numpy as np

stream_url = "http://www.utic.go.kr/view/map/openDataCctvStream.jsp?key=eTILQrIdNdLu9vMvk9WM197IGYAohx88nIlLcuBxfLBfPc15S1rZ1rqKrDXC9G5&cctvid=E910061&cctvName=%255B%25EA%25B2%25BD%25EB%25B6%2580%25EC%2584%25A0%255D%25EA%25B8%2588%25ED%2586%25A0%25EB%25B6%2584%25EA%25B8%25B0%25EC%25A0%25902&kind=Z3&cctvip=null&cctvch=null&id=270/Wk9bhNIYaaImVmLPw9SJNnLqI2R2HCaGLGdpBdukha+jWESrmeO58edKvHpuILhus5+k4ChKnFgt/LKZXIczTQbZmyFvn9WSAU0lrAliCcw=&cctvpasswd=null&cctvport=null"  # CCTV 스트림 URL
stream = requests.get(stream_url, stream=True)  # 스트림에 요청을 보냅니다.

if stream.status_code == 200:
    video = cv2.VideoCapture(stream_url)  # VideoCapture 객체를 생성합니다.
    
    while True:
        ret, frame = video.read()  # 프레임을 읽습니다.
        if not ret:
            break  # 프레임 읽기를 실패하면 종료합니다.

        cv2.imshow("CCTV Stream", frame)  # 프레임을 화면에 표시합니다.

        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  # q 키를 누르면 종료합니다.

    video.release()  # VideoCapture 객체를 해제합니다.

cv2.destroyAllWindows()  # 모든 OpenCV 창을 닫습니다.