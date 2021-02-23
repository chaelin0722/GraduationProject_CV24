# Usage: udpecho -s [port]            (to start a server)
# or:    udpecho -c host [port] <file (client)

import socket
'''
import cv2
import numpy as np
from wrapperPyKinect2.acquisitionKinect import AcquisitionKinect
from wrapperPyKinect2.frame import Frame as Frame
Kinect = AcquisitionKinect()
frame = Frame()


while True:
    Kinect.get_frame(frame)
    Kinect.get_color_frame()

    image_np = Kinect._kinect.get_last_color_frame()
    image_np = np.reshape(image_np,
                          (Kinect._kinect.color_frame_desc.Height, Kinect._kinect.color_frame_desc.Width, 4))
    #image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    #image_np = cv2.resize(image_np, (800, 580))

    cv2.imshow('object_detection', image_np)


    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

cv2.destroyAllWindows()

'''
import sys
from socket import *
ECHO_PORT = 9090
BUFSIZE = 1024

def main():
    if len(sys.argv) < 2:
        usage()
    if sys.argv[1] == '-s':
        server()
    elif sys.argv[1] == '-c':
        client()
    else:
        usage()

def usage():
    sys.stdout = sys.stderr
    print('Usage: udpecho -s [port]            (server)')
    print('or:    udpecho -c host [port] <file (client)')
    sys.exit(2)

def server():
    # 매개변수가 2개 초과이면  두번째 매개변수를 포트로 지정한다.
    if len(sys.argv) > 2:
        port = eval(sys.argv[2])
    # 매개변수가 2개이면 기본포트로 설정한다.
    else:
        port = ECHO_PORT

    #소켓 생성
    s = socket(AF_INET, SOCK_DGRAM)
    #포트 설정
    s.bind(('', port))
    print('udp echo server ready')

    # 무한루프
    while 1:
        #클라이언트로 메세지 도착하면 다음줄로 넘어가고, 그렇지 않다면 대기
        data, addr = s.recvfrom(BUFSIZE)
        #받은 메세지와 클라이언트 주소 출력
        print('server received %r from %r' % (data, addr))
        # 받은 메세지를 클라이언트로 다시 전송
        s.sendto(data, addr)

def client():
    if len(sys.argv) < 3:
        usage()

    # 두번째 매개변수를 서버 IP로 설정
    host = sys.argv[2]

    if len(sys.argv) > 3:
        port = eval(sys.argv[3])
    else:
        port = ECHO_PORT
    addr = host, port

    s = socket(AF_INET, SOCK_DGRAM)
    s.bind(('', 0))

    print('udp echo client ready, reading stdin')

    while 1:
        line = sys.stdin.readline()
        if not line:
            break

        # 입력받은 텍스트 서버로 발송
        s.sendto(line.encode(), addr)
        #s.sendto(line, addr)
        # 리턴 대기
        data, fromaddr = s.recvfrom(BUFSIZE)
        print('client received %r from %r' % (data, fromaddr))

main()