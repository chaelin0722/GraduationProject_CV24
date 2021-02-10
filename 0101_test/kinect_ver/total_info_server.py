import socket
import numpy
import cv2
from socket import *
##
s = [b'\xff' * 61440 for x in range(30)]

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 25.0, (640, 480))

###
UDP_IP = "192.168.0.42"


UDP_PORT2 = 9090
addr2 = UDP_IP, UDP_PORT2
sock2 = socket(AF_INET, SOCK_DGRAM)
sock2.bind(('', UDP_PORT2))
# sock2.bind((UDP_IP2, UDP_PORT2))

test, addr2 = sock2.recvfrom(1024)
# 받은 메시지와 클라이언트 주소 화면에 출력
print('server received %r from %r' % (test, addr2))

# 받은 메시지를 클라이언트로 다시 전송
sock2.sendto(test, addr2)

'''
while True:
    UDP_PORT = 9091
    addr = UDP_IP, UDP_PORT
    sock = socket(AF_INET, SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    picture = b''

    data, addr = sock.recvfrom(61441)
    s[data[0]] = data[1:61441]

    if data[0] == 29:
        for i in range(30):
            picture += s[i]
        frame = numpy.fromstring(picture, dtype=numpy.uint8)
        frame = frame.reshape(480, 640, 3)
        cv2.imshow("frame", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            
            break
'''
''' 
ERROR!
 DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead
  frame = numpy.fromstring(picture, dtype=numpy.uint8)
  
  
  
'''