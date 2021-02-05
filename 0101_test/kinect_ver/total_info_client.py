import socket
from socket import *
import cv2
import numpy as np
#from wrapperPyKinect2.acquisitionKinect import AcquisitionKinect
#from wrapperPyKinect2.frame import Frame as Frame
import json
import os
import requests
import time
from dotenv import load_dotenv
load_dotenv(verbose=True)

#Kinect = AcquisitionKinect()
#frame = Frame()

LOCATION_API_KEY = os.getenv('LOCATION_API_KEY')
url = f'https://www.googleapis.com/geolocation/v1/geolocate?key=AIzaSyBBI9hvhxn9BSa3Zb4dl3OMlBWmivQyNsU'
data = {
    'considerIp': True,
}
### date time ###
import datetime
#########socket

## date, time, situation_description, videostream, loaction
count = 0

result = requests.post(url, data)
data = result.json()
lat = data['location']['lat']
long = data['location']['lng']
now = datetime.datetime.now()
nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
# print(nowDatetime)  # 2015-04-19 12:11:32

while count < 1:
    count += 1
    total_info = {
        "addr": {
            "lat": lat,
            "long": long
        },
        "situation": "emergency occured",
        "DateTime": nowDatetime,
    }
    UDP_IP2 = "192.168.0.44" #0.59
    UDP_PORT2 = 9090
    addr2 = UDP_IP2, UDP_PORT2
    s2 = socket(AF_INET, SOCK_DGRAM)

    s2.sendto(json.dumps(total_info).encode(), addr2)
    data2, fromaddr2 = s2.recvfrom(1024)
    print('client received %r from %r' % (data2, fromaddr2))

    print("sleep 5secs")
    time.sleep(5)
    print("done sleepling")
'''
while True:
    ###############
    Kinect.get_frame(frame)
    Kinect.get_color_frame()

    image_np = Kinect._kinect.get_last_color_frame()
    image_np = np.reshape(image_np,
                          (Kinect._kinect.color_frame_desc.Height, Kinect._kinect.color_frame_desc.Width, 4))
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    image_np = cv2.resize(image_np, (640,480))
    d = image_np.flatten()
    str = d.tostring()

    UDP_IP = "192.168.0.43"
    UDP_PORT = 9091
    s = socket(AF_INET, SOCK_DGRAM)
    addr = UDP_IP, UDP_PORT

    for i in range(30):
        s.sendto(bytes([i]) + str[i * 61440:(i + 1) * 61440], addr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

 '''


'''

#### test python client, emulate server
import sys
from socket import *
import datetime
import os
import requests
from dotenv import load_dotenv
import json
load_dotenv(verbose=True)
LOCATION_API_KEY = os.getenv('LOCATION_API_KEY')
url = f'https://www.googleapis.com/geolocation/v1/geolocate?key=AIzaSyBBI9hvhxn9BSa3Zb4dl3OMlBWmivQyNsU'
data = {
    'considerIp': True,
}
BUFSIZE = 1024
#host = '127.0.0.1'
#host = '192.168.18.161'  # 정빈
host = '192.168.0.44'
#서버컴
#host = '192.168.0.7'
#host='203.153.146.18'
#공기계 IP주소
#host = '192.168.11.159'
port = 9090
#host2 = []
addr = []
host_temp = ['192.168.0.44','192.168.0.59']
#for i in range(10):
#    host2.append(host + str(i))
#addr = (host,port)
s = socket(AF_INET, SOCK_DGRAM)
s.bind(('', 0))
# 준비 완료 화면에 출력
print('udp echo client ready, reading stdin')
count = 0
while count < 1:
    # 터미널 차(입력창)에서 타이핑을하고 ENTER키를 누를때 까지
    #line = sys.stdin.readline()
    # 변수에 값이 없다면
    #if not line:
    #    break
    s = socket(AF_INET, SOCK_DGRAM)
    ## date, time, situation_description, videostream, loaction
    result = requests.post(url, data)
    data = result.json()
    lat = data['location']['lat']
    long = data['location']['lng']
    # print("data : ", data)
    # print(data['location']['lat'])
    # print(data['location']['lng'])
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
    print(nowDatetime)  # 2015-04-19 12:11:32
    total_info = {
        "addr": {
            "lat": lat,
            "long": long
        },
        "situation": "emergency occured",
        "DateTime" : nowDatetime
    }
    for i in range(2):
        addr = (host_temp[i], port)
        print('####',addr)
        # 입력받은 텍스트를 서버로 발송
        s.sendto(json.dumps(total_info).encode(), addr)
        # 리턴 대기
        data, fromaddr = s.recvfrom(BUFSIZE)
        # 서버로부터 받은 메시지 출력
        print('client received %r from %r' % (data, fromaddr))
    count = count+1

'''