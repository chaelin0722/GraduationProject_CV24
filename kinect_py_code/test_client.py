# Usage: udpecho -s [port]            (to start a server)
# or:    udpecho -c host [port] <file (client)

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

host = '192.168.0.7'
#host='203.153.146.18'
#공기계 IP주소
#host = '192.168.11.159'

port = 9090
addr = host, port

s = socket(AF_INET, SOCK_DGRAM)
s.bind(('', 0))

# 준비 완료 화면에 출력
print('udp echo client ready, reading stdin')
count = 0
while count < 2:
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

    # 입력받은 텍스트를 서버로 발송
    s.sendto(json.dumps(total_info).encode(), addr)
    # 리턴 대기
    data, fromaddr = s.recvfrom(BUFSIZE)
    # 서버로부터 받은 메시지 출력
    print('client received %r from %r' % (data, fromaddr))
    count = count+1
'''
import sys
from socket import *

ECHO_PORT = 50000 + 7
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

'''
