
from socket import *
import cv2
import numpy as np
from wrapperPyKinect2.acquisitionKinect import AcquisitionKinect
from wrapperPyKinect2.frame import Frame as Frame
Kinect = AcquisitionKinect()
frame = Frame()

#UDP_IP = '192.168.44.31'
UDP_IP = '127.0.0.1'  # 본인주소

UDP_PORT = 9090

s= socket(AF_INET, SOCK_DGRAM)



while True:
    Kinect.get_frame(frame)
    Kinect.get_color_frame()

    image_np = Kinect._kinect.get_last_color_frame()
    image_np = np.reshape(image_np,
                          (Kinect._kinect.color_frame_desc.Height, Kinect._kinect.color_frame_desc.Width, 4))
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    image_np = cv2.resize(image_np, (480,640))

    d = image_np.flatten()
    str = d.tostring()

    for i in range(20):
        s.sendto(bytes([i]) + str[i * 46080:(i + 1) * 46080], (UDP_IP, UDP_PORT))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

'''
Client의 Webcam에서 생성하는 각 프레임은 640x480 RGB 픽셀을 가지는데 이것의 실제 데이터 크기는 640 x 480 x 3 = 921,600 Byte이다. 이 때 3은 RGB 즉, 빨강, 초록,  파랑을 나타내기 위해 사용하는 것이다.

 

그런데 UDP는 한번에 데이터를 65,535 Byte 까지 보낼 수 있어 위의 921,600 Byte를 한 번에 보낼 수 없다. 그래서 데이터를 나눠서 보내야하는데 해당 코드에서는 921,600 Byte를 20으로 나눈 46,080 Byte를 보내주고 있다. 

 

그래서 46080이 코드에서 계속 나오는 것이다. 

 

그래서 만약 OpenCV에서 생성하는 프레임이 다르다면 그에 맞게 고쳐주면 된다.'''