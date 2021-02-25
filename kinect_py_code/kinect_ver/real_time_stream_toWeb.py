'''
import cv2
import time

import numpy as np
from wrapperPyKinect2.acquisitionKinect import AcquisitionKinect
from wrapperPyKinect2.frame import Frame as Frame
Kinect = AcquisitionKinect()
frame = Frame()

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('saved.avi', fourcc, 25.0, (640, 480))
t_end = time.time() + 60 / 6
cnt = False;
global count

while True:
    global count
    count = 0
    Kinect.get_frame(frame)
    Kinect.get_color_frame()

    image_np = Kinect._kinect.get_last_color_frame()
    image_np = np.reshape(image_np,
                          (Kinect._kinect.color_frame_desc.Height, Kinect._kinect.color_frame_desc.Width, 4))
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    image_np = cv2.resize(image_np, (640,480))

    cv2.imshow('video_Save', image_np)  # 컬러 화면 출력
    if count == 0:
        video_frame = cv2.flip(image_np, 1)
        out.write(video_frame)
        if time.time() >= t_end:
            break

   # print('count: ',count)

    if cv2.waitKey(1) == ord('q'):
        break


out.release()
cv2.destroyAllWindows()
'''


import numpy as np
import camera
#from wrapperPyKinect2.acquisitionKinect import AcquisitionKinect
#from wrapperPyKinect2.frame import Frame as Frame
#Kinect = AcquisitionKinect()
#frame = Frame()
import cv2
from flask import Flask, render_template, Response
import time
app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    #return render_template('index.php')
    return render_template('/index.html')


def gen():
    '''
    while True:
        Kinect.get_frame(frame)
        Kinect.get_color_frame()

        image_np = Kinect._kinect.get_last_color_frame()
        image_np = np.reshape(image_np,
                              (Kinect._kinect.color_frame_desc.Height, Kinect._kinect.color_frame_desc.Width, 4))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        img = cv2.resize(image_np, (0, 0), fx=0.5, fy=0.5)
        frames = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frames + b'\r\n')
        time.sleep(0.1)
'''
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frames = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frames + b'\r\n')
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
