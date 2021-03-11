## import
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import pathlib
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import numpy as np
import cv2

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image

import pygame
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import time
from wrapperPyKinect2.acquisitionKinect import AcquisitionKinect
from wrapperPyKinect2.frame import Frame as Frame

from keras.preprocessing import image

Kinect = AcquisitionKinect()
frame = Frame()

global count
count = 1

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile
print("[INFO] TF verion = ", tf.__version__)

##################
###############
## DEPTH ##
## define
global xCenter
global yCenter
xCenter = 0
yCenter = 0
global body_z
global weapon_z
global diff
global r_z
global l_z
body_z = 0
weapon_z = 0
diff = 0

r_z = 0
l_z = 0
####
global color
color = (0, 255, 0)
global current_state
current_state = "safe"
## skeleton ##
###
SKELETON_COLORS = [pygame.color.THECOLORS["red"],
                   pygame.color.THECOLORS["blue"],
                   pygame.color.THECOLORS["green"],
                   pygame.color.THECOLORS["orange"],
                   pygame.color.THECOLORS["purple"],
                   pygame.color.THECOLORS["yellow"],
                   pygame.color.THECOLORS["violet"]]
array_x = []
array_y = []
array_z = []
array_rx = []
array_ry = []
array_rz = []
array_lx = []
array_ly = []
array_lz = []

### 소켓 통신 부분
from socket import *
s = socket(AF_INET, SOCK_DGRAM)
s.bind(('',0))

##location##
import os
from dotenv import load_dotenv
load_dotenv(verbose=True)

LOCATION_API_KEY = os.getenv('LOCATION_API_KEY')
url = f'https://www.googleapis.com/geolocation/v1/geolocate?key=AIzaSyBBI9hvhxn9BSa3Zb4dl3OMlBWmivQyNsU'
info = {
    'considerIp': True,
}
### date time ###
import datetime
import json
import requests


## date, time, situation_description, videostream, loaction
result = requests.post(url, info)
info = result.json()
lat = info['location']['lat']
long = info['location']['lng']
now = datetime.datetime.now()
nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
# print(nowDatetime)  # 2015-04-19 12:11:32

total_info = {
    "addr": {
        "lat": lat,
        "long": long
    },
    "DateTime": nowDatetime
}
host_temp = ['192.168.0.44','192.168.0.75','192.168.0.59']
#host_temp = ['192.168.0.44','192.168.0.59']

global cnt
cnt = 0
# web browser
from flask import Flask, render_template, Response

## gpu 제어
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

########## classifier
# classifier = load_model('C:/Users/IVPL-D14/models/research/object_detection/knife_bat_0219re.h5')
classifier = load_model('C:/Users/IVPL-D14/models/research/object_detection/knife_bat_0225.h5')
classifier.summary()

img_width, img_height = 224, 224  # Default input size for VGG16
# Instantiate convolutional base
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(img_width, img_height, 3))
###
app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('/index.html')

## 모델 가져오기
def load_model_(model_name):
    # model_dir = 'C:/Users/IVPL-D14/models/research/object_detection/local_models/'+model_name
    model_dir = 'C:/Users/IVPL-D14/FineTunedModels/' + model_name
    model_dir = pathlib.Path(model_dir) / "saved_model"
    print('[INFO] Loading the model from ' + str(model_dir))
    model = tf.saved_model.load(str(model_dir))
    return model


# PATH_TO_LABELS = 'C:/Users/IVPL-D14/models/research/object_detection/local_models/knife_label_map.pbtxt'
PATH_TO_LABELS = 'C:/Users/IVPL-D14/models/research/object_detection/training/weapon_labelmap.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)  # , use_display_name=True)


############################   model    #############################

model_name = 'training_kb_batch8_eff0_newdata_80000_fintuned_model'   # bat인식 괜찮음. 이번엔 둘다 bat
# model_name = 'training_kb_batch8_eff0_0202_arranged_fintuned_model'  # knife인식 좋음

print('[INFO] Downloading model and loading to network : ' + model_name)
detection_model = load_model_(model_name)
detection_model.signatures['serving_default'].output_dtypes
detection_model.signatures['serving_default'].output_shapes


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    return output_dict


## main 부분
def run_inference(model):
    fn = 0
    while True:
        Kinect.get_frame(frame)
        Kinect.get_color_frame()
        image_np = Kinect._kinect.get_last_color_frame()
        image_np = np.reshape(image_np,
                              (Kinect._kinect.color_frame_desc.Height, Kinect._kinect.color_frame_desc.Width, 4))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        image_obj = cv2.resize(image_np, (800, 580))
        # image_obj = cv2.resize(image_np, (640, 480))

        print("[INFO]" + str(fn) + "-th frame -- Running the inference and showing the result....!!!")
        output_dict = run_inference_for_single_image(model, image_obj)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_obj,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        ###
        global current_state
        global color
        ## get bounding box depth
        global body_z
        global weapon_z
        global diff
        global l_z
        global r_z
        boxes = np.squeeze(output_dict['detection_boxes'])
        scores = np.squeeze(output_dict['detection_scores'])
        # set a min thresh score, say 0.8
        min_score_thresh = 0.5
        bboxes = boxes[scores > min_score_thresh]
        # get image size
        im_width = 800
        im_height = 580
        final_box = []

        for box in bboxes:
            ymin, xmin, ymax, xmax = box
            final_box.append([xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height])

        print('box: ', final_box)

        if final_box is not None:
            for i in range(len(final_box)):
                xCenter = (final_box[i][1] + final_box[i][0]) / 2
                yCenter = (final_box[i][3] + final_box[i][2]) / 2
                #### depth of bounding box
                x = int(xCenter)
                y = int(yCenter)

                # bounding box depth를 구하기 위한 중심점 표시
                if x < 500 and y < 500:
                    _depth = Kinect._kinect.get_last_depth_frame()
                    weapon_z = int(_depth[y * 512 + x])
                    ## check and compare with body depth
                    print("bounding box depth : ", weapon_z)

                cv2.circle(image_obj, (x, y), 5, (255, 0, 0), -1)

                ######## classifier
                width_length = int(final_box[i][1] - final_box[i][0])
                height_length = int(final_box[i][3] - final_box[i][2])

                area = image_obj[int(y - (0.5 * height_length)): y + int(0.5 * height_length),
                       int(x - (0.5 * width_length)): x + int(0.5 * width_length)]

                #cv2.imshow('area', area)

                img = cv2.resize(area, (224, 224), interpolation=cv2.INTER_AREA)
                img_tensor = image.img_to_array(img)  # Image data encoded as integers in the 0–255 range
                img_tensor /= 255.  # Normalize to [0,1] for plt.imshow application

                # Extract features
                features = conv_base.predict(img_tensor.reshape(1, img_width, img_height, 3))

                # Make prediction
                try:
                    prediction = classifier.predict(features)
                except:
                    prediction = classifier.predict(features.reshape(1, 7 * 7 * 512))

                # Write prediction
                if prediction < 0.5:
                    wp = 'bat'
                else:
                    wp = 'knife'

                cv2.putText(image_obj, wp, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, 8)

        ### body depth detect
        if Kinect._bodies is not None:
            if Kinect._kinect.has_new_depth_frame:
                for i in range(0, Kinect._kinect.max_body_count):
                    body = Kinect._bodies.bodies[i]
                    if not body.is_tracked:
                        continue
                    joints = body.joints
                    # convert joint coordinates to color space
                    joint_points = Kinect._kinect.body_joints_to_color_space(joints)
                    Kinect.draw_body(joints, joint_points, SKELETON_COLORS[i])

                    # get the skeleton joint x y z
                    depth_points = Kinect._kinect.body_joints_to_depth_space(joints)
                    x = int(depth_points[PyKinectV2.JointType_SpineMid].x)
                    y = int(depth_points[PyKinectV2.JointType_SpineMid].y)

                    _depth = Kinect._kinect.get_last_depth_frame()
                    body_z = int(_depth[y * 512 + x])
                    array_x.append(x)
                    array_y.append(y)
                    array_z.append(body_z)  # array의 필요성..?
                    # change to this!
            dif = body_z - weapon_z
            diff = abs(dif)
            print("body_z : ", body_z)
            print("weapon_z : ", weapon_z)
            print("diff = ", diff)

            # if diff is not None:
            # if body_z != 0 and weapon_z != 0 and diff < 30:

            global cnt
            if body_z != 0 and weapon_z != 0 and diff < 800:
                current_state = "danger"  # while 안에 두자
                color = (0, 0, 255)
            else:
                current_state = "safe"  # while 안에 두자
                color = (0, 255, 0)

            if cnt is 0 and body_z != 0 and weapon_z != 0 and diff < 800:
                ## socket 쏘기!
                UDP_PORT = 9090
                # 여러개용
                cnt = cnt + 1
                for i in range(3):
                    addr = (host_temp[i], UDP_PORT)
                    s.sendto(json.dumps(total_info).encode(), addr)
                    data, fromaddr = s.recvfrom(1024)
                    print('client received %r from %r' % (data, fromaddr))

        cv2.putText(image_obj, current_state, (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, 8)

        ## web
        img = cv2.resize(image_obj, (300,220))
        frames = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frames + b'\r\n')
        time.sleep(0.1)
        cv2.imshow('object_detection', image_obj)

        fn = fn + 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(run_inference(detection_model),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='192.168.0.7', debug=True)
