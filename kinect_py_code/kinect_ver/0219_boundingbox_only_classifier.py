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

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import time
from wrapperPyKinect2.acquisitionKinect import AcquisitionKinect
from wrapperPyKinect2.frame import Frame as Frame

from keras.preprocessing import image

Kinect = AcquisitionKinect()
frame = Frame()
### 소켓 통신 부분
import sys
from socket import *

global count
count = 1

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile
print("[INFO] TF verion = ", tf.__version__)

##################
##location##
import os
import requests
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
body_z = 0
weapon_z = 0
diff = 0
import pygame

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
classifier = load_model('C:/Users/IVPL-D14/models/research/object_detection/knife_bat_fcl.h5')
classifier.summary()

img_width, img_height = 224, 224  # Default input size for VGG16
# Instantiate convolutional base
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(img_width, img_height, 3))


## 모델 가져오기
def load_model_(model_name):
    # model_dir = 'C:/Users/IVPL-D14/models/research/object_detection/local_models/'+model_name
    model_dir = 'C:/Users/IVPL-D14/FineTunedModels/' + model_name
    model_dir = pathlib.Path(model_dir) / "saved_model"
    print('[INFO] Loading the model from ' + str(model_dir))
    model = tf.saved_model.load(str(model_dir))
    return model


# PATH_TO_LABELS = 'C:/Users/IVPL-D14/models/research/object_detection/local_models/knife_label_map.pbtxt'
PATH_TO_LABELS = 'C:/Users/IVPL-D14/models/research/object_detection/training/labelmap.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)  # , use_display_name=True)
# model_name = 'trained_model_large_original_15000'

# model_name = '368_batch8_eff0_finetuned_model'          # 나름 괜찮음
# model_name = '368_batch8_num39500_eff0_finetuned_model'  # 특정구간에서만 됨.. 좀 별로
# model_name = '511_batch8_eff0_finetuned_model'            # 잡히는데 멀어지면 인식이 힘듬q
# model_name = '405_batch4_num30000_eff1_fintuned_model'    # 거의 안잡힘
# model_name = '405_batch4_num40000_eff1_error_fintuned_model'  # 거의 안잡힘


# model_name = 'training_kb_batch8_eff0_40000_fintuned_model_loss0.8'  # bat을 knife로
model_name = 'training_kb_batch8_eff0_newdata_80000_fintuned_model'  # bat인식 괜찮음. 이번엔 둘다 bat
# model_name = 'training_kb_batch8_eff0_newdata_wout_aug_fintuned_model'
# model_name = 'training_kb_batch16_eff0_25000_fintuned_model'
# model_name = 'training_kb_eff0_batch16_28000err_noaug'
##model_name = 'training_kb_eff1_batch8_28000err_aug'  ## 아예 에러남
################################0203####################################

##############################0204##############################
# model_name = 'training_kb_batch8_eff0_0202_arranged_fintuned_model'  # knife인식 좋음
# model_name ='training_kb_batch8_eff0_0203_arranged_fintuned_model'
# model_name = 'training_only_bat_fintuned_model'
# model_name = 'training_only_knife_fintuned_model'  위의 세 모델 인식 진짜 별로
# model_name = 'training_kb_batch8_eff0_newdata_arranged_fintuned_model'

################ kife and bat #########################
# model_name = 'nope/training_kb_batch8_eff0_40000_fintuned_model'
# model_name = 'nope/training_kb_batch12_eff0_40000_fintuned_model'  # knife 가까워야 인식 (가로로 잡히는 순간 존재)
# model_name = 'nope/training_kb_batch12_eff0_50000_fintuned_model' 위에랑 비슷
################################################################+

##############################0207#############################
# model_name = 'training_kb_0206_more_bats_fintuned_model' 80000번 시도?
# model_name = 'training_only_0206_bat_fintuned_model'
###################설연휴 모델###############
# model_name = 'training_kb_0211_re60000_more_bats_fintuned_model'  #가능성 있어보임
# model_name = 'training_kb_0210_80000_more_bats_fintuned_model'


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
    "situation": "emergency occured",
    "DateTime": nowDatetime
}

UDP_IP = "192.168.0.42"
global cnt
cnt = True

#### for video s
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('saved.avi', fourcc, 25.0, (640, 480))
t_end = time.time() + 10


def send_json(cnt):
    while cnt is True:
        UDP_PORT = 9090
        addr = UDP_IP, UDP_PORT
        s = socket(AF_INET, SOCK_DGRAM)
        s.sendto(json.dumps(total_info).encode(), addr)

        data, fromaddr = s.recvfrom(1024)
        print('client received %r from %r' % (data, fromaddr))


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

        ## get bounding box depth
        global body_z
        global weapon_z
        global diff
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
                if x < 300 and y < 300:
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

                cv2.imshow('area', area)

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
                    # change to this!
                    # PyKinectV2.JointType_HandLeft
                    # PyKinectV2.JointType_HandRight
                    _depth = Kinect._kinect.get_last_depth_frame()
                    body_z = int(_depth[y * 512 + x])
                    array_x.append(x)
                    array_y.append(y)
                    array_z.append(body_z)  # array의 필요성..?
                    print("depth spine : ", x)
                    # print('x, y :', x,y)
                    print("depth spine : ", body_z)

            diff = body_z - weapon_z
            ## check
            print('diff', diff)

            '''
            global cnt
            #if diff is not None:
            if body_z > 400 :
                if cnt is True:
                    UDP_PORT = 9090
                    addr = UDP_IP, UDP_PORT
                    s = socket(AF_INET, SOCK_DGRAM)
                    s.sendto(json.dumps(total_info).encode(), addr)
                    data, fromaddr = s.recvfrom(1024)
                    print('client received %r from %r' % (data, fromaddr))

                    # 프레임 좌우반전
                    video_frame = cv2.flip(image_obj, 1)
                    # 프레임 저장
                    out.write(video_frame)

                    if time.time() > t_end:
                        out.release()
                        cnt = False


                UDP_PORT = 9091
                s = socket(AF_INET, SOCK_DGRAM)
                addr = UDP_IP, UDP_PORT

                for i in range(30):
                    s.sendto(bytes([i]) + strr[i * 61440:(i + 1) * 61440], addr)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        '''

        cv2.imshow('object_detection', image_obj)

        ## algorithm
        '''
        # 다시 clinet로부터 ok 데이터 받으면 count를 1로 되돌려 놓고 다시 신호를 줄 준비를 하자
            ## 그리고 fromaddr 을 제외한 나머지 아이들.. 핵신호를 주도록?  => 안드의 서버 딴에서 해야할 일인가..? 이것도 고려하도록 하자
        '''
        fn = fn + 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


## 실행!
run_inference(detection_model)