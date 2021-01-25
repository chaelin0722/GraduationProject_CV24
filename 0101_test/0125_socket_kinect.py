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
import ctypes
import _ctypes
import sys
#import face_recognition
import os
from scipy import io
import math
from gtts import gTTS
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from wrapperPyKinect2.acquisitionKinect import AcquisitionKinect
from wrapperPyKinect2.frame import Frame as Frame
Kinect = AcquisitionKinect()
frame = Frame()
### 소켓 통신 부분
import sys
from socket import *
BUFSIZE = 1024
host = '192.168.0.42'
port = 9090
addr = host, port
global count
count = 1

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile
print("[INFO] TF verion = ",tf.__version__)

##################
##location##
import os
import requests
from dotenv import load_dotenv
load_dotenv(verbose=True)

LOCATION_API_KEY = os.getenv('LOCATION_API_KEY')
url = f'https://www.googleapis.com/geolocation/v1/geolocate?key=AIzaSyBBI9hvhxn9BSa3Zb4dl3OMlBWmivQyNsU'
data = {
    'considerIp': True,
}
### date time ###
import datetime

###############
## DEPTH ##
## define
global xCenter
global yCenter
xCenter = 0
yCenter = 0
global body_z
global weapon_z
body_z = 0
weapon_z = 0
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
array_x =[]
array_y =[]
array_z =[]
##########

## 모델 가져오기
def load_model(model_name):
    #model_dir = 'C:/Users/IVPL-D14/models/research/object_detection/local_models/'+model_name
    model_dir = 'C:/Users/IVPL-D14/FineTunedModels/'+model_name
    model_dir = pathlib.Path(model_dir)/"saved_model"
    print('[INFO] Loading the model from '+ str(model_dir))
    model = tf.saved_model.load(str(model_dir))
    return model


#PATH_TO_LABELS = 'C:/Users/IVPL-D14/models/research/object_detection/local_models/knife_label_map.pbtxt'
PATH_TO_LABELS = 'C:/Users/IVPL-D14/models/research/object_detection/training/labelmap.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)#, use_display_name=True)
#model_name = 'trained_model_large_original_15000'

#model_name = '368_batch8_eff0_finetuned_model'          # 나름 괜찮음
#model_name = '368_batch8_num39500_eff0_finetuned_model'  # 특정구간에서만 됨.. 좀 별로
#model_name = '511_batch8_eff0_finetuned_model'            # 잡히는데 멀어지면 인식이 힘듬
#model_name = '405_batch4_num30000_eff1_fintuned_model'    # 거의 안잡힘
#model_name = '405_batch4_num40000_eff1_error_fintuned_model'  # 거의 안잡힘


#model_name = '511_batch8_finetuned_model'
print('[INFO] Downloading model and loading to network : '+ model_name)
detection_model = load_model(model_name)
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
        image_np = cv2.resize(image_np, (800, 580))


        print("[INFO]" + str(fn) + "-th frame -- Running the inference and showing the result....!!!")
        output_dict = run_inference_for_single_image(model, image_np)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
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

        print('box: ',final_box)

        if final_box is not None:
            for i in range(len(final_box)):
                xCenter = (final_box[i][1] + final_box[i][0]) / 2
                yCenter = (final_box[i][3] + final_box[i][2]) / 2
                #### depth of bounding box
                x = int(xCenter)
                y = int(yCenter)
                #check
                #print('x, y = ',x,y)

                if x < 300 and y < 300:
                    _depth = Kinect._kinect.get_last_depth_frame()
                    weapon_z = int(_depth[y * 512 + x])
                    ## check and compare with body depth
                    print("bounding box depth : ", weapon_z)

                cv2.circle(image_np, (x,y), 10, (255, 0, 0), -1)
                ## draw for just check!!
                #cv2.line(image_np, (int(final_box[i][0]), int(final_box[i][3])),
                #         (int(final_box[i][1]), int(final_box[i][3])), (255, 0, 0), 5, 4)

                #cv2.line(image_np, (int(final_box[i][0]), int(final_box[i][2])),
                #         (int(final_box[i][1]), int(final_box[i][2])), (255, 0, 0), 5, 4)

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
                    #print("depth spine : ", x)
                    #print('x, y :', x,y)
                    print("depth spine : ", body_z)

        cv2.imshow('object_detection', image_np)
##################3
        ## algorithm

        global count
        diff = body_Z - weapon_z
        ## check
        print('diff', diff)
        if diff < 10 and diff > -10
            ## test
            s = socket(AF_INET, SOCK_DGRAM)
            ## date, time, situation_description, videostream, loaction
            result = requests.post(url, data)
            data = result.json()
            lat = data['location']['lat']
            long = data['location']['lng']
            now = datetime.datetime.now()
            nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
            #print(nowDatetime)  # 2015-04-19 12:11:32

            total_info = {
                "addr": {
                    "lat": lat,
                    "long": long
                },
                "situation": "emergency occured",
                "DateTime": nowDatetime
            }

            ## 모든 가능한 server에 전송하도록 for 문돌리는것 고려
            s.sendto(total_info.encode(), addr)
            data, fromaddr = s.recvfrom(BUFSIZE)
            print('client received %r from %r' % (data, fromaddr))
            count += 1
        # 다시 clinet로부터 ok 데이터 받으면 count를 1로 되돌려 놓고 다시 신호를 줄 준비를 하자
            ## 그리고 fromaddr 을 제외한 나머지 아이들.. 핵신호를 주도록?  => 안드의 서버 딴에서 해야할 일인가..? 이것도 고려하도록 하자

        fn = fn + 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
## 실행!
run_inference(detection_model)