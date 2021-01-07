import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import warnings
#warnings.filterwarnings('ignore',category=FutureWarning)
import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
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
##
BUFSIZE = 1024
host = '127.0.0.1'
port = 1111
addr = host, port
cap = cv2.VideoCapture(0)
global count
count = 1
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile
print("[INFO] TF verion = ",tf.__version__)
###############
## DEPTH ##
import pygame
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
def load_model(model_name):
    model_dir = 'C:/Users/IVPL-D14/models/research/object_detection/local_models/'+model_name
    model_dir = pathlib.Path(model_dir)/"saved_model"
    print('[INFO] Loading the model from '+ str(model_dir))
    model = tf.saved_model.load(str(model_dir))
    return model
PATH_TO_LABELS = 'C:/Users/IVPL-D14/models/research/object_detection/local_models/knife_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)#, use_display_name=True)
model_name = 'trained_model_large_original_15000'
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
def run_inference(model, cap):
    fn = 0
    while True:
        Kinect.get_frame(frame)
        Kinect.get_color_frame()
        image_np = Kinect._kinect.get_last_color_frame()
        # image_np = Kinect._frameRGB
        image_depth = Kinect._frameDepthQuantized
        Skeleton_img = Kinect._frameSkeleton
        image_np = np.reshape(image_np,
                              (Kinect._kinect.color_frame_desc.Height, Kinect._kinect.color_frame_desc.Width, 4))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        show_img = image_np
        # show_img = image_np[ 200:1020, 350:1780]
        show_img = cv2.resize(show_img, (512, 424))
        # image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        # image_np = image_np[ 200:1020, 350:1780]
        # image_np = cv2.resize(image_np, (512,424))
        rgb_small_frame = cv2.resize(image_np, (0, 0), fx=0.25, fy=0.25)
        # Actual detection.
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


        ## using coordinates to get centroid of bounding box
        coordinates = vis_util.return_coordinates(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.5)

        if coordinates is not None:
            for i in range(len(coordinates)):
                for j in range(4):
                    print("each coords : ", coordinates[i][j])

        ## define
        xCenter = 0
        yCenter = 0
        if coordinates is not None:
            for i in range(len(coordinates)):
                xCenter = (coordinates[i][0] + coordinates[i][2]) / 2
                yCenter = (coordinates[i][1] + coordinates[i][3]) / 2
                #### depth of bounding box
                x = int(xCenter)
                y = int(yCenter)
                _depth = Kinect._kinect.get_last_depth_frame()
                z = int(_depth[y * 512 + x])
                ## check and compare with body depth
                print("bounding box depth : ", z)

        cv2.circle(image_np, (int(xCenter), int(yCenter)), 10, (255, 0, 0), -1)

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
                    z = int(_depth[y * 512 + x])
                    array_x.append(x)
                    array_y.append(y)
                    array_z.append(z)  # array의 필요성..?
                    print("depth spine : ", x)
        cv2.imshow('object_detection', cv2.resize(image_np, (800, 580)))
##################3
        '''
        global count
        if output_dict['detection_scores'][0] >= 0.8 and count == 1:
            s = socket(AF_INET, SOCK_DGRAM)
            line = "emergency occured!"
            s.sendto(line.encode(), addr)
            data, fromaddr = s.recvfrom(BUFSIZE)
            print('client received %r from %r' % (data, fromaddr))
            count += 1
        # 다시 clinet로부터 ok 데이터 받으면 count를 1로 되돌려 놓고 다시 신호를 줄 준비를 하자
        '''
############33
        fn = fn + 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
## 실행!
run_inference(detection_model, cap)