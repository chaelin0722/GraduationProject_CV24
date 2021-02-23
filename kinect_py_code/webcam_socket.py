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

def load_model(model_name):
    model_dir = './local_models/'+model_name
    model_dir = pathlib.Path(model_dir)/"saved_model"
    print('[INFO] Loading the model from '+ str(model_dir))
    model = tf.saved_model.load(str(model_dir))
    return model

PATH_TO_LABELS = './data/knife_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)#, use_display_name=True)
model_name =  'trained_model_large_original_15000'
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

    while cap.isOpened():
        ret, image_np = cap.read()
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

        cv2.imshow('object_detection', cv2.resize(image_np, (800, 580)))
##################3


        global count
        if output_dict['detection_scores'][0] >= 0.8 and count == 1:
            s = socket(AF_INET, SOCK_DGRAM)
            line = "emergency occured!"
            s.sendto(line.encode(), addr)
            data, fromaddr = s.recvfrom(BUFSIZE)
            print('client received %r from %r' % (data, fromaddr))
            count += 1

# 다시 clinet로부터 ok 데이터 받으면 count를 1로 되돌려 놓고 다시 신호를 줄 준비를 하자
############33
        fn = fn + 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


## 실행!
run_inference(detection_model, cap)