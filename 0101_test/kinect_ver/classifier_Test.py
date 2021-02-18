import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image

import time

import numpy as np
from wrapperPyKinect2.acquisitionKinect import AcquisitionKinect
from wrapperPyKinect2.frame import Frame as Frame

from keras.preprocessing import image
Kinect = AcquisitionKinect()
frame = Frame()

#%%
model = load_model('C:/Users/IVPL-D14/models/research/object_detection/knife_bat_fcl.h5')
model.summary()

img_width, img_height = 224, 224  # Default input size for VGG16
# Instantiate convolutional base
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(img_width, img_height, 3))
while True:
    global count
    count = 0
    Kinect.get_frame(frame)
    Kinect.get_color_frame()

    image_np = Kinect._kinect.get_last_color_frame()
    image_np = np.reshape(image_np,
                          (Kinect._kinect.color_frame_desc.Height, Kinect._kinect.color_frame_desc.Width, 4))
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    #img = cv2.resize(image_np, (224, 224),  interpolation=cv2.INTER_AREA)
    img = cv2.resize(image_np, (224, 224), interpolation=cv2.INTER_AREA)

    #x = img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    #prediction = model.predict(x)
    #predicted_class = np.argmax(prediction[0]) # 예측된 클래스 0, 1, 2

    #print(prediction[0])
    #print(predicted_class)
    ##
    img_tensor = image.img_to_array(img)  # Image data encoded as integers in the 0–255 range
    img_tensor /= 255.  # Normalize to [0,1] for plt.imshow application

    # Extract features
    features = conv_base.predict(img_tensor.reshape(1, img_width, img_height, 3))

    # Make prediction
    try:
        prediction = model.predict(features)
    except:
        prediction = model.predict(features.reshape(1, 7 * 7 * 512))

    # Write prediction
    if prediction < 0.5:
        print('bat')
    else:
        print('knife')

    cv2.imshow('classifier', image_np)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# release resources
cv2.destroyAllWindows()