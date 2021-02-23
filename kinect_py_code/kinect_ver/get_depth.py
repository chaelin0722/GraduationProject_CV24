from wrapperPyKinect2.acquisitionKinect import AcquisitionKinect
from wrapperPyKinect2.frame import Frame
import cv2
import numpy as np
from pykinect2 import PyKinectV2
import pygame
kinect = AcquisitionKinect()
frame= Frame()
#################
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
while True:
    kinect.get_frame(frame)
    kinect.get_color_frame()
    image_np = kinect._kinect.get_last_color_frame()
    image_np = np.reshape(image_np, (kinect._kinect.color_frame_desc.Height, kinect._kinect.color_frame_desc.Width,4))
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
### testing..
    #tt = Kinect._mapper.MapCameraPointToDepthSpace(joints[PyKinectV2.JointType_Head].position)
    ''' 
    if kinect._kinect.has_new_depth_frame():
        kinect._depth = kinect._kinect.get_last_depth_frame()
    if body_frame is not None:
        for i in range(0, kinect.max_body_count):
            body = body_frame.bodies[i]
            if body.is_tracked:
    ###
    '''
    if kinect._bodies is not None:
        if kinect._kinect.has_new_depth_frame:
            for i in range(0, kinect._kinect.max_body_count):
                body = kinect._bodies.bodies[i]
                if not body.is_tracked:
                    continue
                joints = body.joints
                # convert joint coordinates to color space
                joint_points = kinect._kinect.body_joints_to_color_space(joints)
                kinect.draw_body(joints, joint_points, SKELETON_COLORS[i])
                # get the skeleton joint x y z
                depth_points = kinect._kinect.body_joints_to_depth_space(joints)
                x = int(depth_points[PyKinectV2.JointType_SpineMid].x)
                y = int(depth_points[PyKinectV2.JointType_SpineMid].y)
                _depth = kinect._kinect.get_last_depth_frame()
                z = int(_depth[y * 512 + x])
                array_x.append(x)
                array_y.append(y)
                array_z.append(z)  # array의 필요성..?
                print("depth spine : ", z)

    #print("elbow_left_depth : ", array_z)
    cv2.imshow("w",cv2.resize(image_np,(800,580)))
  #  cv2.imshow("skeleton",cv2.resize(image_np,(800,580)))
    if cv2.waitKey(25) & 0xFF==ord('q'):
        cv2.destroyAllWindows()
        break