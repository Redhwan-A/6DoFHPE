#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())
import argparse
import os
import rospy
import cv2
import numpy as np
from numpy import *
import tf #this is from ROS
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from dodo_detector_ros.msg import DetectedObject
import time
import csv
from hdssd import Head_detection
from PIL import Image as mg
from torchvision import transforms
import torch
from torch.backends import cudnn

from model import RepNet6D, RepNet5D
import utils

print(cv2.__version__)
global counter
global count_max
counter = 0
count_max = 200000
# video writer setup
global fourcc
global out
w =640
h= 480
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # Define the codec and create VideoWriter object
out = cv2.VideoWriter('video_save/hpefree_8010.avi', fourcc, 30.0, (w, h))
class Detector:
    def __init__(self):
        self._detector = Head_detection('SSD_models/Head_detection_300x300.pb')
                #  get label map and inference graph from params
        self._global_frame = rospy.get_param('~global_frame', None)
        self._tf_listener = tf.TransformListener()
        self.pubtime = rospy.Publisher('time', Float32, queue_size=1)
        # create detector
        self._bridge = CvBridge()
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.pc_callback)
        self._current_image = None
        self._current_pc = None
        self.sceneYPR = None

        """Parse input arguments."""
        self.parser = argparse.ArgumentParser(
            description='Head pose estimation using the 6DoFHPE.')
        self.parser.add_argument('--gpu',
                            dest='gpu_id', help='GPU device id to use [0], set -1 to use CPU',
                            default=0, type=int)
        self.parser.add_argument('--cam',
                            dest='cam_id', help='Camera device id to use [0]',
                            default=0, type=int)
        self.parser.add_argument('--snapshot',
                            dest='snapshot', help='Name of model snapshot.',
                            default='/home/redhwan/catkin_ws/src/HPE/snapshot/cmu.pth',
                            # default=' ',
                            type=str)
        self.parser.add_argument('--save_viz',
                            dest='save_viz', help='Save images with pose cube.',
                            default=False, type=bool)

        self._publishers = {None: (None, rospy.Publisher('~detected', PointCloud2, queue_size=10))}
        self._imagepub = rospy.Publisher('redhwan', Image, queue_size=10)
        self._tfpub = tf.TransformBroadcaster()
        rospy.loginfo('Ready to detect!')

    def image_callback(self, image):
        """Image callback"""
        self._current_image = image

    def pc_callback(self, pc):
        """Point cloud callback"""
        self._current_pc = pc


    def run(self):
        args = self.parser.parse_args()
        # print(args)
        cudnn.enabled = True
        gpu = args.gpu_id
        if (gpu < 0):
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:%d' % gpu)
        snapshot_path = args.snapshot
        global out
        global count_max
        global x_center, y_center
        global frame_counter, elapsed_time
        global startX, startY, endX, endY

        transformations = transforms.Compose([transforms.Resize(224),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])
        model = RepNet6D(backbone_name='RepVGG-B1g4',
                           backbone_file='',
                           deploy=True,
                           pretrained=False)
        print('Loading data.')

        saved_state_dict = torch.load(os.path.join(snapshot_path),
                                      map_location=None if torch.cuda.is_available() else 'cpu')

        if 'model_state_dict' in saved_state_dict:
            model.load_state_dict(saved_state_dict['model_state_dict'])
        else:
            model.load_state_dict(saved_state_dict)
        model.to(device)
        model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

        start_time = time.time()
        elapsed_time = 0
        frame_counter = 0

        while not rospy.is_shutdown():
            elapsed_time = time.time() - start_time
            frame_counter += 1
            # only run if there's an image present
            if self._current_image is not None:
                try:
                    # if the user passes a fixed frame, we'll ask for transformation
                    # vectors from the camera link to the fixed frame
                    if self._global_frame is not None:
                        (trans, _) = self._tf_listener.lookupTransform('/' + self._global_frame, '/camera_link',
                                                                       rospy.Time(0))
                    # convert image from the subscriber into an OpenCV image
                    scene = self._bridge.imgmsg_to_cv2(self._current_image, "bgr8")

                    (h, w, c) = scene.shape
                    print("It is important to recored video: w, h", w, h)
                    scene, heads = self._detector.run(scene, w, h)  # detect objects
                    #===============time===============
                    hello_str = elapsed_time
                    array = Float32(data=hello_str)
                    self.pubtime.publish(array)
                    # _________________________________________________________________________

                    my_tf_id = []
                    for obj_type_index, dict in enumerate(heads):
                        cal_time = time.time()
                        startX = int(dict['left'])
                        startY = int(dict['top'])
                        endX = int(dict['right'])
                        endY = int(dict['bottom'])
                        idx = int(dict['head_id'])


                        detected_object = DetectedObject()
                        sceneXYZ = scene.copy() # To copy head detection
                        bbox_width = abs(endX - startX)
                        bbox_height = abs(endY - startY)
                        x_min = max(0, startX - int(0.2 * bbox_height))
                        y_min = max(0, startY - int(0.2 * bbox_width))
                        x_max = endX + int(0.2 * bbox_height)
                        y_max = endY + int(0.2 * bbox_width)
                        y_center =  y_min + int(.5 * (y_max - y_min))
                        x_center = x_min + int(.5 * (x_max - x_min))
                        img = scene[y_min:y_max, x_min:x_max]
                        img = mg.fromarray(img)
                        img = img.convert('RGB')
                        img = transformations(img)
                        img = torch.Tensor(img[None, :]).to(device)
                        R_pred = model(img)

                        euler = utils.compute_euler_angles_from_rotation_matrices(
                            R_pred) * 180 / np.pi
                        p_pred_deg = euler[:, 0].cpu()
                        y_pred_deg = euler[:, 1].cpu()
                        r_pred_deg = euler[:, 2].cpu()
                        utils.draw_axis(scene, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], x_center, y_center, size=bbox_width)
                        self.sceneYPR = scene.copy() # To copy head detection and angles


                        publish_tf = False
                        if self._current_pc is None:
                            rospy.loginfo('No point cloud information available to track current object in scene')
                        else:
                            pc_list = list(pc2.read_points(self._current_pc, skip_nans=True, field_names=('x', 'y', 'z'),
                                                uvs=[(int(x_center), int(y_center))]))
                            if len(pc_list) > 0:
                                publish_tf = True
                                tf_id = ("head_id" + '_' + str(idx ))  # object_number(we added 1 to start of 1)
                                my_tf_id.append(tf_id)
                                detected_object.tf_id.data = tf_id
                                # print("detected_object", tf_id)
                                point_x, point_y, point_z = pc_list[0]  # point_z = x, point_x = y
                                conv = 100 # from m to cm
                                if point_z== point_x==point_y ==0:
                                    print('noises')
                                else:
                                    label = "{}: {:.2f}, {}: {:.2f}, {}: {:.2f} ".format('x', point_x*conv, 'y',- point_y*conv,'z', point_z*conv)
                                    # # cv2.rectangle(scene, (startX, startY), (endX, endY), [0, 0, 255], 2)
                                    y = startY - 15 if startY - 15 > 15 else startY + 15
                                    cv2.putText(scene, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
                                    cv2.putText(sceneXYZ, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)

                            if publish_tf:
                                object_tf = [point_z, -point_x, -point_y]
                                if point_z== point_x==point_y ==0:
                                    print('noises')
                                else:
                                    # assign header columns
                                    headerList = ['time','Yaw', 'Pitch', 'Roll', 'X', 'Y', 'Z']
                                    xyztosave = [cal_time, y_pred_deg[0].item(), p_pred_deg[0].item(), r_pred_deg[0].item(), point_x * conv, - point_y * conv, point_z * conv]
                                    with open('video_save/hpefree_8010.csv', "a") as f:
                                        if f.tell() == 0:
                                            dw = csv.DictWriter(f, delimiter=',',fieldnames=headerList)
                                            dw.writeheader()
                                        wr = csv.writer(f, dialect='excel')
                                        wr.writerow(xyztosave)



                                frame_link = 'camera_link'
                                if self._global_frame is not None:
                                    object_tf = np.array(trans) + object_tf
                                    frame_link = self._global_frame
                                self._tfpub.sendTransform((object_tf),
                                                          tf.transformations.quaternion_from_euler(0, 0, 0),
                                                          rospy.Time.now(), tf_id, frame_link)

                    try:
                        self._imagepub.publish(self._bridge.cv2_to_imgmsg(scene, 'bgr8'))
                    except CvBridgeError as e:
                        print(e)

                    if counter < count_max:
                        out.write(scene)

                    if counter == count_max:
                        out.release()
                        print('made video')
                    cv2.imshow("Head Pose", scene)
                    key = cv2.waitKey(1)
                            # =====================================================================================
                except CvBridgeError as e:
                    print(e)
if __name__ == '__main__':
    rospy.init_node('dodo_detector_ros', log_level=rospy.INFO)
    try:
        tr = Detector()
        tr.run()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')
