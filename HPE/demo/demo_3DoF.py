import time
import math
import re
import sys
import os
import argparse
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())
import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
from model import RepNet6D, RepNet5D
import utils
import cv2
from hdssd import Head_detection
import tensorflow as tf
counter = 0
count_max = 200000
# video writer setup
global fourcc
global out
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # Define the codec and create VideoWriter object
out = cv2.VideoWriter('video_save/video.avi', fourcc, 20.0, (1280, 720))

Head_detection = Head_detection('SSD_models/Head_detection_300x300.pb')
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument('--gpu',
                        dest='gpu_id', help='GPU device id to use [0], set -1 to use CPU',
                        default=0, type=int)
    parser.add_argument('--cam',
                        dest='cam_id', help='Camera device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--snapshot',
                        dest='snapshot', help='Name of model snapshot.',
                        default='/home/redhwan/catkin_ws/src/HPE/snapshot/cmu.pth',
                        # default=' ',
                        type=str)
    parser.add_argument('--save_viz',
                        dest='save_viz', help='Save images with pose cube.',
                        default=False, type=bool)

    args = parser.parse_args()
    return args


transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = args.gpu_id
    if (gpu < 0):
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % gpu)
    # cam = args.cam_id
    snapshot_path = args.snapshot
    model = RepNet6D(backbone_name='RepVGG-B1g4',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)

    print('Loading data.')

    print('model', model)

    # Load snapshot
    saved_state_dict = torch.load(os.path.join(snapshot_path), map_location=None if torch.cuda.is_available() else 'cpu')

    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)
    model.to(device)
    model.eval()
    video_path = "/home/redhwan/Desktop/video/mt_hpe.mp4"
    cap = cv2.VideoCapture(video_path)


    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            (h, w, c) = frame.shape
            frame, heads = Head_detection.run(frame, w, h)
            for dict in heads:
                x_min = int(dict['left'])
                y_min = int(dict['top'])
                x_max = int(dict['right'])
                y_max = int(dict['bottom'])
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0, x_min-int(0.2*bbox_height))
                y_min = max(0, y_min-int(0.2*bbox_width))
                x_max = x_max+int(0.2*bbox_height)
                y_max = y_max+int(0.2*bbox_width)

                img = frame[y_min:y_max, x_min:x_max]
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img = transformations(img)
                img = torch.Tensor(img[None, :]).to(device)
                c = cv2.waitKey(1)
                if c == 27:
                    break

                start = time.time()
                R_pred = model(img)
                end = time.time()
                print('Head pose estimation: %2f ms' % ((end - start)*1000.))
                euler = utils.compute_euler_angles_from_rotation_matrices(
                    R_pred)*180/np.pi
                p_pred_deg = euler[:, 0].cpu()
                y_pred_deg = euler[:, 1].cpu()
                r_pred_deg = euler[:, 2].cpu()

                utils.draw_axis(frame, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], x_min + int(.5*(
                    x_max-x_min)), y_min + int(.5*(y_max-y_min)), size=bbox_width)
            if counter < count_max:
                out.write(frame)
            if counter == count_max:
                out.release()
            cv2.imshow("Demo", frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

