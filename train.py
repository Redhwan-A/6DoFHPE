# python3 train.py --dataset AFLW2000 --data_dir /home/redhwan/2/data/AFLW2000 --filename_list /home/redhwan/2/data/AFLW2000/files.txt  --output_string AFLW2000    # fast
# python3 train.py --dataset Pose_300W_LP --data_dir /home/redhwan/2/data/300W_LP --filename_list /home/redhwan/2/data/300W_LP/files.txt  --output_string 300W_LP   #20hours
# python3 train.py --lr 0.00001 --dataset CMU --data_dir /home/redhwan/2/data/CMU/train/   --filename_list /home/redhwan/2/data/CMU/files_train.txt --num_epochs 120 --batch_size 40 --gpu 0

import time
import datetime
import math
import re
import sys
import os
import argparse
import csv
import numpy as np
from numpy.lib.function_base import _quantile_unchecked
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.utils import model_zoo
import torchvision
from torchvision import transforms
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image

matplotlib.use('TkAgg')

from model import RepNet6D, RepNet5D
import utils
import datasets
from loss import GeodesicLoss


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the RepNet6D.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--num_epochs', dest='num_epochs',
        help='Maximum number of training epochs.',
        default=800, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        # default=80,
        default=16,
        type=int)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.0001, type=float)
    parser.add_argument('--scheduler',
                        # default=False,
                        default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument(
        '--dataset', dest='dataset', help='Dataset type.',
        default='Pose_300W_LP', type=str)  # Pose_300W_LP
    parser.add_argument(
        '--data_dir', dest='data_dir', help='Directory path for data.',
        default='datasets/300W_LP', type=str)  # BIWI_70_30_train.npz
    parser.add_argument(
        '--filename_list', dest='filename_list',
        help='Path to text file containing relative paths for every example.',
        default='datasets/300W_LP/files.txt', type=str)  # BIWI_70_30_train.npz #300W_LP/files.txt
    parser.add_argument(
        '--output_string', dest='output_string',
        help='String appended to output snapshots.', default='', type=str)
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
                        default=0.9995, type=float)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)

    args = parser.parse_args()
    return args


def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


if __name__ == '__main__':

    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
    b_scheduler = args.scheduler
    dataset_name = args.dataset
    alpha = args.alpha
    # =====================learn_info tar ==================
    datetime_ = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    time_ = time.time()
    print("datetime_", datetime_, "time_", time_)

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    summary_name = '{}_{}_bs{}'.format(
        dataset_name, datetime_, args.batch_size)

    if not os.path.exists('output/snapshots/{}'.format(summary_name)):
        os.makedirs('output/snapshots/{}'.format(summary_name))
    # =====================learn_info txt==================
    if not os.path.exists('output/learn_info'):
        os.makedirs('output/learn_info')

    name_txt = '{}_{}'.format(
        dataset_name, datetime_)

    if not os.path.exists('output/learn_info/{}'.format(name_txt)):
        os.makedirs('output/learn_info/{}'.format(name_txt))

    model = RepNet6D(backbone_name='RepVGG-B1g4',
                       backbone_file='RepVGG-B1g4-train.pth',
                       deploy=False,
                       pretrained=True)


    if not args.snapshot == '':
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict['model_state_dict'])

    print('Loading data.')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    transformations = transforms.Compose([transforms.RandomResizedCrop(size=224, scale=(0.8, 1)),
                                          transforms.ToTensor(),
                                          normalize])

    pose_dataset = datasets.getDataset(
        args.dataset, args.data_dir, args.filename_list, transformations)
    print('pose_dataset_____________', pose_dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    model.cuda(gpu)
    reg_criterion = GeodesicLoss().cuda(gpu)
    crit = torch.nn.MSELoss().cuda(gpu)
    softmax = nn.Softmax().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    print('optimizer', optimizer)

    if not args.snapshot == '':
        optimizer.load_state_dict(saved_state_dict['optimizer_state_dict'])

    milestones = np.arange(num_epochs)
    milestones = [10, 20]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5)


    print('Starting training.')
    outfile = open('output/learn_info/' + name_txt + '/' + args.output_string + '.txt', "a+")
    for epoch in range(num_epochs):
        loss_sum = .0
        iter = 0

        for i, (images, gt_mat, _, _) in enumerate(train_loader):
            iter += 1
            images = torch.Tensor(images).cuda(gpu)
            # Forward pass
            pred_mat = model(images)
            # Calc loss
            loss = crit(gt_mat.cuda(gpu), pred_mat)
            loss_reg = reg_criterion(gt_mat.cuda(gpu), pred_mat)
            loss = loss_reg * (alpha) + (1 - alpha) * loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

            if (i + 1) % int(len(train_loader) // 1) == 0:  # if (i+1) % 100 == 0:
                a = ('Epoch [%d/%d], Iter [%d/%d] Loss: '
                     '%.6f' % (
                         epoch + 1,
                         num_epochs,
                         i + 1,
                         len(pose_dataset) // batch_size,
                         loss.item(),
                     )
                     )
                print(a)

                outfile.write('\n')
                outfile.write(a)

        if b_scheduler:
            scheduler.step()
            print(f'epoch {epoch + 1}', "   lr  {:.7f}".format(scheduler.get_last_lr()[0]))

        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print('Taking snapshot...',
                  torch.save({
                      'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                  }, 'output/snapshots/' + summary_name + '/' + args.output_string +
                     '_epoch_' + str(epoch + 1) + '.tar')
                  )

    outfile.close()
