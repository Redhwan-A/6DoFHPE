import os
import math
from math import cos, sin

import numpy as np
import torch
#from torch.utils.serialization import load_lua
import scipy.io as sio
import cv2
from scipy.spatial.transform import Rotation

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

    return img


def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, tdx, tdy]
    pose_params = pre_pose_params[:5]
    return pose_params

def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params

def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d

# batch*n
def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    gpu = v_mag.get_device()
    if gpu < 0:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cpu'))
    else:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cuda:%d' % gpu))
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v
    
# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1) #batch*3
        
    return out
        
#in a batch*5, axis int
def stereographic_unproject(a, axis=None):
    """
	Inverse of stereographic projection: increases dimension by one.
	"""
    batch=a.shape[0]
    if axis is None:
        axis = a.shape[1]
    s2 = torch.pow(a,2).sum(1) #batch
    ans = torch.autograd.Variable(torch.zeros(batch, a.shape[1]+1).cuda()) #batch*6
    unproj = 2*a/(s2+1).view(batch,1).repeat(1,a.shape[1]) #batch*5
    if(axis>0):
        ans[:,:axis] = unproj[:,:axis] #batch*(axis-0)
    ans[:,axis] = (s2-1)/(s2+1) #batch
    ans[:,axis+1:] = unproj[:,axis:]	 #batch*(5-axis)		# Note that this is a no-op if the default option (last axis) is used
    return ans

#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:,0:3] #batch*3
    y_raw = poses[:,3:6] #batch*3

    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z) #batch*3
    y = cross_product(z,x) #batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

# a batch*5
# out batch*3*3
def compute_rotation_matrix_from_ortho5d(a):
    batch = a.shape[0]
    proj_scale_np = np.array([np.sqrt(2) + 1, np.sqrt(2) + 1, np.sqrt(2)])  # 3
    proj_scale = torch.autograd.Variable(torch.FloatTensor(proj_scale_np).cuda()).view(1, 3).repeat(batch, 1)  # batch,3

    u = stereographic_unproject(a[:, 2:5] * proj_scale, axis=0)  # batch*4
    norm = torch.sqrt(torch.pow(u[:, 1:], 2).sum(1))  # batch
    u = u / norm.view(batch, 1).repeat(1, u.shape[1])  # batch*4
    b = torch.cat((a[:, 0:2], u), 1)  # batch*6
    matrix = compute_rotation_matrix_from_ortho6d(b)
    return matrix




#input batch*4*4 or batch*3*3
#output torch batch*3 x, y, z in radiant
#the rotation is in the sequence of x,y,z
def compute_euler_angles_from_rotation_matrices(rotation_matrices, full_range=False):
    batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = torch.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
    singular = sy<1e-6
    singular = singular.float()

    '''2023.01.15'''
    for i in range(len(sy)):  # expand y (yaw angle) range into (-180, 180)
        if R[i, 0, 0] < 0 and full_range:
            sy[i] = -sy[i]

    x = torch.atan2(R[:,2,1], R[:,2,2])
    y = torch.atan2(-R[:,2,0], sy)
    z = torch.atan2(R[:,1,0],R[:,0,0])

    xs = torch.atan2(-R[:,1,2], R[:,1,1])
    ys = torch.atan2(-R[:,2,0], sy)
    zs = R[:,1,0]*0

    gpu = rotation_matrices.get_device()
    if gpu < 0:
        out_euler = torch.autograd.Variable(torch.zeros(batch,3)).to(torch.device('cpu'))
    else:
        out_euler = torch.autograd.Variable(torch.zeros(batch,3)).to(torch.device('cuda:%d' % gpu))
    out_euler[:,0] = x*(1-singular)+xs*singular
    out_euler[:,1] = y*(1-singular)+ys*singular
    out_euler[:,2] = z*(1-singular)+zs*singular
    # print('out_euler', out_euler)

    return out_euler


def get_R(x,y,z):
    ''' Get rotation matrix from three rotation angles (radians). right-handed.
    Args:
        angles: [3,]. x, y, z angles
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    """
       Get rotation matrix from three rotation angles (radians). right-handed.

       Args:
           x: Tensor of shape (batch_size,). X-axis rotation angles (in radians).
           y: Tensor of shape (batch_size,). Y-axis rotation angles (in radians).
           z: Tensor of shape (batch_size,). Z-axis rotation angles (in radians).

       Returns:
           R: Tensor of shape (batch_size, 3, 3). Rotation matrices.
       """
    # Create rotation objects from Euler angles
    r = Rotation.from_euler('xyz', np.stack([x, y, z], axis=-1), degrees=False)

    # Convert rotation objects to rotation matrices
    R = torch.tensor(r.as_matrix(), dtype=torch.float32)

    return R

