import argparse
import cv2
import numpy as np
import os
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source', 
                        type=str, default=None, help='source pano path')
    parser.add_argument('--output',
                        type=str, default='scene', help='output directory')
    parser.add_argument('--fov', 
                        type=int, default=90, help='angle range of perspective')
    parser.add_argument('--output_size', 
                        type=int, nargs=2, default=[1440, 960], help='output image size (width, height)')
    return parser.parse_args()

class Equirectangular:
    def __init__(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape

    def GetPerspective(self, FOV, R, T, height, width):
        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ], np.float32)
        K_inv = np.linalg.inv(K)

        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T

        # Apply rotation and translation
        xyz = (xyz @ R.T) + T.T
        lonlat = xyz2lonlat(xyz)
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)
        persp = cv2.remap(self._img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return persp

def xyz2lonlat(xyz):
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = np.arctan2(x, z)
    lat = np.arcsin(y)
    return np.concatenate([lon, lat], axis=-1)

def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    return np.concatenate([X, Y], axis=-1)

def create_rotation_matrix(yaw, pitch=0, roll=0):
    # Convert angles to radians
    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)
    roll = np.deg2rad(roll)
    
    # Rotation matrices for yaw, pitch, and roll
    R_yaw = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    
    R_roll = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix: R = R_roll * R_pitch * R_yaw
    R = R_roll @ R_pitch @ R_yaw
    return R

# Parse arguments from the command line
args = parse_args()

# 加载全景图像
equ = Equirectangular(args.source)

# 定义输出目录
output_folder = '../generate_mvimages'
os.makedirs(output_folder, exist_ok=True)
output_dir = os.path.join(output_folder, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.output}_fov{args.fov}_pose")
os.makedirs(output_dir, exist_ok=True)

# 提取透视图像
for angle in range(0, 360, 1):
    R = create_rotation_matrix(angle)
    T = np.array([0, 0, 0])  # No translation
    perspective_img = equ.GetPerspective(args.fov, R, T, args.output_size[1], args.output_size[0])
    
    output_filename = os.path.join(output_dir, f'{angle}.png')
    cv2.imwrite(output_filename, perspective_img)
