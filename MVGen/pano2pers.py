import argparse
import cv2
import numpy as np
import os
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(
        description='从全景图像生成多视角透视图。',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--source', 
                        type=str, required=True, help='源全景图像路径')
    parser.add_argument('--output',
                        type=str, default='scene', help='输出目录名称')
    parser.add_argument('--fov', 
                        type=float, default=60.0, help='透视图的视场角（度）')
    parser.add_argument('--output_size', 
                        type=int, nargs=2, default=[1080, 720], help='输出图像尺寸（宽 高）')
    parser.add_argument('--theta_step', 
                        type=float, default=10.0, help='theta（绕z轴旋转）的步长（度）')
    parser.add_argument('--phi_values', 
                        type=float, nargs='*', default=[0.0], help='phi（绕y轴旋转）的角度列表（度）')
    return parser.parse_args()

def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out

def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out 

class Equirectangular:
    def __init__(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        if self._img is None:
            raise FileNotFoundError(f"图像未找到: {img_name}")
        self._height, self._width, _ = self._img.shape

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        """
        获取透视图像。

        参数：
            FOV (float): 视场角，单位为度。
            THETA (float): 绕z轴的旋转角度（右旋为正，左旋为负），单位为度。
            PHI (float): 绕y轴的旋转角度（向上旋转为正，向下旋转为负），单位为度。
            height (int): 输出图像的高度。
            width (int): 输出图像的宽度。
        
        返回：
            persp (numpy.ndarray): 生成的透视图像。
        """

        f = 0.5 * width / np.tan(0.5 * np.deg2rad(FOV))
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0,  1],
            ], np.float32)
        K_inv = np.linalg.inv(K)
        
        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T

        # 创建旋转矩阵
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(x_axis * np.radians(PHI))
        R = R2 @ R1

        # 应用旋转
        xyz = xyz @ R.T
        lonlat = xyz2lonlat(xyz) 
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)
        persp = cv2.remap(self._img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return persp

def main():
    args = parse_args()

    # 加载全景图像
    equ = Equirectangular(args.source)

    # 定义输出目录
    output_folder = '../generate_mvimages'
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = os.path.join(output_folder, f"{timestamp}_{args.output}_fov{args.fov}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"output: {output_dir}")

    # 提取透视图像
    for phi in args.phi_values:
        for theta in np.arange(0, 360, args.theta_step):
            # 捕捉透视图像
            persp_img = equ.GetPerspective(args.fov, theta, phi, args.output_size[1], args.output_size[0])
            
            # 生成文件名
            output_filename = os.path.join(output_dir, f'theta_{int(theta)}_phi_{int(phi)}.png')
            
            # 保存图像
            cv2.imwrite(output_filename, persp_img)
            print(f"saved: {output_filename}")

    print(f"All the multi-view images: {output_dir}")

if __name__ == '__main__':
    main()
