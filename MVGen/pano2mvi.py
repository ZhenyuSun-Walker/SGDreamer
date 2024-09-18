import argparse
import cv2
import numpy as np
import os
from datetime import datetime

def crop_perspective(image, fov, angle, output_size):
    h, w = image.shape[:2]
    crop_width = int((fov / 360.0) * w)
    start_x = int((angle / 360.0) * w)
    
    if start_x + crop_width > w:
        part1 = image[:, start_x:]
        part2 = image[:, :start_x + crop_width - w]
        cropped_image = np.hstack((part1, part2))
    else:
        cropped_image = image[:, start_x:start_x + crop_width]
    
    return cv2.resize(cropped_image, output_size, interpolation=cv2.INTER_LINEAR)

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source', 
                    type=str, default=None, help='source pano path')
    parser.add_argument('--output',
                    type=str, default='scene', help='output directiory')
    parser.add_argument('--fov', 
                    type=int, help='angle range of perspective', default=90 )

    return parser.parse_args()

args = parse_args()
# 加载全景图像
image = cv2.imread(args.source)

# 定义参数
fov = args.fov
output_size = (int(image.shape[1] * (fov / 360.0)), image.shape[0])

output_folder = '../generate_mvimages'
os.makedirs(output_folder, exist_ok=True)

# 输出目录
output_dir = os.path.join(output_folder, f"{args.output}")
i = 1
while os.path.exists(output_dir):
    output_dir = os.path.join(output_folder, f'{datetime.now().strftime('--%Y%m%d-%H%M%S')}_{args.output}_{i}_fov{fov}')
    i += 1
os.makedirs(output_dir, exist_ok=True)

# 生成360张视角图像
for angle in range(0, 360, 1):
    cropped_image = crop_perspective(image, fov, angle, output_size)
    output_filename = os.path.join(output_dir, f'{angle}.png')
    cv2.imwrite(output_filename, cropped_image)







