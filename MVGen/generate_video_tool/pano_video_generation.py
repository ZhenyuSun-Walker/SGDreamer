# import cv2
# import numpy as np
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# import lib.Equirec2Perspec as E2P
# import lib.multi_Perspec2Equirec as m_P2E
# import uuid
# from PIL import Image
# from tqdm import tqdm


# def generate_video(image_paths, out_dir,  FOV, deg, gen_video=True, save_frames=True):
#     pers = [cv2.imread(image_path) for image_path in image_paths]

#     # General version for variation
#     pers_matrix = [[FOV, deg * i, 0] for i in range(int(360 / deg))]

#     print(pers_matrix)

#     ee = m_P2E.Perspective(pers, pers_matrix)

#     new_pano = ee.GetEquirec(2048, 4096)
#     cv2.imwrite(os.path.join(out_dir, 'pano.png'), new_pano.astype(np.uint8)[540:-540])
#     if not gen_video:
#         return
#     equ = E2P.Equirectangular(new_pano)
#     fov = 90
#     video_size = (727, 1280)

#     img = equ.GetPerspective(fov, 0, 0, video_size[0], video_size[1])  # Specify parameters(FOV, theta, phi, height, width)

#     h = img.shape[0]
#     margin = 0
#     if margin > 0:
#         img = img[margin:-margin]
#     size = (img.shape[1], img.shape[0])

#     tmp_video_path = '/tmp/' + str(uuid.uuid4()) + '.mp4'
#     save_video_path = os.path.join(out_dir, 'video.mp4')
#     out = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'MP4V'), 60, size)

#     interval_deg = 0.5
#     num_frames = int(360 / interval_deg)

#     image_save_path = os.path.join(out_dir, 'images')
#     os.makedirs(image_save_path, exist_ok=True)

#     for i in range(num_frames):
#         deg = i * interval_deg
#         img = equ.GetPerspective(fov, deg, 0, video_size[0], video_size[1])  # Specify parameters(FOV, theta, phi, height, width)
#         h = img.shape[0]
#         if margin > 0:
#             img = img[margin:-margin]
#         img = np.clip(img, 0, 255).astype(np.uint8)

#         # Save each frame as an image
#         if save_frames:
#             frame_filename = os.path.join(image_save_path, f'{i:04d}.png')
#             cv2.imwrite(frame_filename, img)

#         out.write(img)
#     out.release()
#    # os.system(f"ffmpeg -y -i {tmp_video_path} -vcodec libx264 {save_video_path}")

import cv2
import numpy as np
import os
import uuid
from tqdm import tqdm


# 计算针孔相机内参矩阵
def get_pinhole_intrinsics(fov, width, height):
    focal_length = width / (2 * np.tan(np.radians(fov) / 2))
    K = np.array([[focal_length, 0, width / 2],
                  [0, focal_length, height / 2],
                  [0, 0, 1]])
    return K


# 使用针孔相机模型生成图像
def pinhole_projection(image, K, theta, phi, width, height):
    # 创建变换矩阵
    R = cv2.Rodrigues(np.array([phi, theta, 0]))[0]  # 旋转矩阵
    P = K @ np.hstack((R, np.array([[0], [0], [0]])))  # 投影矩阵

    # 对图像进行变换
    projected_img = cv2.warpPerspective(image, P, (width, height), flags=cv2.INTER_LINEAR)
    return projected_img


def generate_video(image_paths, out_dir, FOV, deg, gen_video=True, save_frames=True):
    pers = [cv2.imread(image_path) for image_path in image_paths]

    # PinHole intrinsics matrix
    video_size = (727, 1280)
    K = get_pinhole_intrinsics(FOV, video_size[1], video_size[0])

    tmp_video_path = '/tmp/' + str(uuid.uuid4()) + '.mp4'
    save_video_path = os.path.join(out_dir, 'video.mp4')
    out = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'MP4V'), 60, video_size)

    interval_deg = 0.5
    num_frames = int(360 / interval_deg)

    image_save_path = os.path.join(out_dir, 'images')
    os.makedirs(image_save_path, exist_ok=True)

    for i in range(num_frames):
        deg = i * interval_deg
        theta = np.radians(deg)  # Y轴旋转
        phi = 0  # X轴旋转可以根据需要调整

        # 逐帧生成图像
        for img in pers:
            img = pinhole_projection(img, K, theta, phi, video_size[0], video_size[1])
            img = np.clip(img, 0, 255).astype(np.uint8)

            # Save each frame as an image
            if save_frames:
                frame_filename = os.path.join(image_save_path, f'{i:04d}.png')
                cv2.imwrite(frame_filename, img)

            out.write(img)
    
    out.release()

