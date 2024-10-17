import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import lib.Equirec2Perspec as E2P
import lib.multi_Perspec2Equirec as m_P2E
import uuid
from PIL import Image
from tqdm import tqdm


def generate_video(image_paths, out_dir,  FOV, deg, gen_video=False, save_frames=False):
    pers = [cv2.imread(image_path) for image_path in image_paths]

    # General version for variation
    pers_matrix = [[FOV, deg * i, 0] for i in range(int(360 / deg))]

    print(pers_matrix)

    ee = m_P2E.Perspective(pers, pers_matrix)

    new_pano = ee.GetEquirec(2048, 4096)
    cv2.imwrite(os.path.join(out_dir, 'pano.png'), new_pano.astype(np.uint8)[540:-540])
    if not gen_video:
        return
    equ = E2P.Equirectangular(new_pano)
    fov = 90
    video_size = (727, 1280)

    img = equ.GetPerspective(fov, 0, 0, video_size[0], video_size[1])  # Specify parameters(FOV, theta, phi, height, width)

    h = img.shape[0]
    margin = 0
    if margin > 0:
        img = img[margin:-margin]
    size = (img.shape[1], img.shape[0])

    tmp_video_path = '/tmp/' + str(uuid.uuid4()) + '.mp4'
    save_video_path = os.path.join(out_dir, 'video.mp4')
    out = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'MP4V'), 60, size)

    interval_deg = 0.5
    num_frames = int(360 / interval_deg)

    image_save_path = os.path.join(out_dir, 'images')
    os.makedirs(image_save_path, exist_ok=True)

    for i in range(num_frames):
        deg = i * interval_deg
        img = equ.GetPerspective(fov, deg, 0, video_size[0], video_size[1])  # Specify parameters(FOV, theta, phi, height, width)
        h = img.shape[0]
        if margin > 0:
            img = img[margin:-margin]
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Save each frame as an image
        if save_frames:
            frame_filename = os.path.join(image_save_path, f'{i:04d}.png')
            cv2.imwrite(frame_filename, img)

        out.write(img)
    out.release()
   # os.system(f"ffmpeg -y -i {tmp_video_path} -vcodec libx264 {save_video_path}")


# import cv2
# import numpy as np
# import os
# import uuid
# from tqdm import tqdm

# def get_perspective(panorama, fov, theta, phi, width, height):
#     # 计算视角范围
#     aspect_ratio = width / height
#     scale = np.tan(np.radians(fov) / 2)

#     perspective_img = np.zeros((height, width, 3), dtype=np.uint8)

#     for y in range(height):
#         for x in range(width):
#             # 计算视点在球面上的位置
#             nx = (x / width) * 2 - 1  # [-1, 1]
#             ny = 1 - (y / height) * 2  # [1, -1]

#             # 计算球面坐标
#             r = np.sqrt(nx**2 + ny**2 + 1)
#             theta_s = np.arctan2(ny, np.sqrt(nx**2 + 1))  # theta
#             phi_s = np.arctan2(nx, 1)  # phi

#             # 将球面坐标转换为全景图中的像素
#             u = (phi_s / (2 * np.pi) + 0.5) * panorama.shape[1]
#             v = (0.5 - theta_s / np.pi) * panorama.shape[0]

#             # 使用双线性插值来提高图像质量
#             u0, u1 = int(np.floor(u)), int(np.ceil(u))
#             v0, v1 = int(np.floor(v)), int(np.ceil(v))

#             # 确保在图像范围内
#             if u0 < 0: u0 = 0
#             if u1 >= panorama.shape[1]: u1 = panorama.shape[1] - 1
#             if v0 < 0: v0 = 0
#             if v1 >= panorama.shape[0]: v1 = panorama.shape[0] - 1

#             # 获取四个相邻像素值
#             A = panorama[v0, u0]
#             B = panorama[v0, u1]
#             C = panorama[v1, u0]
#             D = panorama[v1, u1]

#             # 进行双线性插值
#             perspective_img[y, x] = (
#                 A * (u1 - u) * (v1 - v) +
#                 B * (u - u0) * (v1 - v) +
#                 C * (u1 - u) * (v - v0) +
#                 D * (u - u0) * (v - v0)
#             ).astype(np.uint8)

#     return perspective_img



# def generate_video(image_paths, out_dir, FOV, deg, gen_video=True, save_frames=True):
#     pers = [cv2.imread(image_path) for image_path in image_paths]
#     # 使用一个示例全景图，假设它是第一个视角图
#     new_pano = pers[0]  # 这里可以选择合适的全景图

#     if not gen_video:
#         return
    
#     video_size = (727, 1280)
#     save_video_path = os.path.join(out_dir, 'video.mp4')
#     out = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'MP4V'), 60, video_size)

#     interval_deg = 0.5
#     num_frames = int(360 / interval_deg)

#     image_save_path = os.path.join(out_dir, 'images')
#     os.makedirs(image_save_path, exist_ok=True)

#     for i in range(num_frames):
#         deg = i * interval_deg
#         img = get_perspective(new_pano, FOV, deg, 0, video_size[0], video_size[1])

#         img = np.clip(img, 0, 255).astype(np.uint8)

#         # 保存每帧为图像
#         if save_frames:
#             frame_filename = os.path.join(image_save_path, f'{i:04d}.png')
#             cv2.imwrite(frame_filename, img)

#         out.write(img)
#     out.release()














