import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import lib.Equirec2Perspec as E2P
import lib.multi_Perspec2Equirec as m_P2E
import uuid
from tqdm import tqdm

def generate_video(image_paths, out_dir, FOV, deg, gen_video=False, save_frames=False):
    # Load images from the provided paths
    pers = [cv2.imread(image_path) for image_path in image_paths]
    if any(img is None for img in pers):
        print("Error: One or more images could not be loaded.")
        return

    # Generate panoramic image
    pers_matrix = [[FOV, deg * i, 0] for i in range(int(360 / deg))]
    ee = m_P2E.Perspective(pers, pers_matrix)
    new_pano = ee.GetEquirec(2048, 4096)
    cv2.imwrite(os.path.join(out_dir, 'pano.png'), new_pano.astype(np.uint8)[540:-540])

    if not gen_video:
        return

    # Set video parameters
    equ = E2P.Equirectangular(new_pano)
    fov = 90
    video_size = (512, 512)
    img = equ.GetPerspective(fov, 0, 0, video_size[0], video_size[1])
    size = (img.shape[1], img.shape[0])

    # Create video writer
    save_video_path = os.path.join(out_dir, 'video.mp4')
    out = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'avc1'), 60, size)

    interval_deg = 1
    num_frames = int(360 / interval_deg)
    image_save_path = os.path.join(out_dir, 'images')
    os.makedirs(image_save_path, exist_ok=True)

    for i in range(num_frames):
        frame_deg = i * interval_deg
        img = equ.GetPerspective(fov, frame_deg, 0, size[1], size[0])
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Optionally save each frame as an image
        if save_frames:
            frame_filename = os.path.join(image_save_path, f'{i:04d}.png')
            cv2.imwrite(frame_filename, img)

        out.write(img)

    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved at {save_video_path}")
