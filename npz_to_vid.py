## concatenate all depth and images together

import numpy as np
import cv2
import glob
import os.path as osp
import os
from make_dataset_utils import parallel_map
from tqdm import tqdm

def format_data_save(classical_f):
    camera = "flea3_7"
    save_f = osp.join(camera, osp.dirname(classical_f.split("flea3_7")[-1])[1:]) + ".mp4"
    save_dir = osp.dirname(save_f)
    os.makedirs(save_dir, exist_ok=True)

    depth_f = osp.join(osp.dirname(classical_f), "dataset_depth.npz")

    rgb_data = np.load(classical_f, allow_pickle=True)
    depth_data = np.load(depth_f, allow_pickle=True)
    rgb_keys = sorted(list(rgb_data.keys()))
    depth_keys = sorted(list(depth_data.keys()))

    # Assume the first RGB image defines the resolution
    first_rgb = rgb_data[rgb_keys[0]]
    height, width, _ = first_rgb.shape
    combined_width = width * 2  # since images are concatenated horizontally
    fps = 16  # or another value based on your requirements
    img_size = (combined_width//2, height//2)

    calc_max = depth_data[depth_keys[len(depth_data)//2]].max() * 1.2
    max_depth = 255 if calc_max == 0 else calc_max

    # Initialize cv2 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or another codec
    video_writer = cv2.VideoWriter(save_f, fourcc, fps, img_size)

    for rgb_k, depth_k in tqdm(zip(rgb_keys, depth_keys), total=len(rgb_keys), desc=osp.basename(save_f)):
        rgb = rgb_data[rgb_k]
        dp = (255 * depth_data[depth_k].astype(np.float32) / max_depth).astype(np.uint8)
        dp = np.stack([dp] * 3, axis=-1)
        comb_img = np.concatenate([rgb, dp], axis=1)
        comb_img = cv2.flip(comb_img, -1)
        comb_img = cv2.resize(comb_img, img_size, interpolation = cv2.INTER_LINEAR)

        # Write comb_img to VideoWriter
        video_writer.write(comb_img)

    # Close the VideoWriter
    video_writer.release()
    


def main():
    cam_dir = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/evimo2_v2_data/npz/flea3_7"

    camera = osp.basename(cam_dir)
    npz_fs = sorted(glob.glob(osp.join(cam_dir, "**/dataset_classical.npz"), recursive=True))


    parallel_map(format_data_save, npz_fs)
    # format_data_save(npz_fs[0])



main()