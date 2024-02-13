import numpy as np
import glob
import os.path as osp

import cv2
from tqdm import tqdm

def main():
    prep_dir = "dev_est_scene_03_03_000002"
    save_f = osp.join(prep_dir, "comb_data.mp4")

    sort_key_fn = lambda x: int(osp.basename(x.split('.')[0]).split("_")[-1])
    img_fs = sorted(glob.glob(osp.join(prep_dir,"images/*.png")), key=sort_key_fn)
    sumE_fs = sorted(glob.glob(osp.join(prep_dir, "sumE/*.txt")), key=sort_key_fn)
    sumP_fs = sorted(glob.glob(osp.join(prep_dir, "sumP/*.txt")), key=sort_key_fn)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or another codec
    video_writer = cv2.VideoWriter(save_f, fourcc, 15, cv2.imread(img_fs[0]).shape[:2][::-1])

    for idx in tqdm(range(len(sumE_fs))):
        img = cv2.imread(img_fs[idx])
        sump = np.loadtxt(sumP_fs[idx], delimiter=",")
        img[sump != 0] = 255
        video_writer.write(img)
    
    video_writer.release()


if __name__ == "__main__":
    main()
