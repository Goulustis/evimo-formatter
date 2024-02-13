from format_col_set import load_and_save_image
from ev_buffer import EventBuffer
from eimg_maker import create_eimg_by_triggers
import numpy as np
from make_dataset_utils import parallel_map
import os.path as osp
import matplotlib.pyplot as plt
import os


def main():
    load_and_save_image("", "tmp/reproj",
                        img_npz_f="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/evimo2_v2_data/npz/samsung_mono/sfm/eval/scene_03_03_000002/dataset_reprojected_classical.npz",
                        scale=1)

def eimg_to_img(eimg):
    img = np.zeros((*eimg.shape[:2],3), dtype=np.uint8)
    img[eimg < 0,0] = 255
    img[eimg > 0,1] = 255
    return img

def save_eimg2imgs(eimgs, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    def make_img_fn(inp):
        idx, eimg = inp
        img = eimg_to_img(eimg)
        save_f = osp.join(save_dir, str(idx).zfill(4) + ".png")
        plt.imsave(save_f, img)
    
    parallel_map(make_img_fn, list(zip(list(range(len(eimgs))), eimgs)), show_pbar=True, desc="makeing eimgs")


def trig_reproj():
    evs_data_dir = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/evimo2_v2_data/npz/samsung_mono/sfm/eval/scene_03_03_000002"
    events = EventBuffer(evs_data_dir)

    reproj_npz_f = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/evimo2_v2_data/npz/samsung_mono/sfm/eval/scene_03_03_000002/dataset_reprojected_classical.npz"
    re = np.load(reproj_npz_f, allow_pickle=True)
    triggers = re["t"]
    eimgs, _, _, _ = create_eimg_by_triggers(events, triggers, 0.005,True)
    save_eimg2imgs(eimgs, "tmp/eimg_reproj")



def show_proj_overlay():
    eimg_f = "/ubc/cs/research/kmyi/matthew/projects/evimo_formatter/tmp/eimg_reproj/0221.png"
    img_f = "/ubc/cs/research/kmyi/matthew/projects/evimo_formatter/tmp/reproj/00221.png"

    eimg = plt.imread(eimg_f)
    img = plt.imread(img_f)

    msk = eimg.sum(axis=-1) > 1
    img[msk] = 255
    
    assert 0

if __name__ == "__main__":
    # trig_reproj()
    # eimgs = np.load("/ubc/cs/research/kmyi/matthew/projects/evimo_formatter/data_formatted/checkerboard_2_tilt_fb_000000/trig_ecam_set/eimgs/eimgs_1x.npy")
    # save_eimg2imgs(eimgs, "tmp/checker_trig_eimgs")
    trig_reproj()