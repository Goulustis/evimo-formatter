import numpy as np
import glob
import os
import os.path as osp
import cv2
from tqdm import tqdm
import argparse

from make_dataset_utils import find_data_dir, parallel_map
from eimg_maker import ev_to_eimg
from ev_buffer import EventBuffer


def save_imgs_and_ts(reproj_f, save_img_dir):
    os.makedirs(save_img_dir, exist_ok=True)
    reproj = np.load(reproj_f, allow_pickle=True)
    img_keys = sorted([key for key in reproj.keys() if (not "mask" in key) and ("classical" in key)])[1:]

    def save_fn(inp):
        i, key = inp
        img = reproj[key]
        save_f = osp.join(save_img_dir, str(i) + ".png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(save_f, img)
    
    parallel_map(save_fn, list(zip(list(range(1, len(img_keys) + 1)), img_keys)), show_pbar=True, desc="saving imgs")
    np.savetxt(osp.join(osp.dirname(save_img_dir), "classical_keys.txt"), np.array(img_keys), fmt="%s")


def save_sume_and_sump(evs_data_dir, reproj_f, save_dir):
    sume_dir = osp.join(save_dir, "sumE")
    sump_dir = osp.join(save_dir, "sumP")
    msk_dir = osp.join(save_dir , "msk")  # mask of flatten events of whether it contributes to sum{e,p}
    [os.makedirs(f, exist_ok=True) for f in [sume_dir, sump_dir, msk_dir]]

    buffer = EventBuffer(evs_data_dir)
    reproj = np.load(reproj_f, allow_pickle=True)
    msk_keys = sorted([key for key in reproj.keys() if ("mask" in key) and ("classical" in key)])
    get_msk = lambda idx : reproj[msk_keys[idx]]

    frame_ts = reproj["t"][1:]

    for sv_idx, i in tqdm(enumerate(range(len(frame_ts) - 1),1), total=len(frame_ts) - 1):
        prev_t, next_t = frame_ts[i], frame_ts[i+1]

        if i == 0:
            prev_msk, next_msk = get_msk(1), get_msk(2)
        else:
            prev_msk, next_msk = get_msk(i), get_msk(i+1)

        msk = prev_msk & next_msk

        ts, xs, ys, ps = buffer.retrieve_data(prev_t, next_t)

        sume = ev_to_eimg(xs, ys, np.ones(len(xs)))*msk
        sump = ev_to_eimg(xs, ys, ps)*msk
        lst_msk = msk

        save_sume_f = osp.join(sume_dir, f"data_event_{sv_idx}.txt")
        save_sump_f = osp.join(sump_dir, f"data_polarity_{sv_idx}.txt")
        save_msk_f = osp.join(msk_dir, f"mask_{sv_idx}.npy")

        np.savetxt(save_sume_f, sume, delimiter=",", fmt="%i")
        np.savetxt(save_sump_f, sump, delimiter=",", fmt="%i")
        np.save(save_msk_f, lst_msk)
    
    np.save(osp.join(save_dir, "frame_ts.npy"), frame_ts)



        


def main(evs_data_dir=None, targ_dir=None):
    targ_dir = "dev_est_scene_03_03_000002"
    evs_data_dir = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/evimo2_v2_data/npz/samsung_mono/sfm/eval/scene_03_03_000002"


    reproj_f = osp.join(evs_data_dir, "dataset_reprojected_classical.npz")
    save_img_dir = osp.join(targ_dir, "images")
    save_imgs_and_ts(reproj_f, save_img_dir)

    save_sume_and_sump(evs_data_dir, reproj_f, targ_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evs_data_dir", default="")
    parser.add_argument("--scene", default="")
    parser.add_argument("--targ_dir", default="")

    args = parser.parse_args()
    find_dir_fn = lambda x : find_data_dir(x, args.scene)
    evs_data_dir = find_dir_fn(args.evs_data_dir)


    main(evs_data_dir, args.targ_dir)