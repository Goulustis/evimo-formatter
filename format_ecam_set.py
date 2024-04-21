import glob
import numpy as np
import os
import os.path as osp
import json
import argparse


from ev_buffer import EventBuffer
from eimg_maker import create_event_imgs, create_eimg_by_triggers
from camera_utils import (load_intrinsics_data, 
                          load_extrinsics_data, 
                          load_rig_extrnxs,
                          warp_camera_frame)
from slerp_qua import create_interpolated_cams
from make_dataset_utils import (create_and_write_camera_extrinsics, 
                               write_metadata, 
                               find_data_dir,
                               write_train_valid_split)


def save_eimgs(eimgs, targ_dir):
    if eimgs is None:
        return 

    eimgs_dir = targ_dir
    os.makedirs(eimgs_dir, exist_ok=True)
    np.save(osp.join(eimgs_dir, "eimgs_1x.npy"), eimgs)
    del eimgs


def calc_time_delta(triggers, min_mult=3):
    t_diffs = np.diff(triggers)
    delta_t = np.mean(t_diffs) + np.std(t_diffs)
    # delta_t = triggers[1] - triggers[0]
    n_mult = np.round(delta_t/0.005)
    n_mult = max(min_mult, n_mult)
    
    return np.ceil(delta_t/n_mult * 1e6)/1e6

def main(targ_dir, evs_data_dir, rgb_data_dir, make_eimgs=True):
    make_trig_events = False  ## for debug only

    
    ## create event images
    events = EventBuffer(evs_data_dir)
    rgb_info = np.load(osp.join(rgb_data_dir, "dataset_info.npz"), allow_pickle=True)["meta"].item()["frames"]
    triggers = np.array([frame["ts"] for frame in rgb_info])
    time_delta = calc_time_delta(triggers)

    if make_trig_events:
        eimgs, eimg_ts, eimgs_ids, trig_ids = create_eimg_by_triggers(events, triggers, time_delta, make_eimg=make_eimgs)
    else:    
        eimgs, eimg_ts, eimgs_ids, trig_ids = create_event_imgs(events, triggers, create_imgs=make_eimgs, time_delta=time_delta)

    save_eimgs(eimgs, osp.join(targ_dir, "eimgs"))

    ## create extrinsics
    extrinsic_targ_dir = osp.join(targ_dir, "camera")

    ## warp the rgb full trajectory camera to the event camera
    intr_mtx, dist = load_intrinsics_data(osp.join(evs_data_dir, "dataset_info.npz"))
    rgb_cams, rgb_ts = load_extrinsics_data(osp.join(rgb_data_dir, "dataset_info.npz"))
    Tre, Trc = load_rig_extrnxs(osp.join(evs_data_dir, "dataset_extrinsics.npz")), load_rig_extrnxs(osp.join(rgb_data_dir, "dataset_extrinsics.npz"))
    ctrl_evs_cams = warp_camera_frame(rgb_cams, Tre, Trc)
    ctrl_evs_ts = rgb_ts

    ecams = create_interpolated_cams(eimg_ts, ctrl_evs_ts, ctrl_evs_cams)

    create_and_write_camera_extrinsics(extrinsic_targ_dir, ecams, eimg_ts * 1e6, intr_mtx, dist, img_size=(640, 480))

    # create metadata.json
    write_metadata(eimgs_ids, eimg_ts, targ_dir)

    # save the trig_ids; make the color camera ids the same
    np.save(osp.join(targ_dir, "trig_ids.npy"), trig_ids)

    # create train valid split
    write_train_valid_split(eimgs_ids, targ_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--targ_dir", default="")
    parser.add_argument("--evs_data_dir", default="")
    parser.add_argument("--rgb_data_dir", default="")
    parser.add_argument("--scene", default="")
    parser.add_argument("--make_eimgs", default=True)
    args = parser.parse_args()

    find_dir_fn = lambda x : find_data_dir(x, args.scene)
    evs_data_dir, rgb_data_dir = find_dir_fn(args.evs_data_dir), find_dir_fn(args.rgb_data_dir)


    main(args.targ_dir, evs_data_dir, rgb_data_dir, args.make_eimgs)
