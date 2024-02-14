import numpy as np
import glob
import os.path as osp
import os


from ev_buffer import EventBuffer
from format_ecam_set import calc_time_delta, save_eimgs
from eimg_maker import DeimgsCreator
from camera_utils import (load_intrinsics_data, 
                          load_extrinsics_data, 
                          load_rig_extrnxs,
                          warp_camera_frame)
from slerp_qua import create_interpolated_ecams
from make_dataset_utils import (create_and_write_camera_extrinsics, 
                               write_metadata, 
                               find_data_dir,
                               write_train_valid_split)
import argparse

def main(evs_data_dir, rgb_data_dir, colcam_dir, ecam_dir, thresh_dir, targ_dir):

    ## create denerf images
    events = EventBuffer(evs_data_dir)
    rgb_info = np.load(osp.join(rgb_data_dir, "dataset_info.npz"), allow_pickle=True)["meta"].item()["frames"]
    reproj_npz = np.load(osp.join(evs_data_dir, "dataset_reprojected_classical.npz"), allow_pickle=True)
    biases = np.loadtxt(osp.join(thresh_dir, "bias.csv"), delimiter=",")
    scales = np.loadtxt(osp.join(thresh_dir, "scale.csv"), delimiter=",")
    triggers = np.array([frame["ts"] for frame in rgb_info])
    time_delta = calc_time_delta(triggers)

    deimgCreator = DeimgsCreator(events, triggers, reproj_npz, 
                                 scales, biases, colcam_dir, ecam_dir, time_delta)
    deimgs, deimg_ts, deimg_ids, deimg_msks  = deimgCreator.create_deimgs()
    save_eimgs(deimgs, osp.join(targ_dir, "eimgs"))
    save_eimgs(deimg_msks, osp.join(targ_dir, "eimg_msks"))

    ## create extrinsics
    extrinsic_targ_dir = osp.join(targ_dir, "camera")

    ## warp the rgb full trajectory camera to the event camera
    intr_mtx, dist = load_intrinsics_data(osp.join(evs_data_dir, "dataset_info.npz"))
    rgb_cams, rgb_ts = load_extrinsics_data(osp.join(rgb_data_dir, "dataset_info.npz"))
    Tre, Trc = load_rig_extrnxs(osp.join(evs_data_dir, "dataset_extrinsics.npz")), load_rig_extrnxs(osp.join(rgb_data_dir, "dataset_extrinsics.npz"))
    ctrl_evs_cams = warp_camera_frame(rgb_cams, Tre, Trc)
    ctrl_evs_ts = rgb_ts

    ecams = create_interpolated_ecams(deimg_ts, ctrl_evs_ts, ctrl_evs_cams)

    create_and_write_camera_extrinsics(extrinsic_targ_dir, ecams, deimg_ts, intr_mtx, dist, img_size=(640, 480))

    # create metadata.json
    write_metadata(deimg_ids, deimg_ts, targ_dir)

    # create train valid split
    write_train_valid_split(deimg_ids, targ_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evs_data_dir", default="")
    parser.add_argument("--rgb_data_dir", default="")
    parser.add_argument("--scene", default="")
    parser.add_argument("--colcam_dir", default="")
    parser.add_argument("--ecam_dir", default="")
    parser.add_argument("--thresh_dir", default="")
    parser.add_argument("--targ_dir", default="")
    
    args = parser.parse_args()

    find_dir_fn = lambda x : find_data_dir(x, args.scene)
    evs_data_dir, rgb_data_dir = find_dir_fn(args.evs_data_dir), find_dir_fn(args.rgb_data_dir)


    main(evs_data_dir, rgb_data_dir, args.colcam_dir, args.ecam_dir, args.thresh_dir, args.targ_dir)