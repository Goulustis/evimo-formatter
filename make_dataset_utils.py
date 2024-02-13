import numpy as np
import json
import os
import os.path as osp
import glob

from concurrent import futures
from tqdm import tqdm

SRC_EVS_ROOT="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/evimo2_v2_data/npz/samsung_mono"
SRC_RGB_ROOT="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/evimo2_v2_data/npz/flea3_7"

def find_data_dir(src_dir, scene):
    dir_ls = glob.glob(osp.join(src_dir, "**", scene), recursive=True)
    assert len(dir_ls) == 1, f"{len(dir_ls)} found!"
    return dir_ls[0]


def get_data_dir(scene, key = "rgb"):
    if key == "rgb":
        return find_data_dir(SRC_RGB_ROOT, scene)
    elif key == "evs":
        return find_data_dir(SRC_EVS_ROOT, scene)


def parallel_map(f, iterable, max_threads=None, show_pbar=False, desc="", **kwargs):
  """Parallel version of map()."""
  with futures.ThreadPoolExecutor(max_threads) as executor:
    if show_pbar:
      results = tqdm(
          executor.map(f, iterable, **kwargs), total=len(iterable), desc=desc)
    else:
      results = executor.map(f, iterable, **kwargs)
    return list(results)

def make_camera(ext_mtx, intr_mtx, dist, img_size, transform=None):
    """
    input:
        ext_mtx (np.array): World to cam matrix - shape = 4x4
        intr_mtx (np.array): intrinsic matrix of camera - shape = 3x3
        img_size (np.array): images shape in (w, h)

    return:
        nerfies.camera.Camera of the given mtx
    """
    from nerfies.camera import Camera
    R = ext_mtx[:3,:3]
    t = ext_mtx[:3,3]
    k1, k2, p1, p2, k3 = dist

    if transform is not None:
       R = R@transform.T

    coord = -t.T@R  

    cx, cy = intr_mtx[:2,2].astype(int)

    new_camera = Camera(
        orientation=R,
        position=coord,
        focal_length=intr_mtx[0,0],
        pixel_aspect_ratio=1,
        principal_point=np.array([cx, cy]),
        radial_distortion=(k1, k2, 0),
        tangential_distortion=(p1, p2),
        skew=0,
        image_size=img_size  ## (width, height) of camera
    )

    return new_camera

def create_and_write_camera_extrinsics(extrinsic_dir, cams, time_stamps, intr_mtx, dist, img_size, scale=None, ret_cam=False, transform=None):
    """
    create the extrinsics and save it
    scale: float = scale to resize image by; will apply to camera
    """
    os.makedirs(extrinsic_dir, exist_ok=True)

    # if len(glob.glob(osp.join(extrinsic_dir, "*.json"))) == len(cams):
    #     return

    cameras = []
    for i, (ecam,t) in enumerate(zip(cams, time_stamps)):
        camera = make_camera(ecam, intr_mtx, dist, img_size, transform=transform)
        if scale is not None:
           camera = camera.scale(scale)

        cameras.append(camera)
        targ_cam_path = osp.join(extrinsic_dir, str(i).zfill(6) + ".json")
        print("saving to", targ_cam_path)
        cam_json = camera.to_json()
        cam_json["t"] = float(t)
        with open(targ_cam_path, "w") as f:
            json.dump(cam_json, f, indent=2)

    if ret_cam:
        return cameras


def write_metadata(eimgs_ids, eimgs_ts, targ_dir):
    """
    saves the eimg ids as metatdata 
    input:
        eimgs_ids (np.array [int]) : event image ids
        eimgs_ts (np.array [int]) : time stamp of each event image
        targ_dir (str): directory to save to
    """
    metadata = {}

    for i, (id, t) in enumerate(zip(eimgs_ids, eimgs_ts)):
        metadata[str(i).zfill(6)] = {"warp_id":int(id),
                                     "appearance_id":int(id),
                                     "camera_id":0,
                                     "t":int(t)}
    
    with open(osp.join(targ_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
