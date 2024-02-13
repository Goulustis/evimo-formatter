import numpy as np
import json
import glob
import matplotlib.pyplot as plt
import os.path as osp
from scipy.spatial.transform import Rotation

from camera_utils import load_rig_extrnxs, apply_transform, inv_transform

def load_pos(json_f, ret_mtx=False):
    with open(json_f, "r") as f:
        data = json.load(f)
        R = np.array(data["orientation"])
        T = np.array(data["position"]).reshape(3,1)
        if ret_mtx:
            return np.concatenate([R, T], axis=1)

        return T.squeeze()


def main():
    scene_dir = "debug"
    
    rgb_cam_dir = osp.join(scene_dir, "colcam_set", "camera")
    evs_cam_dir = osp.join(scene_dir, "ecam_set", "camera")

    rgb_pos = np.stack([load_pos(f) for f in sorted(glob.glob(osp.join(rgb_cam_dir, "*.json")))])
    evs_pos = np.stack([load_pos(f) for f in sorted(glob.glob(osp.join(evs_cam_dir, "*.json")))])[::6]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot RGB camera positions
    ax.scatter(rgb_pos[:, 0], rgb_pos[:, 1], rgb_pos[:, 2], c='r', marker='o', label='RGB Cameras')
    
    # Plot EVS camera positions
    ax.scatter(evs_pos[:, 0], evs_pos[:, 1], evs_pos[:, 2], c='b', marker='^', label='EVS Cameras')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Camera Positions')
    plt.legend()
    plt.show()

    return rgb_pos, evs_pos

main()