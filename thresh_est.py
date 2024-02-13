import numpy as np
import os
import os.path as osp
import glob

from make_dataset_utils import get_data_dir

def main():
    scene = "checkerboard_2_tilt_fb_000000"
    rgb_data_dir, evs_data_dir = get_data_dir(scene, "rgb"), get_data_dir(scene, "evs")

    assert 0


main()