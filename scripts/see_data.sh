#!/bin/bash
conda activate tb

which python

python /ubc/cs/research/kmyi/matthew/projects/evimo_formatter/evimo/tools/evimo2_v2_inspect/view_sequence_all_cameras.py \
       --idir /ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/evimo2_v2_data/npz \
       --seq scene_04_00_000001
       # --seq checkerboard_2_tilt_fb_000000


