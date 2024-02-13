#!/bin/bash
source ~/.bashrc
conda activate ecam_proc

which python

SRC_EVS_ROOT=/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/evimo2_v2_data/npz/samsung_mono
SRC_RGB_ROOT=/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/evimo2_v2_data/npz/flea3_7

SCENE=checkerboard_2_tilt_fb_000000
TARG_DIR=data_formatted/checkerboard_2_tilt_fb_000000



python format_ecam_set.py --targ_dir $TARG_DIR/trig_ecam_set \
                          --evs_data_dir $SRC_EVS_ROOT \
                          --rgb_data_dir $SRC_RGB_ROOT \
                          --scene $SCENE

# python format_col_set.py --targ_dir $TARG_DIR/colcam_set \
#                          --trig_ids_f $TARG_DIR/ecam_set/trig_ids.npy \
#                          --rgb_data_dir $SRC_RGB_ROOT \
#                          --scene $SCENE
