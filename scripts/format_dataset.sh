#!/bin/bash
source ~/.bashrc
conda activate ecam_proc

which python

################# options needed ######################
SCENE=scene7_00_000001
################# options needed ######################



SRC_EVS_ROOT=/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/evimo2_v2_data/npz/samsung_mono
SRC_RGB_ROOT=/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/evimo2_v2_data/npz/flea3_7
TARG_DIR=/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/$SCENE
THRESH_PREP_DIR=cache_thresh_est/$SCENE
DATAPATH=$(python make_dataset_utils.py --src_dir $SRC_EVS_ROOT --scene $SCENE)


python format_ecam_set.py --targ_dir $TARG_DIR/ecam_set \
                          --evs_data_dir $SRC_EVS_ROOT \
                          --rgb_data_dir $SRC_RGB_ROOT \
                          --scene $SCENE

python format_col_set.py --targ_dir $TARG_DIR/colcam_set \
                         --trig_ids_f $TARG_DIR/ecam_set/trig_ids.npy \
                         --rgb_data_dir $SRC_RGB_ROOT \
                         --scene $SCENE


###### MAKING DECAM SET ##############↓↓↓↓


python project_rgb_to_evs.py \
                    --overwrite \
                    --reprojectbgr \
                    --format=evimo2v2 \
                    $DATAPATH


python thresh_est/prepare_for_thresh_est.py --evs_data_dir $SRC_EVS_ROOT \
                                            --scene $SCENE \
                                            --targ_dir $prep_dir

echo threshold estimating
matlab -nodisplay -nosplash -nodesktop -batch "threshEst('$THRESH_PREP_DIR', '$THRESH_PREP_DIR');exit;"
echo threshold estimating SUCCESS!!

python format_decam_set.py --evs_data_dir $SRC_EVS_ROOT \
                           --rgb_data_dir $SRC_RGB_ROOT \
                           --scene $SCENE \
                           --colcam_dir $TARG_DIR/colcam_set \
                           --ecam_dir $TARG_DIR/ecam_set \
                           --thresh_dir $THRESH_PREP_DIR \
                           --targ_dir $TARG_DIR/decam_set