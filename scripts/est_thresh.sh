source ~/.bashrc
conda activate ecam_proc

evs_data_dir=/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/evimo2_v2_data/npz/samsung_mono/sfm/eval/scene_03_03_000002
prep_dir=dev_est_scene_03_03_000002

# python thresh_est/prepare_for_thresh_est.py --evs_data_dir $evs_data_dir \
#                                             --targ_dir $prep_dir

matlab -nodisplay -nosplash -nodesktop -batch "threshEst('$prep_dir', '$prep_dir');exit;"                                 