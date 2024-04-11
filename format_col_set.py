import numpy as np
import os.path as osp
import os
import cv2
import bisect
from tqdm import tqdm
import scipy.ndimage as ndimage
import scipy.signal as signal
import glob
import json
import argparse
from sklearn.model_selection import train_test_split

from camera_utils import load_camera_data, load_ts
from slerp_qua import create_interpolated_ecams
from make_dataset_utils import create_and_write_camera_extrinsics, parallel_map, find_data_dir


## drop the last frame to avoid complications
get_img_ids = lambda img_dir : [osp.basename(f).split(".")[0] for f in sorted(glob.glob(osp.join(img_dir, "*.png")))]


def split_and_find_indices(nums, num_parts):
    part_size = len(nums) // num_parts  # Determine the size of each part
    indices = []  # To store the original indices of the smallest elements in each part

    for part in range(num_parts):
        start_index = part * part_size
        # Handle the last part which might contain the remaining elements
        end_index = start_index + part_size if part < num_parts - 1 else len(nums)

        # Extract the part
        current_part = nums[start_index:end_index]
        
        # Find the smallest element and its index in the current part
        min_index_in_part = current_part.index(min(current_part))
        
        # Calculate the original index in the full list and add it to the result
        original_index = start_index + min_index_in_part
        indices.append(original_index)

    return indices

def find_clear_val_test(img_dir):
    ## ignore last frame for complications
    img_fs = sorted(glob.glob(osp.join(img_dir, "*.png")))[:-1]
    
    ignore_first = 32
    img_idxs = [osp.basename(f).split(".")[0] for f in img_fs]

    # Load images
    images = parallel_map(cv2.imread, img_fs, show_pbar=True, desc="loading imgs")

    blur_scores = []
    laplacian_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)
    blur_kernels = np.array([[
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ], [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ], [
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0]
    ]], dtype=np.float32) / 5.0
    for image in tqdm(images, desc="caculating blur score"):
        gray_im = np.mean(image, axis=2)[::4, ::4]

        directional_blur_scores = []
        for i in range(4):
            blurred = ndimage.convolve(gray_im, blur_kernels[i])

            laplacian = signal.convolve2d(blurred, laplacian_kernel, mode="valid")
            var = laplacian**2
            var = np.clip(var, 0, 1000.0)

            directional_blur_scores.append(np.mean(var))

        antiblur_index = (np.argmax(directional_blur_scores) + 2) % 4

        blur_score = directional_blur_scores[antiblur_index]
        blur_scores.append(blur_score)
    
    # ids = np.argsort(blur_scores) + ignore_first
    ids = np.argsort(blur_scores[-ignore_first:]) + ignore_first
    best = ids[-30:]
    np.random.shuffle(best)
    # best = list(str(x) for x in best)

    # test, val = best[:15], best[15:]
    # clear_image_idxs = [img_idxs[e] for e in best]
    clear_image_idxs = []
    for e in best:
        clear_image_idxs.append(img_idxs[e])
    # return test, val
    return sorted(clear_image_idxs[:15]), sorted(clear_image_idxs[15:])


def load_frame_data(img_npz_f, ret_id=False, prefix=None, idxs=None):
    img_npz = np.load(img_npz_f)
    classical_ids = sorted(list(img_npz.keys()))

    if (prefix is not None) and (idxs is not None):
        classical_ids = [f"{prefix}_{idx}" for idx in idxs]


    imgs = parallel_map(lambda x : img_npz[x], classical_ids, show_pbar=True, desc="loading rgb imgs")
    imgs = [x.astype(imgs[0].dtype) for x in imgs]

    if ret_id:
        return imgs, classical_ids
    
    return imgs


def load_and_save_image(rgb_data_dir, save_img_dir, scale=2, img_npz_f = None):

    os.makedirs(save_img_dir, exist_ok=True)
    if img_npz_f is None:
        img_npz_f = osp.join(rgb_data_dir, "dataset_classical.npz")
    
    imgs, classical_ids = load_frame_data(img_npz_f, ret_id=True)
    ## downsize image by 2x as in de-nerf paper
    h, w = imgs[0].shape[:2]
    new_size = w//scale, h//scale
    imgs = parallel_map(lambda x : cv2.resize(x, new_size, interpolation=cv2.INTER_AREA), 
                        imgs, show_pbar=True, desc="downsize imgs")

    def save_fn(inp):
        idx, img = inp
        save_f = osp.join(save_img_dir, str(idx).zfill(5) + ".png")
        cv2.imwrite(save_f, img)

    img_ids = [str(idx).zfill(5) for idx in list(range(len(imgs)))]
    parallel_map(save_fn, list(zip(img_ids, imgs)), show_pbar=True, desc="saving imgs")

    return img_ids, classical_ids


def load_and_save_msk(rgb_data_dir, save_msk_dir, scale=2, depth_npz_f=None):
    os.makedirs(save_msk_dir, exist_ok=True)
    if depth_npz_f is None:
        depth_npz_f = osp.join(rgb_data_dir, "dataset_mask.npz")
    
    msks, _ = load_frame_data(depth_npz_f, ret_id=True)
    ## downsize image by 2x as in de-nerf paper
    h, w = msks[0].shape[:2]
    new_size = w//scale, h//scale
    msks = parallel_map(lambda x : cv2.resize(x, new_size, interpolation=cv2.INTER_AREA), 
                        msks, show_pbar=True, desc="downsize imgs")
    msks = np.stack(msks) > 0

    save_mks_f = osp.join(save_msk_dir, "msk.npy")
    np.save(save_mks_f, msks)
    return len(msks)


def write_train_test_metadata(img_dir, classical_ids, save_dir):
    # drop last frame to deal with complications
    img_ids = get_img_ids(img_dir)[:-1]
    val_ids, test_ids = find_clear_val_test(img_dir)
    train_ids = sorted(set(img_ids) - set(val_ids + test_ids))

    dataset_json = {
        'count': len(img_ids),
        'num_exemplars': len(train_ids),
        'ids': img_ids,
        "classical_ids": classical_ids,
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids
    }

    with open(osp.join(save_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)
    
    return train_ids, val_ids, test_ids



def write_metadata(img_ids, trig_ids, triggers, train_ids, val_ids, test_ids, save_dir):
    img_trig_dic = {}
    img_trig_id_dic = {}
    for image_id, trig_t, trig_id in zip(img_ids, triggers, trig_ids):
        img_trig_dic[image_id] = int(trig_t)
        img_trig_id_dic[image_id] = int(trig_id)
    

    metadata_json = {}
    for i, image_id in enumerate(train_ids):
        metadata_json[image_id] = {
            'warp_id': img_trig_id_dic[image_id],
            'appearance_id': img_trig_id_dic[image_id],
            'camera_id': 0,
            't':img_trig_dic[image_id]
        }

    for i, image_id in enumerate(val_ids + test_ids):
        i = bisect.bisect_left(train_ids, image_id)
        metadata_json[image_id] = {
            'warp_id': img_trig_id_dic[image_id],
            'appearance_id': img_trig_id_dic[image_id],
            'camera_id': 0,
            't':img_trig_dic[image_id]
        }
    
    metadata_json_path = osp.join(save_dir , 'metadata.json')
    with open(metadata_json_path, "w") as f:
        json.dump(metadata_json, f, indent=2)



def main(targ_dir, trig_ids_f, rgb_data_dir):

    os.makedirs(targ_dir, exist_ok=True)

    ## load the image and copy them over
    rgb_save_dir = osp.join(targ_dir, "rgb", "1x")
    img_ids, classical_ids = load_and_save_image(rgb_data_dir, rgb_save_dir)
    n_msks = load_and_save_msk(rgb_data_dir,targ_dir)

    assert n_msks == len(img_ids), "num mask != num images!"
    img_ids, classical_ids = img_ids[:-1], classical_ids[:-1]

    ## make train test split
    train_ids, val_ids, test_ids = write_train_test_metadata(rgb_save_dir, classical_ids, targ_dir)

    trig_ids = np.load(trig_ids_f)
    rgb_ts = load_ts(osp.join(rgb_data_dir, "dataset_info.npz"), key="frames")
    ctrl_extrxs, ctrl_ts, intrxs, dist = load_camera_data(osp.join(rgb_data_dir, "dataset_info.npz"))

    ## write the extrinsics
    extrinsic_targ_dir = osp.join(targ_dir, "camera")
    ## rescale image by half according to de-nerf paper
    extrinsics = create_interpolated_ecams(rgb_ts, ctrl_ts, ctrl_extrxs)
    create_and_write_camera_extrinsics(extrinsic_targ_dir, extrinsics, rgb_ts, intrxs, dist, scale=0.5, img_size=(2080, 1552))
    # create_and_write_camera_extrinsics(extrinsic_targ_dir, extrinsics, triggers, intrxs, dist, img_size=(2080, 1552))

    ## write metadata
    write_metadata(img_ids, trig_ids, rgb_ts, train_ids, val_ids, test_ids, targ_dir)


if __name__ == "__main__":
    np.random.seed(32)
    parser = argparse.ArgumentParser()
    parser.add_argument("--targ_dir", default="")
    parser.add_argument("--trig_ids_f", default="")
    parser.add_argument("--rgb_data_dir", default="")
    parser.add_argument("--scene", default="")
    args = parser.parse_args()

    rgb_data_dir = find_data_dir(args.rgb_data_dir, args.scene)
    main(args.targ_dir, args.trig_ids_f, rgb_data_dir)