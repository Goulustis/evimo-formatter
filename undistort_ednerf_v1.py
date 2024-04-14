import os
import json
import cv2
import numpy as np
import glob
import os.path as osp
import shutil
import copy
from tqdm import tqdm

from camera_utils import to_hom
from make_dataset_utils import parallel_map, load_json




class ColUndistorter:
    def __init__(self, src_dir, dst_dir) -> None:
        self.src_dir = src_dir
        self.cam_dir = osp.join(src_dir, "camera")
        self.frame_dir = osp.join(src_dir, "rgb/1x")
        self._init_cam_data()
        self._build_undist_fn()

        self.dst_dir = dst_dir
        self.dst_frame_dir = osp.join(dst_dir, "rgb/1x")
        self.dst_cam_dir = osp.join(dst_dir, "camera")

        [os.makedirs(d, exist_ok=True) for d in [self.dst_frame_dir, self.dst_cam_dir]]

    def get_img_shape(self):
        img_f = glob.glob(osp.join(self.frame_dir, "*.png"))[0]
        img = cv2.imread(img_f)
        return img.shape[:2]


    def _init_cam_data(self):
        self.cam_data_fs = sorted(glob.glob(osp.join(self.cam_dir, "*.json")))
        self.cam_data = [load_json(f) for f in self.cam_data_fs]

        def build_intrxs(cam_datum):
            fx = fy = cam_datum["focal_length"]
            cx = cam_datum["principal_point"][0]
            cy = cam_datum["principal_point"][1]
            intrxs = np.array([[fx, 0 , cx],
                                [0, fy, cy],
                                [0, 0, 1]])
            return intrxs
        
        def build_dist(cam_datum):
            k1, k2, k3 = cam_datum["radial_distortion"]
            p1, p2 = cam_datum["tangential_distortion"]
            return np.array([k1, k2, p1, p2, k3])

        self.K = build_intrxs(self.cam_data[0])
        self.D = build_dist(self.cam_data[0])

        self.img_shape = self.get_img_shape()
    
    def _build_undist_fn(self):
        im_h, im_w = self.img_shape
        self.undist_K, self.roi = cv2.getOptimalNewCameraMatrix(
            self.K, self.D, (im_w, im_h), 1, (im_w, im_h)
        )
        x, y, w, h = self.roi
        self.undist_K[0, 2] = self.undist_K[0, 2] - x
        self.undist_K[1, 2] = self.undist_K[1, 2] - y

        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            self.K, self.D, None, self.undist_K, (im_w, im_h), cv2.CV_32FC1
        )

        self.undist_fn = lambda img: cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)[y:y+h, x:x+w]
        self.targ_img_shape = (h, w)
    
    def undistort_and_save(self):
        img_fs = sorted(glob.glob(osp.join(self.frame_dir, "*.png")))

        def transform_and_save(img_f):
            img = cv2.imread(img_f)
            undist_img = self.undist_fn(img)
            undist_img_f = osp.join(self.dst_frame_dir, osp.basename(img_f))
            cv2.imwrite(undist_img_f, undist_img)
        
        os.makedirs(self.dst_frame_dir, exist_ok=True)
        parallel_map(transform_and_save, img_fs, show_pbar=True, desc="Undistorting and save images")

    def update_intrxs(self, cam_datum):
        cam_datum["principal_point"] = [self.undist_K[0, 2], self.undist_K[1, 2]]
        cam_datum["focal_length"] = self.undist_K[0, 0]
        cam_datum["radial_distortion"] = [0, 0, 0]
        cam_datum["tangential_distortion"] = [0, 0]
        cam_datum["image_size"] = [self.targ_img_shape[1], self.targ_img_shape[0]]
        cam_datum["t"] = cam_datum["t"]*1e6   # convert to micro seconds
        return cam_datum

    def update_intrxs_and_save(self):

        undist_cam_data = parallel_map(self.update_intrxs, self.cam_data)
        os.makedirs(self.dst_cam_dir, exist_ok=True)

        def save_cam_datum(inp):
            data_f, cam_datum = inp
            targ_data_f = osp.join(self.dst_cam_dir, osp.basename(data_f))
            with open(targ_data_f, "w") as f:
                json.dump(cam_datum, f, indent=2)
        
        parallel_map(save_cam_datum, list(zip(self.cam_data_fs, undist_cam_data)), show_pbar=True, desc="Updating and saving camera data")

    def cp_other_files(self):
        targ_dataset_f = osp.join(self.dst_dir, "dataset.json")
        src_dataset_f = osp.join(self.src_dir, "dataset.json")
        src_dataset = load_json(src_dataset_f)
        src_dataset["train_ids"] = src_dataset["train_ids"][:-1]

        with open(targ_dataset_f, "w") as f:
            json.dump(src_dataset, f, indent=2)

        
        meta_f = osp.join(self.src_dir, "metadata.json")
        meta = {
            "colmap_scale":1
        }
        with open(meta_f, "r") as f:
            meta.update(json.load(f))
        
        targ_meta_f = osp.join(self.dst_dir, "metadata.json")
        with open(targ_meta_f, "w") as f:
            json.dump(meta, f, indent=2)

    def reformat(self):
        self.undistort_and_save()
        self.update_intrxs_and_save()
        self.cp_other_files()


class EcamUndistorter(ColUndistorter):
    def __init__(self, src_dir, dst_dir) -> None:
        self.src_dir = src_dir
        self.cam_dir = osp.join(src_dir, "camera")
        self.frame_dir = osp.join(src_dir, "eimgs")
        self._init_cam_data()
        self._build_undist_fn()

        self.dst_dir = dst_dir
        self.dst_frame_dir = osp.join(dst_dir, "eimgs")
        self.dst_prev_cam_dir = osp.join(dst_dir, "prev_camera")
        self.dst_next_cam_dir = osp.join(dst_dir, "next_camera")

        [os.makedirs(d, exist_ok=True) for d in [self.dst_frame_dir, self.dst_prev_cam_dir, self.dst_next_cam_dir]]
    
    def _build_undist_fn(self):
        super()._build_undist_fn()

        x, y, w, h = self.roi
        undist_fn = lambda img : cv2.remap(img, self.mapx, self.mapy, 
                                           cv2.INTER_NEAREST)[y:y+h, x:x+w]
        def undist_ev_img(eimg):
            pos_eimg, neg_eimg = np.copy(eimg), np.copy(eimg)
            pos_cond = eimg > 0
            pos_eimg[~pos_cond] = 0
            neg_eimg[pos_cond] = 0
            pos_eimg, neg_eimg = pos_eimg.astype(np.uint8), np.abs(neg_eimg).astype(np.uint8)
            pos_re, neg_re = undist_fn(pos_eimg), undist_fn(neg_eimg)

            return pos_re.astype(np.int8) + neg_re.astype(np.int8) * -1

        self.undist_fn = undist_ev_img
    
    def get_img_shape(self):
        return np.load(osp.join(self.frame_dir, "eimgs_1x.npy"), "r").shape[1:]

    def undistort_and_save(self):
        eimgs_f = osp.join(self.frame_dir, "eimgs_1x.npy")
        eimgs = np.load(eimgs_f, "r")

        undist_eimgs = np.zeros((len(eimgs), *self.targ_img_shape), dtype=np.int8)

        for i, eimg in tqdm(enumerate(eimgs), total=len(eimgs), desc="undist + save eimgs"):
            undist_eimg = self.undist_fn(eimg)
            undist_eimgs[i] = undist_eimg

        targ_eimg_f = osp.join(self.dst_frame_dir, "eimgs_1x.npy")
        np.save(targ_eimg_f, undist_eimgs)
            
    def update_intrxs_and_save(self):
        prev_cams, next_cams = copy.deepcopy(self.cam_data[:-1]), copy.deepcopy(self.cam_data[1:])
        undist_prev_cams = parallel_map(self.update_intrxs, prev_cams)
        undist_next_cams = parallel_map(self.update_intrxs, next_cams)

        idxs = [str(i).zfill(6) for i in range(len(prev_cams))]

        def save_prev_cam_datum(inp):
            idx, cam_datum = inp
            targ_data_f = osp.join(self.dst_prev_cam_dir, f"{idx}.json")
            with open(targ_data_f, "w") as f:
                json.dump(cam_datum, f, indent=2)
        
        def save_next_cam_datum(inp):
            idx, cam_datum = inp
            targ_data_f = osp.join(self.dst_next_cam_dir, f"{idx}.json")
            with open(targ_data_f, "w") as f:
                json.dump(cam_datum, f, indent=2)
        
        parallel_map(save_prev_cam_datum, list(zip(idxs, undist_prev_cams)), show_pbar=True, desc="Updating and saving previous camera data")
        parallel_map(save_next_cam_datum, list(zip(idxs, undist_next_cams)), show_pbar=True, desc="Updating and saving next camera data")

        

    def reformat(self):
        self.undistort_and_save()
        self.update_intrxs_and_save()
        self.cp_other_files()


        


def main(src_dir, targ_dir):
    colcam_src_dir = osp.join(src_dir, "colcam_set")
    colcam_targ_dir = osp.join(targ_dir, "colcam_set")

    ecam_src_dir = osp.join(src_dir, "ecam_set")
    ecam_targ_dir = osp.join(targ_dir, "ecam_set")

    col_undistorter = ColUndistorter(colcam_src_dir, colcam_targ_dir)
    col_undistorter.reformat()

    evs_uncistorter = EcamUndistorter(ecam_src_dir, ecam_targ_dir)
    evs_uncistorter.reformat()

    

if __name__ == "__main__":
    src_dir = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/depth_var_1_lr_000000"
    targ_dir = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/depth_var_1_lr_000000_undist"

    main(src_dir, targ_dir)