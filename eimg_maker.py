import numpy as np
from tqdm import tqdm
import os.path as osp
import bisect
import cv2

from ev_buffer import EventBuffer
from make_dataset_utils import load_json

def eimg_to_img(eimg, col=True):

    if col:
        img = np.zeros((*eimg.shape[:2],3), dtype=np.uint8)
        img[eimg < 0,0] = 255
        img[eimg > 0,1] = 255
    else:
        img = np.zeros(eimg.shape[:2], dtype=np.uint8)
        img[eimg < 0] = 255
        img[eimg > 0] = 255
    return img


def ev_to_eimg(x, y, p, img_size = None):
    """
    input:
        evs (np.array [type (t, x, y, p)]): events such that t in [t_st, t_st + time_delta]
        img_size (tuple [int, int]): image size in (h,w)
    return:
        event_img (np.array): of shape (h, w)
    """


    if img_size is None:
        h, w = 720, 1280
    else:
        h, w = img_size

    pos_p = p==1
    neg_p = p==0

    e_img = np.zeros((h,w), dtype=np.int32)
    
    np.add.at(e_img, (y[pos_p], x[pos_p]), 1)
    np.add.at(e_img, (y[neg_p], x[neg_p]), -1)
    

    assert np.abs(e_img).max() < np.iinfo(np.int8).max, "type needs to be bigger"

    return e_img.astype(np.int8)


def create_event_imgs(events:EventBuffer, triggers=None, time_delta=0.005, create_imgs = True):
    """
    input:
        events (np.array [type (t, x, y, p)]): events
        triggers (np.array [int]): list of trigger time; will generate tight gap if none
        time_delta (int): time in ms, the time gap to create event images
        create_imgs (bool): actually create the event images, might use this function just to
                            get timestamps and ids

    return:
        eimgs (np.array): list of images with time delta of 50
        eimgs_ts (np.array): list of time at which the event image is accumulated to
        eimgs_ids (np.array): list of embedding ids for each image
        trigger_ids (np.array): list of embedding ids of each trigger time
    """
    if create_imgs:
        print("creating event images")
    else:
        print("not creating event images, interpolating cameras and creating ids only")


    eimgs = []       # the event images
    eimgs_ts = []    # timestamp of event images
    eimgs_ids = []   # embedding ids for each img
    trig_ids = []    # id at each trigger

    id_cnt = 0
    with tqdm(total=(len(triggers) - 1)) as pbar:
        for trig_idx in range(1, len(triggers)):
            trig_st, trig_end = triggers[trig_idx - 1], triggers[trig_idx]

            if (events is not None) and create_imgs:
                curr_t, curr_x, curr_y, curr_p = events.retrieve_data(trig_st, trig_end)
                if len(curr_t) == 0:
                    break
            
            if trig_st >= events.t_f[-1]:
                break

            st_t = trig_st
            end_t = trig_st + time_delta
            trig_ids.append(id_cnt)

            while st_t < trig_end:
                if (events is not None) and create_imgs:
                    cond = (st_t <= curr_t) & (curr_t <= end_t)
                    if cond.sum() == 0:
                        break

                    eimg = ev_to_eimg(curr_x[cond], curr_y[cond], curr_p[cond])
                    eimgs.append(eimg)

                eimgs_ids.append(id_cnt)
                eimgs_ts.append(st_t)
                # eimgs_ts.append(int((st_t + end_t)/2))

                # update
                st_t = end_t
                end_t = end_t + time_delta
                end_t = min(end_t, trig_end)
                id_cnt += 1

            pbar.update(1)

    if (events is not None) and create_imgs:
        return np.stack(eimgs), np.array(eimgs_ts, dtype=np.float32), np.array(eimgs_ids, dtype=np.int32), np.array(trig_ids, dtype=np.int32)
    else:
        return None, np.array(eimgs_ts), np.array(eimgs_ids, dtype=np.int32), np.array(trig_ids, dtype=np.int32)

def create_eimg_by_triggers(events, triggers, exposure_time = 0.005, make_eimg=True):
    eimgs = np.zeros((len(triggers), 480, 640), dtype=np.int8)
    eimg_ts = []
    for i, trigger in tqdm(enumerate(triggers), total=len(triggers), desc="making ev imgs"):
        st_t, end_t = max(trigger - exposure_time/2, 0), trigger + exposure_time/2
        eimg_ts.append(st_t + exposure_time/2)

        if make_eimg:
            curr_t, curr_x, curr_y, curr_p = events.retrieve_data(st_t, end_t)

            eimg = ev_to_eimg(curr_x, curr_y, curr_p)
            eimgs[i] = eimg
            
            events.drop_cache_by_t(st_t)
    
    dummy = np.array(list(range(len(eimg_ts))))
    return eimgs, np.array(eimg_ts), dummy, dummy


class DeimgsCreator:

    def __init__(self, events:EventBuffer, 
                       triggers,  reproj_npz, scales, 
                       biases, colcam_dir, ecam_dir, 
                       time_delta=0.005, create_imgs = True) -> None:
        self.logSafetyOffset = 90 # random number the Event threshold estimation algorithm uses 

        self.events = events
        self.triggers = triggers 
        self.reproj_npz = reproj_npz
        self.scales = scales 
        self.biases = biases
        self.time_delta = time_delta
        self.create_imgs = create_imgs
        self.colcam_dir = colcam_dir
        self.ecam_dir = ecam_dir

        # make threshes
        self.pos_threshes = (self.scales + self.biases)
        self.neg_threshes = (-self.scales + self.biases)

        self._filter_test_valid_keys()
        self._prep_appearence_id()
    
    def _prep_appearence_id(self):
        ecam_meta = load_json(osp.join(self.ecam_dir, "metadata.json"))
        keys = sorted(list(ecam_meta.keys()))
        self.ecam_apperance_ids = np.array([ecam_meta[key]["appearance_id"] for key in keys])
        self.ecam_ts = np.array([ecam_meta[key]["t"] for key in keys])
    

    def find_closest_index(self, nums, target):
        # Get the position where 'target' should be inserted to keep the list sorted
        pos = bisect.bisect_left(nums, target)

        # If 'pos' is 0, the target is less than the first element
        if pos == 0:
            return 0
        # If 'pos' is equal to the length of the list, target is greater than any element in the list
        if pos == len(nums):
            return len(nums) - 1

        # Check if the target is closer to the previous element or the element at 'pos'
        if abs(target - nums[pos - 1]) <= abs(nums[pos] - target):
            return pos - 1
        else:
            return pos

    def get_close_apprce_id(self, t):
        idx = self.find_closest_index(self.ecam_ts, t)
        return self.ecam_apperance_ids[idx]
    
    def _filter_test_valid_keys(self):
        # filter out pixels that are in the test and valid images

        dataset_json_f = osp.join(self.colcam_dir, "dataset.json")
        dataset = load_json(dataset_json_f)
        clss2img_id = dict(zip(dataset["classical_ids"], dataset["ids"]))
        val_test_ids = dataset["val_ids"] + dataset["test_ids"]
        
        ## NOTE: drop first reproj frame because it is all black and not used for thresh estimation
        ## see thresh_est/prepare_for_thresh_est.py for detail
        reproj_clss_ids = sorted(self.reproj_npz["class_keys"])[1:]
        val_cond = np.array([not (clss2img_id.get(cls_id) in val_test_ids) for cls_id in reproj_clss_ids])

        all_keys = sorted(list(self.reproj_npz.keys()))
        reproj_img_keys = np.array([e for e in all_keys if ("classical" in e and (not "mask" in e))])[1:]
        reproj_msk_keys = np.array([e for e in all_keys if ("classical" in e and "mask" in e)])[1:]
        reproj_ts = self.reproj_npz["t"][1:]

        self.val_rprj_img_keys = reproj_img_keys[val_cond]
        self.val_rprj_msk_keys = reproj_msk_keys[val_cond]
        self.val_rprj_ts = reproj_ts[val_cond]
    
    
    def evs_to_deimgs(self, xs, ys, ps, prev_img, msk):
        pos_cond = ps == 1
        neg_cond = ps == 0

        pos_acc = ev_to_eimg(xs[pos_cond], ys[pos_cond], ps[pos_cond], img_size=prev_img.shape[:2]).astype(np.float32)
        neg_acc = ev_to_eimg(xs[neg_cond], ys[neg_cond], np.ones(sum(neg_cond)), img_size=prev_img.shape[:2]).astype(np.float32)

        pred_log = np.log(prev_img + self.logSafetyOffset) + pos_acc * self.pos_threshes + neg_acc * self.neg_threshes
        pred_img = np.exp(pred_log) - self.logSafetyOffset

        return pred_img*msk

    def get_img(self, idx):
        return cv2.cvtColor(self.reproj_npz[self.val_rprj_img_keys[idx]], cv2.COLOR_BGR2GRAY).astype(np.float32)

    def create_deimgs(self):
        
        ## stuff to save & return
        deimgs = []
        deimg_ids = []
        deimg_ts = []
        deimg_msks = []

        ## initial values
        reproj_idx = 0
        frame_cnt = 0
        prev_img = self.get_img(reproj_idx) #self.reproj_npz[self.val_rprj_img_keys[reproj_idx]]
        prev_msk = self.reproj_npz[self.val_rprj_msk_keys[reproj_idx]]

        with tqdm(total=(len(self.triggers) - 1)) as pbar:
            for trig_idx in range(1, len(self.triggers)):
                trig_st, trig_end = self.triggers[trig_idx - 1], self.triggers[trig_idx]

                if frame_cnt == 0:
                    st_t = self.val_rprj_ts[reproj_idx]
                    end_t = max(trig_st + self.time_delta, 
                                trig_st + np.ceil((st_t - trig_st)/self.time_delta)*self.time_delta)

                if self.create_imgs:
                    curr_t, curr_x, curr_y, curr_p = self.events.retrieve_data(st_t, trig_end)
                
                visited_end_trig = False
                while end_t <= trig_end:  
                    if self.create_imgs:
                        cond = (st_t <= curr_t) & (curr_t <= end_t)
                        if cond.sum() != 0:
                            deimg = self.evs_to_deimgs(curr_x[cond], curr_y[cond], curr_p[cond], prev_img, prev_msk)
                
                            # store stuff
                            deimgs.append(deimg)
                            deimg_ids.append(self.get_close_apprce_id(0.5*(st_t + end_t)))
                            deimg_ts.append(end_t)
                            deimg_msks.append(prev_msk)
                            frame_cnt += 1

                    # update
                    end_t = end_t + self.time_delta
                    end_t = min(end_t, trig_end) if not visited_end_trig else end_t
                    visited_end_trig = end_t == trig_end if not visited_end_trig else visited_end_trig

                    if (reproj_idx + 1) > len(self.val_rprj_ts) :
                        break

                    # update prev data
                    if end_t > self.val_rprj_ts[reproj_idx + 1]:
                        reproj_idx += 1
                        prev_img = self.get_img(reproj_idx) #self.reproj_npz[self.val_rprj_img_keys[reproj_idx]]
                        prev_msk = self.reproj_npz[self.val_rprj_msk_keys[reproj_idx]]
                        st_t = self.val_rprj_ts[reproj_idx]
                    
                pbar.update(1)
                if reproj_idx > len(self.val_rprj_ts):
                    break
        
        if self.create_imgs:
            # NOTE: no trig ids, since assume colcam set already made
            return np.stack(deimgs), np.array(deimg_ts, dtype=np.float32), np.array(deimg_ids, dtype=np.int32), np.stack(deimg_msks)
        else:
            return None, np.array(deimg_ts), np.array(deimg_ids, dtype=np.int32)
