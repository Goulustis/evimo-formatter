import numpy as np
from tqdm import tqdm
from ev_buffer import EventBuffer

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

def ev_to_eimg(x, y, p, e_thresh=0.15):
    """
    input:
        evs (np.array [type (t, x, y, p)]): events such that t in [t_st, t_st + time_delta]
        NOTE: 
            - Samsung Camera event polarity is opposite of prophesee
            - polarity is flipped from eventbuffer
    return:
        event_img (np.array): of shape (h, w)
    """
    h, w = 480, 640

    pos_p = p==1
    neg_p = p==0

    e_img = np.zeros((h,w), dtype=np.int32)
    
    # np.add.at(e_img, (y[pos_p], x[pos_p]), -1)
    # np.add.at(e_img, (y[neg_p], x[neg_p]), 1)
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


# def create_deimgs(events:EventBuffer, triggers=None, time_delta=0.005, create_imgs = True, st_t=0):
