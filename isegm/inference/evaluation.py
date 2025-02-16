from time import time

import numpy as np
import torch

from isegm.inference import utils
from isegm.inference.clicker import Clicker

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        _, sample_ious, _ = evaluate_sample(sample.image, sample.gt_mask, predictor,
                                            sample_id=index, **kwargs)
        all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time


def evaluate_sample(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None):
    if gt_mask is not None:
        print(f"gt_mask.shape = {gt_mask.shape}")
    
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []

    with torch.no_grad():
        print(f"image.shape = {image.shape}")
        print(f"image.dtype = {image.dtype}")
        predictor.set_input_image(image)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr

            if callback is not None:
                callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs


def evaluate_sample_reefnet(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None, clicks=None, prev_mask=None):
    clicker = Clicker(gt_mask=gt_mask) # gt_mask: None
    with torch.no_grad():
        predictor.set_input_image(image)
        
        ## Limit number of clicks
        if max_clicks is not None:
            max_pos_clicks, max_neg_clicks = max_clicks//2, max_clicks-(max_clicks//2)
            if max_pos_clicks > 0 and len(clicks['foreground']) > max_pos_clicks:
                step = max(1, len(clicks['foreground']) // max_pos_clicks)  # Prevent step=0
                clicks['foreground'] = clicks['foreground'][::step]
            if max_neg_clicks > 0 and len(clicks['background']) > max_neg_clicks:
                step = max(1, len(clicks['background']) // max_neg_clicks)  # Prevent step=0    
                clicks['background'] = clicks['background'][::step]
        num_clicks = 0
        for point in clicks['foreground']:
            clicker.make_next_click_reefnet(point[1], point[0], True) # y, x, positive_negative
            num_clicks+=1
            if max_clicks is not None and num_clicks>=max_pos_clicks:
                break
        num_clicks = 0
        for point in clicks['background']:
            clicker.make_next_click_reefnet(point[1], point[0], False)
            num_clicks+=1
            if max_clicks is not None and num_clicks>=max_neg_clicks:
                break
        
        pred_probs = predictor.get_prediction(clicker, prev_mask=prev_mask)
        pred_mask = pred_probs > pred_thr
    return clicker.clicks_list, pred_probs, pred_mask
