from isegm.utils import vis, exp
from isegm.inference import utils
from isegm.inference.evaluation import evaluate_dataset, evaluate_sample, evaluate_sample_reefnet
from isegm.inference.predictors import get_predictor
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

if not hasattr(np, 'int'):
    np.int = int
    np.bool = bool

device = torch.device('cpu')
cfg = exp.load_config_file('./isegm/config.yml', return_edict=True)
checkpoint_path = './isegm/ritm_corals.pth'
MODEL_THRESH=0.6
brs_mode = 'NoBRS'
EVAL_MAX_CLICKS = None

checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, checkpoint_path)
image_path = "../ocean_fish.jpg"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.imread("./object_1_mask_20250216_152747.png", cv2.IMREAD_UNCHANGED)

# If the mask has multiple channels and you need a single-channel mask,
# you can convert it to grayscale (if applicable)
if mask is not None:
    if mask.ndim == 3:
        # For example, if mask is BGRA or BGR, convert to grayscale:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask
    print("Mask shape:", mask_gray.shape)
else:
    print("Failed to load mask.png")
    
# Ensure mask is binary 0/1
print(np.unique(mask_gray, return_counts=True))
mask_gray = mask_gray / np.max(mask_gray)
print(np.unique(mask_gray, return_counts=True))

prev_mask = mask_gray.astype(np.float32)
prev_mask = torch.tensor(prev_mask, device=device).unsqueeze(0).unsqueeze(0)

print(image.shape, prev_mask.shape)

points = {
    # Positive points
    "foreground": [
            # [x,y]
            [1420.0, 692.0],
        ],
    # Negative Points
    "background": 
        [
            # # [x,y]
            # [0.0, 0.0],
            # [0.0, 1.0],
        ], 
}

model = utils.load_is_model(checkpoint_path, device)

predictor = get_predictor(model, brs_mode, device, prob_thresh=MODEL_THRESH)
clicks_list, pred, pred_mask = evaluate_sample_reefnet(image, None, predictor, 
                        pred_thr=MODEL_THRESH, 
                        max_iou_thr=None, max_clicks=EVAL_MAX_CLICKS, clicks=points, prev_mask=prev_mask)

print(np.unique(pred_mask, return_counts=True))
# draw = vis.draw_with_blend_and_clicks(image, mask=pred_mask, clicks_list=clicks_list, alpha=0.5)

plt.figure(figsize=(40, 60))
plt.imshow(pred_mask)
plt.show()