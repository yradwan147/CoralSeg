# server.py
from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import torch
import datetime

# For backward compatibility with newer numpy versions (optional)
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool

# Imports for isegm (RITM)
from isegm.utils import exp
from isegm.inference import utils
from isegm.inference.evaluation import evaluate_sample_reefnet
from isegm.inference.predictors import get_predictor

# Try to load SAM components (if available)
try:
    from segment_anything import sam_model_registry, SamPredictor
except Exception as e:
    print("Error importing SAM:", e)
    sam_model_registry = None
    SamPredictor = None

app = Flask(__name__)

# Global settings
MODEL_THRESH = 0.6
EVAL_MAX_CLICKS = None
brs_mode = 'NoBRS'
device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load SAM model ---
if sam_model_registry is not None and SamPredictor is not None:
    try:
        sam_checkpoint = "../sam_vit_h_4b8939.pth"  # UPDATE PATH
        model_type = "vit_h"  # or "vit_l", "vit_b"
        sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam_model.to(device=device)
        sam_predictor = SamPredictor(sam_model)
    except Exception as e:
        print("Error loading SAM model:", e)
        sam_predictor = None
else:
    sam_predictor = None

# --- Load RITM predictor ---
try:
    cfg = exp.load_config_file('./isegm/config.yml', return_edict=True)
    ritm_checkpoint = './isegm/ritm_corals.pth'  # UPDATE PATH if needed
    ritm_checkpoint = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, ritm_checkpoint)
    ritm_model = utils.load_is_model(ritm_checkpoint, device)
    ritm_predictor = get_predictor(ritm_model, brs_mode, device, prob_thresh=MODEL_THRESH)
except Exception as e:
    print("Error loading RITM predictor:", e)
    ritm_predictor = None

def decode_image(b64_str):
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def encode_image(image, fmt='.png'):
    success, encoded_image = cv2.imencode(fmt, image)
    if not success:
        return None
    return base64.b64encode(encoded_image.tobytes()).decode('utf-8')

@app.route("/run_sam", methods=["POST"])
def run_sam():
    print("Received SAM request at", datetime.datetime.now())
    data = request.get_json()
    image_b64 = data.get("image")
    points = data.get("points")
    if image_b64 is None or points is None:
        return jsonify({"error": "Missing image or points"}), 400

    img_bgr = decode_image(image_b64)
    if img_bgr is None:
        return jsonify({"error": "Failed to decode image"}), 400
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if sam_predictor is None:
        return jsonify({"error": "SAM predictor not available"}), 500

    sam_predictor.set_image(image_rgb)
    input_points = np.array(points)
    input_labels = np.ones(input_points.shape[0], dtype=int)
    try:
        masks, scores, logits = sam_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    mask = masks[0].astype(np.uint8)
    # For visualization, convert mask to 0/255
    mask_img = mask * 255
    mask_b64 = encode_image(mask_img, fmt='.png')
    return jsonify({"mask": mask_b64})

@app.route("/run_ritm", methods=["POST"])
def run_ritm():
    print("Received RITM request at", datetime.datetime.now())
    data = request.get_json()
    image_b64 = data.get("image")
    prev_mask_b64 = data.get("prev_mask")
    points = data.get("points")
    if image_b64 is None or prev_mask_b64 is None or points is None:
        return jsonify({"error": "Missing required data"}), 400

    img_bgr = decode_image(image_b64)
    if img_bgr is None:
        return jsonify({"error": "Failed to decode image"}), 400
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    prev_mask_img = decode_image(prev_mask_b64)
    if prev_mask_img is None:
        return jsonify({"error": "Failed to decode prev_mask"}), 400
    if len(prev_mask_img.shape) == 3:
        prev_mask_gray = cv2.cvtColor(prev_mask_img, cv2.COLOR_BGR2GRAY)
    else:
        prev_mask_gray = prev_mask_img
    prev_mask_gray = (prev_mask_gray > 128).astype(np.float32)
    prev_mask_tensor = torch.tensor(prev_mask_gray, device=device).unsqueeze(0).unsqueeze(0)

    if ritm_predictor is None:
        return jsonify({"error": "RITM predictor not available"}), 500

    try:
        clicks_list, pred, pred_mask = evaluate_sample_reefnet(
            image_rgb,
            None,
            ritm_predictor,
            pred_thr=MODEL_THRESH,
            max_iou_thr=None,
            max_clicks=EVAL_MAX_CLICKS,
            clicks=points,
            prev_mask=prev_mask_tensor
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    mask = pred_mask.astype(np.uint8)
    mask_img = mask * 255
    mask_b64 = encode_image(mask_img, fmt='.png')
    return jsonify({"mask": mask_b64})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
