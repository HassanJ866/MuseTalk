import sys
from face_detection import FaceAlignment, LandmarksType
import numpy as np
import cv2
import os
import torch
from tqdm import tqdm
import onnxruntime as ort

# ── DWPose via ONNX (replaces mmpose/mmdet/mmcv entirely) ────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
_providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cuda"
    else ["CPUExecutionProvider"]
)

_det_session  = ort.InferenceSession("./models/dwpose/yolox_l.onnx",  providers=_providers)
_pose_session = ort.InferenceSession("./models/dwpose/dw-ll_ucoco_384.onnx", providers=_providers)

coord_placeholder = (0.0, 0.0, 0.0, 0.0)

fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)


def _preprocess_det(img, input_size=(640, 640)):
    h, w = img.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized
    blob = padded.transpose(2, 0, 1)[None].astype(np.float32)
    return blob, scale


def _nms(boxes, scores, iou_thr=0.45):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return keep


def _detect_person(img):
    blob, scale = _preprocess_det(img)
    out = _det_session.run(None, {_det_session.get_inputs()[0].name: blob})[0][0]
    obj_conf = out[:, 4]
    scores = obj_conf * out[:, 5]   # class 0 = person
    mask = scores > 0.3
    if not mask.any():
        return None
    out, scores = out[mask], scores[mask]
    cx = out[:, 0] / scale
    cy = out[:, 1] / scale
    bw = out[:, 2] / scale
    bh = out[:, 3] / scale
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    keep = _nms(boxes, scores)
    return boxes[keep[0]].astype(int)


def _preprocess_pose(img, bbox, input_size=(384, 288)):
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None, None
    scale_x = input_size[1] / crop.shape[1]
    scale_y = input_size[0] / crop.shape[0]
    resized = cv2.resize(crop, (input_size[1], input_size[0]))
    mean = np.array([123.675, 116.28,  103.53],  dtype=np.float32)
    std  = np.array([58.395,   57.12,   57.375], dtype=np.float32)
    blob = ((resized.astype(np.float32) - mean) / std).transpose(2, 0, 1)[None]
    return blob, (x1, y1), (scale_x, scale_y)


def _get_keypoints(img):
    """Return (133, 2) keypoints in original image coords, or None."""
    h, w = img.shape[:2]
    bbox = _detect_person(img)
    if bbox is None:
        bbox = [0, 0, w, h]
    blob, offset, scale = _preprocess_pose(img, bbox)
    if blob is None:
        return None
    out = _pose_session.run(None, {_pose_session.get_inputs()[0].name: blob})
    if len(out) == 2 and out[0].ndim == 3:
        # SimCC output: [1, 133, W_simcc], [1, 133, H_simcc]
        kp_x = np.argmax(out[0][0], axis=1) / 2.0
        kp_y = np.argmax(out[1][0], axis=1) / 2.0
    else:
        # Heatmap output: [1, 133, H, W]
        hm = out[0][0]
        flat = hm.reshape(133, -1)
        idx = flat.argmax(1)
        kp_y = idx // hm.shape[2]
        kp_x = idx  % hm.shape[2]
        kp_x = kp_x / hm.shape[2] * (bbox[2] - bbox[0])
        kp_y = kp_y / hm.shape[1] * (bbox[3] - bbox[1])
    ox, oy = offset
    sx, sy = scale
    kp_x = kp_x / sx + ox
    kp_y = kp_y / sy + oy
    return np.stack([kp_x, kp_y], axis=1)


def read_imgs(img_list):
    frames = []
    print("reading images...")
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def get_landmark_and_bbox(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    coords_list = []
    if upperbondrange != 0:
        print("get key_landmark and face bounding boxes with the bbox_shift:", upperbondrange)
    else:
        print("get key_landmark and face bounding boxes with the default value")
    average_range_minus, average_range_plus = [], []
    for frame in tqdm(frames):
        kps = _get_keypoints(frame)
        if kps is None:
            coords_list.append(coord_placeholder)
            continue
        face_land_mark = kps[23:91].astype(np.int32)
        bbox = fa.get_detections_for_batch(np.asarray([frame]))
        for f in bbox:
            if f is None:
                coords_list.append(coord_placeholder)
                continue
            half_face_coord = face_land_mark[29].copy()
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus  = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange + half_face_coord[1]
            half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
            upper_bond = max(0, half_face_coord[1] - half_face_dist)
            f_landmark = (
                int(np.min(face_land_mark[:, 0])), int(upper_bond),
                int(np.max(face_land_mark[:, 0])), int(np.max(face_land_mark[:, 1]))
            )
            x1, y1, x2, y2 = f_landmark
            if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
                coords_list.append(f)
            else:
                coords_list.append(f_landmark)
    n = len(frames)
    rm = int(sum(average_range_minus) / len(average_range_minus)) if average_range_minus else 0
    rp = int(sum(average_range_plus)  / len(average_range_plus))  if average_range_plus  else 0
    print("*" * 60 + "bbox_shift parameter adjustment" + "*" * 60)
    print(f"Total frame:「{n}」 Manually adjust range : [ -{rm}~{rp} ] , the current value: {upperbondrange}")
    print("*" * 120)
    return coords_list, frames


def get_bbox_range(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    average_range_minus, average_range_plus = [], []
    for frame in tqdm(frames):
        kps = _get_keypoints(frame)
        if kps is None:
            continue
        face_land_mark = kps[23:91].astype(np.int32)
        bbox = fa.get_detections_for_batch(np.asarray([frame]))
        for f in bbox:
            if f is None:
                continue
            average_range_minus.append((face_land_mark[30] - face_land_mark[29])[1])
            average_range_plus.append( (face_land_mark[29] - face_land_mark[28])[1])
    n = len(frames)
    rm = int(sum(average_range_minus) / len(average_range_minus)) if average_range_minus else 0
    rp = int(sum(average_range_plus)  / len(average_range_plus))  if average_range_plus  else 0
    return f"Total frame:「{n}」 Manually adjust range : [ -{rm}~{rp} ] , the current value: {upperbondrange}"
