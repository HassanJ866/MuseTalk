import numpy as np
import cv2
import os
import torch
from tqdm import tqdm
from face_detection import FaceAlignment, LandmarksType

# face_alignment gives us both the face bbox AND 68-point landmarks (dlib order).
# We use it for everything — no DWPose/mmpose needed for the face crop.
device = "cuda" if torch.cuda.is_available() else "cpu"
fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)

coord_placeholder = (0.0, 0.0, 0.0, 0.0)


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
        # get_landmarks returns list of (68,2) arrays, one per face; None entries for no face
        lms = fa.get_landmarks(frame)
        bbox = fa.get_detections_for_batch(np.asarray([frame]))
        f = bbox[0] if bbox is not None and len(bbox) > 0 else None

        if lms is None or len(lms) == 0 or f is None:
            coords_list.append(coord_placeholder)
            continue

        # 68-point dlib landmarks for the first (largest) face
        face_land_mark = lms[0].astype(np.int32)  # (68, 2)

        # dlib landmark 33 = nose tip — natural upper/lower face split point.
        # Points 28-30 in dlib are the nose bridge, not relevant here.
        # MuseTalk logic: split face at nose tip (pt 33), use distance to chin as margin.
        half_face_coord = face_land_mark[33].copy()   # nose tip (x, y)

        # range_minus/plus: how far the split can move up/down
        range_minus = (face_land_mark[34] - face_land_mark[33])[1]
        range_plus  = (face_land_mark[33] - face_land_mark[32])[1]
        average_range_minus.append(range_minus)
        average_range_plus.append(range_plus)

        if upperbondrange != 0:
            half_face_coord[1] = upperbondrange + half_face_coord[1]

        # distance from nose tip down to chin = how far up we crop
        half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
        upper_bond = max(0, half_face_coord[1] - half_face_dist)

        f_landmark = (
            int(np.min(face_land_mark[:, 0])), int(upper_bond),
            int(np.max(face_land_mark[:, 0])), int(np.max(face_land_mark[:, 1]))
        )
        x1, y1, x2, y2 = f_landmark
        if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
            # fallback to raw face_alignment bbox
            fa_x1, fa_y1, fa_x2, fa_y2 = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            coords_list.append((fa_x1, fa_y1, fa_x2, fa_y2))
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
        lms = fa.get_landmarks(frame)
        if lms is None or len(lms) == 0:
            continue
        face_land_mark = lms[0].astype(np.int32)
        average_range_minus.append((face_land_mark[34] - face_land_mark[33])[1])
        average_range_plus.append( (face_land_mark[33] - face_land_mark[32])[1])
    n = len(frames)
    rm = int(sum(average_range_minus) / len(average_range_minus)) if average_range_minus else 0
    rp = int(sum(average_range_plus)  / len(average_range_plus))  if average_range_plus  else 0
    return f"Total frame:「{n}」 Manually adjust range : [ -{rm}~{rp} ] , the current value: {upperbondrange}"
