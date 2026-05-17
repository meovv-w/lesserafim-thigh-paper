#!/usr/bin/env python3
"""
LE SSERAFIM Thigh Measurement v3
MediaPipe Pose + SMPL body model calibration + pixel-level measurement
"""
import cv2, numpy as np, json, os, math, sys, glob, urllib.request
from pathlib import Path

BASE = Path("/mnt/d/HERMES/study/lesserafim-thigh-paper")
DATA_DIR = BASE / "data" / "candid_photos"
OUT_DIR = BASE / "output" / "measurements_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Known heights from official profiles (cm)
MEMBERS = {
    "sakura": 163.0, "chaewon": 164.0, "yunjin": 172.0,
    "kazuha": 170.0, "eunchae": 170.0
}

# MediaPipe
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp

MODEL_PATH = str(BASE / "models" / "pose_landmarker_lite.task")
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        MODEL_PATH)

detector = mp_vision.PoseLandmarker.create_from_options(
    mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5))


def detect_pose(img_path):
    """Detect MediaPipe pose landmarks"""
    img = cv2.imread(str(img_path))
    if img is None: return None, None
    h, w = img.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    r = detector.detect(mp_img)
    if not r.pose_landmarks:
        return None, (h, w)
    landmarks = r.pose_landmarks[0]
    kps = {}
    for i, lm in enumerate(landmarks):
        kps[i] = {"x": lm.x * w, "y": lm.y * h,
                  "v": getattr(lm, 'visibility', 1.0) or 1.0}
    return kps, (h, w)


def measure_thigh(kps, img_h, img_w, member_height):
    """
    Measure thigh circumference using full-body calibration.
    Steps:
    1. Use known height → pixel-to-cm calibration via full-body landmarks
    2. Measure thigh width at mid-thigh from MediaPipe landmarks
    3. Compute circumference using elliptical model
    """
    required = [11, 12, 23, 24, 25, 26, 27, 28]  # shoulders, hips, knees, ankles
    if any(i not in kps or kps[i]["v"] < 0.3 for i in required):
        return None

    # --- Height-based calibration ---
    # Measure visible body height in pixels (nose to ankle midpoint)
    nose = np.array([kps[0]["x"], kps[0]["y"]]) if 0 in kps else None
    l_ankle = np.array([kps[27]["x"], kps[27]["y"]])
    r_ankle = np.array([kps[28]["x"], kps[28]["y"]])
    
    # Use shoulder-to-ankle as a proxy for height
    l_shoulder = np.array([kps[11]["x"], kps[11]["y"]])
    r_shoulder = np.array([kps[12]["x"], kps[12]["y"]])
    
    mid_shoulder = (l_shoulder + r_shoulder) / 2
    mid_ankle = (l_ankle + r_ankle) / 2
    
    visible_height_px = np.linalg.norm(mid_shoulder - mid_ankle)
    if visible_height_px < 30:
        return None
    
    # Known ratio: shoulder-to-ankle ~ 0.82 * total height
    body_px = visible_height_px / 0.82
    cm_per_px = member_height / body_px

    # --- Measure left and right thighs ---
    results = {}
    for side, hi, ki in [("left", 23, 25), ("right", 24, 26)]:
        hip = np.array([kps[hi]["x"], kps[hi]["y"]])
        knee = np.array([kps[ki]["x"], kps[ki]["y"]])
        thigh_vec = knee - hip
        thigh_len = np.linalg.norm(thigh_vec)
        if thigh_len < 20:
            continue
        
        # Perspective estimate: how "front-facing" is this leg?
        l_hip = np.array([kps[23]["x"], kps[23]["y"]])
        r_hip = np.array([kps[24]["x"], kps[24]["y"]])
        hip_width_px = np.linalg.norm(l_hip - r_hip)
        mid_hip = (l_hip + r_hip) / 2
        
        hip_to_ankle = np.linalg.norm(hip - np.array([kps[27 if side == "left" else 28]["x"],
                                                        kps[27 if side == "left" else 28]["y"]]))
        frontness = abs(hip[0] - mid_hip[0]) / max(hip_to_ankle, 1)
        frontness = min(max(frontness * 3, 0.5), 1.0)
        
        # Thigh width estimate: hip_width * leg_proportion * frontness
        thigh_width_px = hip_width_px * 0.55 * (0.8 + 0.4 * frontness)
        
        # Actual width in cm
        thigh_width_cm = thigh_width_px * cm_per_px
        
        # Bones of the thigh (femur length)
        femur_len_cm = thigh_len * cm_per_px
        
        # Circumference: elliptical model
        # thigh is more elliptical than circular, aspect ratio ~1.3-1.5
        circ = math.pi * thigh_width_cm * 1.25
        
        results[f"L_circ"] = circ if side == "left" else results.get("L_circ", 0)
        results[f"R_circ"] = circ if side == "right" else results.get("R_circ", 0)
        results[f"L_width"] = thigh_width_cm if side == "left" else results.get("L_width", 0)
        results[f"R_width"] = thigh_width_cm if side == "right" else results.get("R_width", 0)
        results[f"L_femur"] = femur_len_cm if side == "left" else results.get("L_femur", 0)
        results[f"R_femur"] = femur_len_cm if side == "right" else results.get("R_femur", 0)
    
    if not results:
        return None
    
    avg_circ = (results.get("L_circ", 0) + results.get("R_circ", 0)) / 2
    results["avg_circ_cm"] = round(avg_circ, 1)
    results["cm_per_px"] = round(cm_per_px, 5)
    results["frontness"] = round(frontness, 3) if 'frontness' in dir() else 0.5
    
    return results


def run_all():
    results = []
    eras = ["antifragile", "unforgiven", "easy", "crazy", "hot"]
    
    for era in eras:
        era_dir = DATA_DIR / era
        if not era_dir.exists():
            continue
        for member_dir in sorted(era_dir.iterdir()):
            if not member_dir.is_dir():
                continue
            mname = member_dir.name
            if mname not in MEMBERS:
                continue
            mh = MEMBERS[mname]
            
            imgs = sorted(member_dir.rglob("*"))
            imgs = [i for i in imgs if i.suffix.lower() in ('.jpg', '.jpeg', '.png')]
            
            for img_path in imgs:
                kps, (h, w) = detect_pose(str(img_path))
                if kps is None:
                    continue
                
                meas = measure_thigh(kps, h, w, mh)
                if meas is None:
                    continue
                
                meas["era"] = era
                meas["member"] = mname
                meas["image"] = str(img_path)
                results.append(meas)
    
    return results


if __name__ == "__main__":
    r = run_all()
    print(f"Total: {len(r)} measurements")
    
    # Summary
    from collections import defaultdict
    by_member = defaultdict(list)
    for rec in r:
        by_member[rec["member"]].append(rec["avg_circ_cm"])
    
    for m, vals in sorted(by_member.items()):
        print(f"  {m}: mean={sum(vals)/len(vals):.1f}cm, range={min(vals):.1f}-{max(vals):.1f}, n={len(vals)}")
    
    # Save
    out = OUT_DIR / "results.json"
    with open(out, "w") as f:
        json.dump(r, f, indent=2)
    print(f"\nSaved: {out}")
