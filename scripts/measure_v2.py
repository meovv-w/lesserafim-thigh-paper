#!/usr/bin/env python3
"""
LE SSERAFIM Thigh Measurement Pipeline v2.1
MediaPipe Pose + calibration using known height
"""
import cv2, numpy as np, json, os, math
from pathlib import Path

BASE = Path("/mnt/d/HERMES/study/lesserafim-thigh-paper")
DATA_DIR = BASE / "data"
OUTPUT_DIR = BASE / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MEMBERS = {"sakura":{"h":163.0,"w":44.0},"chaewon":{"h":163.9,"w":42.0},
           "yunjin":{"h":172.0,"w":53.0},"kazuha":{"h":170.0,"w":50.0},
           "eunchae":{"h":170.0,"w":48.0}}
ERAS = ["antifragile","unforgiven","easy","crazy","hot"]

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
POSE_MODEL = str(BASE/"models"/"pose_landmarker_lite.task")
if not os.path.exists(POSE_MODEL):
    os.makedirs(os.path.dirname(POSE_MODEL), exist_ok=True)
    import urllib.request
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        POSE_MODEL)
detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
    running_mode=vision.RunningMode.IMAGE, min_pose_detection_confidence=0.5))

def get_pose(img_path):
    img = cv2.imread(str(img_path))
    if img is None: return None
    h,w = img.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    r = detector.detect(mp_img)
    if not r.pose_landmarks: return None
    kps = {i:{"x":lm.x*w,"y":lm.y*h,"v":getattr(lm,'visibility',1.0)}
           for i,lm in enumerate(r.pose_landmarks[0])}
    return {"pts":kps,"h":h,"w":w}

def measure(pose_data, member_h):
    if not pose_data: return None
    p = pose_data["pts"]
    req = [23,24,25,26,27,28]  # hips, knees, ankles
    if any(i not in p or p[i]["v"]<0.3 for i in req): return None
    
    # Scale calibration
    left_leg = math.sqrt((p[23]["x"]-p[27]["x"])**2+(p[23]["y"]-p[27]["y"])**2)
    right_leg = math.sqrt((p[24]["x"]-p[28]["x"])**2+(p[24]["y"]-p[28]["y"])**2)
    avg_leg_px = (left_leg+right_leg)/2
    if avg_leg_px < 10: return None
    leg_cm = 0.53*member_h
    cm_per_px = leg_cm/avg_leg_px
    
    hip_width = math.sqrt((p[23]["x"]-p[24]["x"])**2+(p[23]["y"]-p[24]["y"])**2)
    
    m = {}
    for side,hi,ki in [("left",23,25),("right",24,26)]:
        hip = np.array([p[hi]["x"],p[hi]["y"]])
        knee = np.array([p[ki]["x"],p[ki]["y"]])
        
        thigh_vec = knee-hip
        thigh_len = np.linalg.norm(thigh_vec)
        if thigh_len<20: continue
        
        # Measure thigh width at multiple cross-sections
        # Width ≈ distance between left/right leg boundaries
        # Approximated as: width = inter-hip-distance * leg_separation_ratio
        leg_sep_ratio = (p[23]["x"]-p[24]["x"])/max(hip_width,1)
        thigh_width_px = 0.55*hip_width*(1+abs(leg_sep_ratio-1)*0.5)
        
        thigh_width_cm = thigh_width_px*cm_per_px
        circ = math.pi*thigh_width_cm*1.15
        bone_cm = thigh_len*cm_per_px
        
        m[f"{side}_circ_cm"] = round(circ,1)
        m[f"{side}_width_cm"] = round(thigh_width_cm,1)
        m[f"{side}_leg_cm"] = round(bone_cm,1)
    
    if not m: return None
    circs = [v for k,v in m.items() if k.endswith("_circ_cm")]
    m["avg_circ_cm"] = round(float(np.mean(circs)),1)
    m["cm_per_px"] = round(cm_per_px,5)
    return m

def run_all():
    results = []
    for era in ERAS:
        ed = DATA_DIR/"candid_photos"/era
        if not ed.exists(): continue
        for md in sorted(ed.iterdir()):
            if not md.is_dir() or md.name not in MEMBERS: continue
            mname = md.name
            for img in md.rglob("*"):
                if img.suffix.lower() not in ('.jpg','.jpeg','.png'): continue
                if "_pose" in img.name: continue
                pose = get_pose(str(img))
                if not pose: continue
                meas = measure(pose, MEMBERS[mname]["h"])
                if meas:
                    meas["era"]=era; meas["member"]=mname; meas["image"]=str(img)
                    results.append(meas)
    
    return results

if __name__ == "__main__":
    import sys; sys.path.insert(0,"..")
    r = run_all()
    print(f"Results: {len(r)}")
    for res in r[:3]:
        print(f"  {res['era']}/{res['member']}: {res['avg_circ_cm']} cm")
    
    # Save
    out = OUTPUT_DIR/"measurements"/"thigh_results.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    with open(out,"w") as f: json.dump(r,f,indent=2)
    print(f"Saved: {out}")
