#!/usr/bin/env python3
"""Pose Detection Test - MediaPipe 0.10 Tasks API"""
import cv2, numpy as np, os
from pathlib import Path

BASE = Path("/mnt/d/HERMES/study/lesserafim-thigh-paper")
DATA_DIR = BASE / "data"

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

POSE_MODEL = str(BASE / "models" / "pose_landmarker_lite.task")
if not os.path.exists(POSE_MODEL):
    os.makedirs(os.path.dirname(POSE_MODEL), exist_ok=True)
    import urllib.request
    url = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
           "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
    urllib.request.urlretrieve(url, POSE_MODEL)

pose_opts = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
    running_mode=vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.5,
)
detector = vision.PoseLandmarker.create_from_options(pose_opts)

# MediaPipe Pose connections for drawing
POSE_CONNECTIONS = frozenset([
    (11,12),(11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(27,31),(28,30),(28,32),(29,31),(30,32),
    (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),(12,14),(14,16),
    (16,18),(16,20),(16,22),(18,20),(3,4),(0,1),(0,2),(1,3),(2,4),
])

def test(img_path):
    img = cv2.imread(str(img_path))
    if img is None: return print("  No image")
    h, w = img.shape[:2]
    
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_img)
    
    if not result.pose_landmarks:
        return print("  No pose detected")
    
    kps = result.pose_landmarks[0]
    leg = {"L_hip":23,"R_hip":24,"L_knee":25,"R_knee":26,"L_ankle":27,"R_ankle":28}
    print(f"  {w}x{h}, {len(kps)} keypoints")
    for name, idx in leg.items():
        if idx < len(kps):
            print(f"    {name}: ({kps[idx].x*w:.0f},{kps[idx].y*h:.0f}) v={kps[idx].visibility:.2f}")
    
    # Draw keypoints with OpenCV
    annotated = img.copy()
    pts = [(int(k.x*w), int(k.y*h)) for k in kps if hasattr(k, 'x') and len(kps) > 0]
    
    # Draw connections
    for (i,j) in POSE_CONNECTIONS:
        if i < len(pts) and j < len(pts):
            cv2.line(annotated, pts[i], pts[j], (0,255,0), 2)
    
    # Draw keypoints
    for pt in pts:
        cv2.circle(annotated, pt, 4, (0,0,255), -1)
    
    out = str(img_path).rsplit(".",1)[0]+"_pose.jpg"
    cv2.imwrite(out, annotated)
    print(f"  Saved: {out}")

if __name__ == "__main__":
    for path in [
        DATA_DIR/"candid_photos"/"crazy"/"kazuha"/"240901_Crazy_Inkigayo"/"img_001.jpeg",
        DATA_DIR/"candid_photos"/"crazy"/"group"/"240905_Crazy_Mcountdown"/"img_001.jpeg",
    ]:
        if path.exists():
            print(f"\nTesting: {path}")
            test(str(path))
