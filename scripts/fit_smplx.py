#!/usr/bin/env python3
"""
SMPL-X Fitting from 2D Keypoints
Optimizes SMPL-X shape (beta) and pose (theta) to match detected 2D keypoints.

Method: Differentiable optimization using PyTorch
- Loss: 2D reprojection error (weighted by keypoint visibility)
- Prior: L2 regularization on shape (beta) and pose (theta)
- Camera: Weak perspective model (scale + translation)
"""
import torch, numpy as np, os, json, cv2
from pathlib import Path

BASE = Path("/mnt/d/HERMES/study/lesserafim-thigh-paper")
SMPLX_DIR = str(BASE / "blade" / "body_models" / "smplx")

import smplx

# SMPL-X joint indices corresponding to MediaPipe Pose landmarks
# MediaPipe Pose has 33 landmarks
# SMPL-X has 55 joints (22 body + 30 hands + 1 face + 2 extra)
# We need a mapping from MediaPipe to SMPL-X

# MediaPipe Pose keypoint names (indices):
MP_JOINTS = {
    'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_pinky': 17, 'right_pinky': 18, 'left_index': 19, 'right_index': 20,
    'left_thumb': 21, 'right_thumb': 22,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_heel': 29, 'right_heel': 30, 'left_foot': 31, 'right_foot': 32,
}

# SMPL-X joint names (body joints 0-21):
SMPLX_BODY_JOINTS = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
]

# Mapping from MediaPipe to SMPL-X body joints (MP index -> SMPL-X index)
# Based on semantic correspondence
MP_TO_SMPLX = {
    MP_JOINTS['left_hip']: 1,      # SMPL-X left_hip
    MP_JOINTS['right_hip']: 2,     # SMPL-X right_hip  
    MP_JOINTS['left_knee']: 4,     # SMPL-X left_knee
    MP_JOINTS['right_knee']: 5,    # SMPL-X right_knee
    MP_JOINTS['left_ankle']: 7,    # SMPL-X left_ankle
    MP_JOINTS['right_ankle']: 8,   # SMPL-X right_ankle
    MP_JOINTS['left_shoulder']: 16, # SMPL-X left_shoulder
    MP_JOINTS['right_shoulder']: 17,# SMPL-X right_shoulder
    MP_JOINTS['left_elbow']: 18,   # SMPL-X left_elbow
    MP_JOINTS['right_elbow']: 19,  # SMPL-X right_elbow
    MP_JOINTS['left_wrist']: 20,   # SMPL-X left_wrist
    MP_JOINTS['right_wrist']: 21,  # SMPL-X right_wrist
    MP_JOINTS['nose']: 15,        # SMPL-X head (approximate)
}

def load_smplx(gender='neutral', device='cuda'):
    """Load SMPL-X model."""
    model = smplx.SMPLX(
        model_path=SMPLX_DIR,
        gender=gender,
        use_pca=False,
        flat_hand_mean=True,
        use_face_contour=True,
        num_betas=10,
        num_expression_coeffs=10,
    ).to(device)
    model.eval()
    return model

def fit_smplx(image_path, device='cuda', n_iter=200, lr=0.1):
    """
    Fit SMPL-X to 2D keypoints from a single image.
    
    Returns: (vertices, joints_3d, camera_params, losses)
    """
    # 1. Get image and detect pose
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    # Detect pose using our mediapipe code
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    pose_model = str(BASE / "models" / "pose_landmarker_lite.task")
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=pose_model),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
    )
    detector = vision.PoseLandmarker.create_from_options(pose_opts)
    
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_img)
    
    if not result.pose_landmarks:
        print("  No pose detected")
        return None
    
    kps = result.pose_landmarks[0]
    
    # 2. Extract 2D keypoints with visibility weights
    kp_2d = torch.zeros((33, 2), device=device)
    kp_conf = torch.zeros((33,), device=device)
    for i in range(min(len(kps), 33)):
        kp_2d[i, 0] = kps[i].x * w
        kp_2d[i, 1] = kps[i].y * h
        kp_conf[i] = getattr(kps[i], 'visibility', 1.0)
    
    # 3. Setup SMPL-X
    body_model = load_smplx(device=device)
    
    # 4. Optimize
    betas = torch.zeros((1, 10), device=device, requires_grad=True)
    body_pose = torch.zeros((1, 63), device=device, requires_grad=True)
    global_orient = torch.zeros((1, 3), device=device, requires_grad=True)
    
    # Camera: scale + translation (weak perspective)
    init_scale = max(w, h) * 0.45
    cam_scale = torch.tensor([[init_scale]], device=device, requires_grad=True)
    # Initialize translation to image center
    cam_trans = torch.tensor([[w / 2, h / 2]], device=device, requires_grad=True)
    
    # Initialize pose to T-pose for better starting point
    with torch.no_grad():
        body_pose.data.zero_()
        global_orient.data.zero_()
    
    optimizer = torch.optim.Adam([betas, body_pose, global_orient, cam_scale, cam_trans], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    losses = []
    for step in range(n_iter):
        optimizer.zero_grad()
        
        # Forward
        output = body_model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            return_verts=True,
        )
        
        joints = output.joints[0]  # (55, 3) SMPL-X joints
        
        # Build corresponding keypoints for matched joints
        mp_indices = sorted(MP_TO_SMPLX.keys())
        smplx_indices = [MP_TO_SMPLX[mp_i] for mp_i in mp_indices]
        
        # Get predicted 3D joints
        pred_joints_3d = joints[smplx_indices]  # (N, 3)
        
        # Weak perspective projection
        pred_2d = pred_joints_3d[:, :2] * cam_scale + cam_trans
        
        # Get ground truth 2D
        gt_2d = kp_2d[mp_indices]
        conf = kp_conf[mp_indices].clamp(min=0.1)  # weight floor
        
        # Reprojection loss (weighted)
        reproj_loss = (conf * torch.sum((pred_2d - gt_2d) ** 2, dim=1)).mean()
        
        # Regularization
        beta_reg = 0.001 * torch.sum(betas ** 2)
        pose_reg = 0.0001 * torch.sum(body_pose ** 2)
        
        total_loss = reproj_loss + beta_reg + pose_reg
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)
        
        losses.append(total_loss.item())
        
        if step % 50 == 0:
            print(f"    Step {step}: loss={total_loss.item():.2f}, reproj={reproj_loss.item():.2f}")
    
    # 5. Extract final result
    with torch.no_grad():
        output = body_model(betas=betas, body_pose=body_pose, global_orient=global_orient, return_verts=True)
        vertices = output.vertices[0].cpu().numpy()  # (10475, 3)
        joints_3d = output.joints[0].cpu().numpy()
    
    print(f"  Fitting done: loss={losses[-1]:.2f}")
    
    return {
        "vertices": vertices,
        "joints_3d": joints_3d,
        "betas": betas.detach().cpu().numpy().tolist(),
        "camera": {"scale": cam_scale.item(), "tx": cam_trans[0,0].item(), "ty": cam_trans[0,1].item()},
        "losses": losses,
        "image_shape": (h, w),
    }

def measure_thigh(result):
    """Measure thigh circumference from fitted SMPL-X mesh."""
    if result is None:
        return None
    
    v = result["vertices"]
    
    # SMPL-X thigh vertex indices (approximate mid-thigh region)
    # Left thigh: vertices 3949-4130 (roughly 180 vertices)
    # Right thigh: vertices 4407-4588
    left_thigh_idx = list(range(3949, 4130))
    right_thigh_idx = list(range(4407, 4588))
    
    measurements = {}
    for side, idx in [("left", left_thigh_idx), ("right", right_thigh_idx)]:
        pts = v[idx]
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        
        # PCA to find cross-section
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[0]  # axis direction
        
        # Project onto plane perpendicular to axis
        proj = centered - np.outer(np.dot(centered, normal), normal)
        radii = np.linalg.norm(proj, axis=1)
        mean_radius = np.mean(radii)
        circumference = 2 * np.pi * mean_radius
        
        measurements[f"{side}_circumference"] = float(circumference)
    
    measurements["avg_circumference"] = (measurements["left_circumference"] + measurements["right_circumference"]) / 2
    
    return measurements

if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not img_path:
        # Test with a known image
        img_path = str(BASE / "data" / "candid_photos" / "crazy" / "kazuha" / "240901_Crazy_Inkigayo" / "img_001.jpeg")
    
    print(f"Fitting SMPL-X to: {img_path}")
    result = fit_smplx(img_path, device='cuda')
    
    if result:
        print(f"\nMeasurements:")
        meas = measure_thigh(result)
        if meas:
            for k, v in meas.items():
                print(f"  {k}: {v:.2f} cm")
        
        # Save result
        out_dir = BASE / "output" / "measurements" / "test"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_dir / "vertices.npy"), result["vertices"])
        with open(str(out_dir / "result.json"), "w") as f:
            json.dump({k: float(v) if isinstance(v, (np.floating,)) else v for k, v in meas.items()}, f)
        print(f"\nSaved to {out_dir}")
