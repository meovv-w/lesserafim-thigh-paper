#!/usr/bin/env python3
"""Debug SMPL thigh measurement - find correct vertex slicing"""
import torch, numpy as np, sys, json

BLADE_ROOT = "/mnt/d/HERMES/study/lesserafim-thigh-paper/blade"
sys.path.insert(0, BLADE_ROOT)

from blade.models.body_models.builder import build_body_model
body_cfg = dict(type='SMPL', keypoint_src='smpl_54', keypoint_dst='smpl_54',
                model_path=f'{BLADE_ROOT}/body_models/smpl', keypoint_approximate=True)
smpl = build_body_model(body_cfg)
smpl.eval()

# Neutral pose
betas = torch.zeros(1, 10)
body_pose = torch.zeros(1, 69)
global_orient = torch.zeros(1, 3)
transl = torch.zeros(1, 3)

out = smpl(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
verts = out['vertices'][0].cpu().numpy()

print(f"Vertices shape: {verts.shape}")
print(f"X range: {verts[:,0].min():.3f} to {verts[:,0].max():.3f}")
print(f"Y range: {verts[:,1].min():.3f} to {verts[:,1].max():.3f}")
print(f"Z range: {verts[:,2].min():.3f} to {verts[:,2].max():.3f}")

# Find mid-thigh height
y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
print(f"\nBody height: {y_max - y_min:.3f}")

# Try different fractions
for frac in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6]:
    target_y = y_min + (y_max - y_min) * frac
    eps = 0.02
    mask = (verts[:, 1] > target_y - eps) & (verts[:, 1] < target_y + eps)
    left = verts[mask & (verts[:, 0] < 0)]
    right = verts[mask & (verts[:, 0] > 0)]
    print(f"  frac={frac:.2f}, target_y={target_y:.3f}: left={len(left)} verts, right={len(right)} verts")

# Print some sample vertices around mid-thigh
target_y = y_min + (y_max - y_min) * 0.45
eps = 0.02
mask = (verts[:, 1] > target_y - eps) & (verts[:, 1] < target_y + eps)
candidates = verts[mask]
print(f"\nSample vertices near mid-thigh ({target_y:.3f}):")
for v in candidates[:10]:
    print(f"  x={v[0]:.4f}, y={v[1]:.4f}, z={v[2]:.4f}")

# The issue: right side has verts, left side might not.
# Let's find verts with x < -0.05 (left side of body)
left_verts = verts[verts[:, 0] < -0.05]
right_verts = verts[verts[:, 0] > 0.05]
print(f"\nLeft half: {len(left_verts)} verts, Right half: {len(right_verts)} verts")
print(f"Left Y range: {left_verts[:,1].min():.3f} to {left_verts[:,1].max():.3f}")
print(f"Right Y range: {right_verts[:,1].min():.3f} to {right_verts[:,1].max():.3f}")

# Use SMPL convention: left leg has similar Y range to right
# The issue is the left leg is at more negative X, so we need to filter differently
left_v = verts[verts[:, 0] < -0.02]
right_v = verts[verts[:, 0] > 0.02]

# At mid-thigh Y, count verts on each side
for frac in [0.3, 0.35, 0.4, 0.45, 0.5]:
    ty = y_min + (y_max - y_min) * frac
    eps = 0.03
    in_slice = (verts[:, 1] > ty - eps) & (verts[:, 1] < ty + eps)
    l = verts[in_slice & (verts[:, 0] < -0.02)]
    r = verts[in_slice & (verts[:, 0] > 0.02)]
    if len(l) > 0 and len(r) > 0:
        def peri(pts):
            if len(pts) < 3: return 0
            cx, cz = pts.mean(axis=0)
            ang = np.arctan2(pts[:,2]-cz, pts[:,0]-cx)
            s = pts[np.argsort(ang)][:,[0,2]]
            d = np.diff(s, axis=0)
            p = np.sum(np.sqrt((d**2).sum(axis=1)))
            p += np.sqrt(((s[0]-s[-1])**2).sum())
            return p
        print(f"  frac={frac:.2f}: L={len(l)} R={len(r)} | perim L={peri(l):.3f} R={peri(r):.3f}")
