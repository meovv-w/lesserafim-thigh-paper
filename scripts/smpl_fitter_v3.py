#!/usr/bin/env python3
"""
SMPL Fitting Pipeline v3 - Batch run on all photos
Low beta regularization to capture individual body shapes
"""
import sys, os, json, glob, torch, cv2, numpy as np, urllib.request
from pathlib import Path

BLADE_ROOT = "/mnt/d/HERMES/study/lesserafim-thigh-paper/blade"
sys.path.insert(0, BLADE_ROOT)
os.environ["CUDA_HOME"] = os.path.expanduser("~/cuda-11.8")
os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{os.environ['CUDA_HOME']}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# 1. SMPL model
print("Loading SMPL...")
from blade.models.body_models.builder import build_body_model
smpl = build_body_model(dict(type='SMPL', keypoint_src='smpl_54', keypoint_dst='smpl_54',
    model_path=f'{BLADE_ROOT}/body_models/smpl', keypoint_approximate=True)).to(device)
smpl.eval()

# 2. MediaPipe
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp

MP_MODEL = "/tmp/pose_landmarker_lite.task"
if not os.path.exists(MP_MODEL):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task", MP_MODEL)
detector = mp_vision.PoseLandmarker.create_from_options(
    mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL),
        running_mode=mp_vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5))

# 3. Known heights
MEMBER_H = {"sakura": 163.0, "chaewon": 164.0, "yunjin": 172.0, "kazuha": 170.0, "eunchae": 170.0}

# 4. Thigh circumference from zero-pose mesh
def measure_thigh_from_mesh(verts):
    """verts: (6890,3) in canonical SMPL space (Y-up, zero pose)"""
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    slices = []
    for frac in [0.35, 0.40, 0.45, 0.50]:
        ty = y_min + (y_max - y_min) * frac
        eps = 0.04
        mask = (verts[:, 1] > ty - eps) & (verts[:, 1] < ty + eps)
        left = verts[mask & (verts[:, 0] < -0.02)][:, [0, 2]]
        right = verts[mask & (verts[:, 0] > 0.02)][:, [0, 2]]
        def peri(pts):
            if len(pts) < 3: return 0.0
            c = pts.mean(axis=0)
            ang = np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
            s = pts[np.argsort(ang)]
            d = np.diff(s, axis=0)
            return float(np.sum(np.sqrt((d**2).sum(axis=1))) + np.sqrt(((s[0]-s[-1])**2).sum()))
        lc, rc = peri(left), peri(right)
        if lc > 0 and rc > 0:
            slices.append((lc + rc) / 2)
    return float(np.mean(slices)) if slices else 0.0

# 5. Fitter with weak regularization
def fit_smpl(landmarks, W, H, member_h):
    """Fit SMPL params to 2D keypoints then measure from zero-pose mesh"""
    mp_ids = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    smpl_ids = [12, 9, 13, 10, 14, 11, 3, 0, 4, 1, 5, 2]
    
    # Keypoints
    kp, conf = [], []
    for idx in mp_ids:
        lm = landmarks[idx]
        kp.append([lm.x * W, lm.y * H])
        vis = lm.visibility if hasattr(lm, 'visibility') and lm.visibility else 0.9
        conf.append(vis if vis > 0.5 else 0.1)
    kp2d = torch.tensor(kp, device=device, dtype=torch.float32)
    kpc = torch.tensor(conf, device=device, dtype=torch.float32)
    smpl_ids_t = torch.tensor(smpl_ids, device=device)
    
    focal = max(W, H) * 1.2
    K = torch.tensor([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]], device=device, dtype=torch.float32)
    
    betas = torch.zeros(1, 10, device=device, requires_grad=True)
    body_pose = torch.zeros(1, 69, device=device, requires_grad=True)
    global_orient = torch.zeros(1, 3, device=device, requires_grad=True)
    transl = torch.zeros(1, 3, device=device, requires_grad=True)
    scale = torch.ones(1, device=device, requires_grad=True)
    
    opt = torch.optim.Adam([betas, body_pose, global_orient, transl, scale], lr=0.1)
    
    for i in range(250):
        opt.zero_grad()
        out = smpl(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
        j3d = out['joints'][0, smpl_ids_t]
        proj = j3d @ K.T
        p2d = proj[:, :2] / (proj[:, 2:3] + 1e-8) * scale
        loss_2d = (kpc * torch.norm(p2d - kp2d, dim=1)).mean()
        # Weak regularization: 1/100 of previous value
        loss_reg = 0.0001 * betas.norm() + 0.0005 * body_pose.norm() + 0.001 * transl.norm()
        loss = loss_2d + loss_reg
        loss.backward()
        opt.step()
    
    with torch.no_grad():
        # Get zero-pose mesh (shape only, reflects actual body shape)
        out_zero = smpl(betas=betas.detach(),
                        body_pose=torch.zeros(1, 69, device=device),
                        global_orient=torch.zeros(1, 3, device=device),
                        transl=torch.zeros(1, 3, device=device))
        zero_verts = out_zero['vertices'][0].cpu().numpy()
        
        # Get posed mesh for height measurement
        out_posed = smpl(betas=betas.detach(), body_pose=body_pose.detach(),
                         global_orient=global_orient.detach(), transl=transl.detach())
        posed_verts = out_posed['vertices'][0].cpu().numpy()
    
    # Measure thigh from zero-pose mesh
    circ_smpl = measure_thigh_from_mesh(zero_verts)  # SMPL meters
    
    # Height calibration: scale SMPL mesh to actual height
    smpl_height = zero_verts[:, 1].max() - zero_verts[:, 1].min()  # SMPL height (m)
    scale_factor = member_h / 100.0 / smpl_height  # actual height / SMPL height
    circ_cm = circ_smpl * 100 * scale_factor  # convert to cm
    
    return {
        'circ_cm': float(circ_cm),
        'circ_smpl': float(circ_smpl),
        'smpl_height': float(smpl_height),
        'scale_factor': float(scale_factor),
        'loss_2d': float(loss_2d),
        'loss_total': float(loss),
        'betas': betas.detach().cpu().numpy()[0].tolist(),
    }

# 6. Process all photos
BASE = Path("/mnt/d/HERMES/study/lesserafim-thigh-paper")
DATA = BASE / "data" / "candid_photos"
OUT = BASE / "output" / "measurements_smpl_v3"
OUT.mkdir(parents=True, exist_ok=True)

results = []
eras = ["antifragile", "unforgiven", "easy", "crazy", "hot"]

for era in eras:
    ed = DATA / era
    if not ed.exists(): continue
    for md in sorted(ed.iterdir()):
        if not md.is_dir() or md.name not in MEMBER_H: continue
        mname = md.name
        mh = MEMBER_H[mname]
        imgs = sorted([f for f in md.rglob("*") if f.suffix.lower() in ('.jpg','.jpeg','.png')])
        
        for img_path in imgs:
            print(f"[{era}/{mname}] {img_path.name}", end=" ")
            img = cv2.imread(str(img_path))
            if img is None: print("NO_IMG"); continue
            H, W = img.shape[:2]
            
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            res = detector.detect(mp_img)
            if not res.pose_landmarks: print("NO_POSE"); continue
            
            try:
                fit = fit_smpl(res.pose_landmarks[0], W, H, mh)
            except Exception as e:
                print(f"FAIL: {e}")
                continue
            
            if fit['loss_2d'] > 5000:
                print(f"BAD_FIT(loss={fit['loss_2d']:.0f})")
                continue
            
            print(f"{fit['circ_cm']:.1f}cm (loss={fit['loss_2d']:.0f})")
            
            results.append({
                'member': mname, 'era': era, 'image': str(img_path),
                'circ_cm': round(fit['circ_cm'], 1),
                'circ_smpl': round(fit['circ_smpl'], 4),
                'smpl_height': round(fit['smpl_height'], 3),
                'loss_2d': round(fit['loss_2d'], 1),
                'betas': fit['betas'][:5],
            })

with open(OUT / "results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n=== DONE: {len(results)} measurements ===")
from collections import defaultdict
by_m = defaultdict(list)
for r in results:
    by_m[r['member']].append(r['circ_cm'])
for m, v in sorted(by_m.items()):
    print(f"{m}: {sum(v)/len(v):.1f}cm ± {np.std(v):.1f} (n={len(v)})")
