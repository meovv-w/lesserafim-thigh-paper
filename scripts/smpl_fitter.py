#!/usr/bin/env python3
"""
SMPL 3D Body Fitting → Thigh Measurement
BLADE's SMPL body model + MediaPipe tasks API
"""
import sys, os, json, glob, torch, cv2, numpy as np
import urllib.request

BLADE_ROOT = "/mnt/d/HERMES/study/lesserafim-thigh-paper/blade"
sys.path.insert(0, BLADE_ROOT)
os.environ["CUDA_HOME"] = os.path.expanduser("~/cuda-11.8")
os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{os.environ['CUDA_HOME']}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# 1. Load SMPL
print("Loading SMPL...")
from blade.models.body_models.builder import build_body_model
body_cfg = dict(type='SMPL', keypoint_src='smpl_54', keypoint_dst='smpl_54',
                model_path=f'{BLADE_ROOT}/body_models/smpl', keypoint_approximate=True)
smpl = build_body_model(body_cfg).to(device)
smpl.eval()
print(f"SMPL loaded. Faces: {smpl.faces_tensor.shape}")

# 2. MediaPipe
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp

MODEL_PATH = "/tmp/pose_landmarker_lite.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading pose model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        MODEL_PATH)

pose_detector = mp_vision.PoseLandmarker.create_from_options(
    mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5))

# 3. Thigh circumference from mesh
def thigh_circ_from_mesh(verts):
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    target_y = y_min + (y_max - y_min) * 0.45
    eps = 0.02
    mask = (verts[:, 1] > target_y - eps) & (verts[:, 1] < target_y + eps)
    
    def perim(pts):
        if len(pts) < 3: return 0.0
        cx, cz = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - cz, pts[:, 0] - cx)
        s = pts[np.argsort(angles)]
        d = np.diff(s, axis=0)
        p = np.sum(np.sqrt((d**2).sum(axis=1)))
        p += np.sqrt(((s[0] - s[-1])**2).sum())
        return p
    
    left = perim(verts[mask & (verts[:, 0] < 0)][:, [0, 2]])
    right = perim(verts[mask & (verts[:, 0] > 0)][:, [0, 2]])
    return left, right

# 4. Fitter
class Fitter:
    def __init__(self, smpl, dev):
        self.smpl = smpl
        self.dev = dev
        # MediaPipe landmarks to SMPL 54 joints
        self.mp_ids = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        self.smpl_ids = [12, 9, 13, 10, 14, 11, 3, 0, 4, 1, 5, 2]
    
    def fit(self, landmarks, W, H, iters=200, lr=0.1):
        mp_ids_t = torch.tensor(self.mp_ids, device=self.dev)
        smpl_ids_t = torch.tensor(self.smpl_ids, device=self.dev)
        
        # Landmarks from tasks API
        kp = []
        conf = []
        for idx in self.mp_ids:
            lm = landmarks[idx]
            kp.append([lm.x * W, lm.y * H])
            conf.append(0.9 if lm.visibility and lm.visibility > 0.5 else 0.1)
        kp2d = torch.tensor(kp, device=self.dev, dtype=torch.float32)
        kpc = torch.tensor(conf, device=self.dev, dtype=torch.float32)
        
        focal = max(W, H) * 1.2
        K = torch.tensor([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]],
                         device=self.dev, dtype=torch.float32)
        
        betas = torch.zeros(1, 10, device=self.dev, requires_grad=True)
        body_pose = torch.zeros(1, 69, device=self.dev, requires_grad=True)
        global_orient = torch.zeros(1, 3, device=self.dev, requires_grad=True)
        transl = torch.zeros(1, 3, device=self.dev, requires_grad=True)
        scale = torch.ones(1, device=self.dev, requires_grad=True)
        
        opt = torch.optim.Adam([betas, body_pose, global_orient, transl, scale], lr=lr)
        
        for i in range(iters):
            opt.zero_grad()
            out = self.smpl(betas=betas, body_pose=body_pose,
                          global_orient=global_orient, transl=transl)
            j3d = out['joints'][0, smpl_ids_t]
            proj = j3d @ K.T
            p2d = proj[:, :2] / (proj[:, 2:3] + 1e-8) * scale
            loss_2d = (kpc * torch.norm(p2d - kp2d, dim=1)).mean()
            loss = loss_2d + 0.01 * betas.norm() + 0.001 * body_pose.norm() + 0.01 * transl.norm()
            loss.backward()
            opt.step()
            if i % 50 == 0:
                print(f"  {i}: loss={loss.item():.3f} (2D={loss_2d.item():.3f})")
        
        with torch.no_grad():
            out = self.smpl(betas=betas, body_pose=body_pose,
                          global_orient=global_orient, transl=transl)
            verts = out['vertices'][0].cpu().numpy()
        return {'vertices': verts}

fitter = Fitter(smpl, device)

# 5. Process images
DATA = "/mnt/d/HERMES/study/lesserafim-thigh-paper/data/candid_photos"
OUT = "/mnt/d/HERMES/study/lesserafim-thigh-paper/output/measurements_smpl"
os.makedirs(OUT, exist_ok=True)

results = []
imgs = sorted(glob.glob(f"{DATA}/**/*.jpeg", recursive=True))[:5]

for img_path in imgs:
    parts = img_path.replace(DATA, '').strip('/').split('/')
    era, member = parts[0], parts[1]
    print(f"\n[{era}/{member}] {os.path.basename(img_path)}")
    
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    res = pose_detector.detect(mp_img)
    if not res or not res.pose_landmarks:
        print("  No pose")
        continue
    
    landmarks = res.pose_landmarks[0]
    H, W = img.shape[:2]
    
    try:
        fit = fitter.fit(landmarks, W, H)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  Fail: {e}")
        continue
    
    lc, rc = thigh_circ_from_mesh(fit['vertices'])
    avg = (lc + rc) / 2 if lc > 0 or rc > 0 else 0
    print(f"  Thigh: L={lc:.1f} R={rc:.1f} AVG={avg:.1f}")
    
    results.append({
        'member': member, 'era': era, 'image': img_path,
        'left_circ': float(lc), 'right_circ': float(rc), 'avg_circ': float(avg),
    })

with open(f"{OUT}/smpl_test.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n{len(results)} images done!")
