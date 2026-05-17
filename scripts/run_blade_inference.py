#!/usr/bin/env python3
"""
BLADE Inference Pipeline for LE SSERAFIM Thigh Analysis.
Runs BLADE on all concept photos and extracts SMPL-X meshes.

Usage: python3 scripts/run_blade_inference.py
Output: output/meshes/{era}/{img_name}_mesh.ply
"""

import os, sys, json, subprocess, glob
from pathlib import Path

BASE = Path("/mnt/d/HERMES/study/lesserafim-thigh-paper")
BLADE_REPO = BASE / "blade"
DATA_DIR = BASE / "data" / "concept_photos"
OUTPUT_DIR = BASE / "output" / "meshes"

# SMPL-X keypoint/vertex indices for thigh measurements
# SMPL-X model: 10475 vertices
# Thigh region vertices (approximate, based on SMPL-X vertex segmentation):
# Left thigh: vertices 3949-4406
# Right thigh: vertices 4407-4864
# These need to be refined once we have the actual model loaded

THIGH_VERTEX_RANGES = {
    "left_thigh": (3949, 4406),
    "right_thigh": (4407, 4864),
}

def check_prerequisites():
    """Check that BLADE is installed and models exist."""
    # Check BLADE repo exists
    if not (BLADE_REPO / "api" / "BLADE_API.py").exists():
        print("ERROR: BLADE repo not found at", BLADE_REPO)
        return False
    
    # Check SMPL-X models
    smplx_dir = BLADE_REPO / "body_models" / "smplx"
    if not smplx_dir.exists():
        print("WARNING: SMPL-X models not found at", smplx_dir)
        print("Please download from https://smpl-x.is.tue.mpg.de/")
        print(f"and place in: {smplx_dir}")
    
    # Check pretrained weights
    if not (BLADE_REPO / "pretrained" / "epoch_2.pth").exists():
        print("WARNING: BLADE checkpoint not found")
        print("Download with: cd blade && huggingface-cli download McMvMc/BLADE epoch_2.pth --local-dir pretrained")
    
    return True

def collect_images():
    """Get all images organized by era."""
    era_images = {}
    for era_dir in sorted(DATA_DIR.iterdir()):
        if not era_dir.is_dir():
            continue
        era_name = era_dir.name
        images = []
        
        # Check raw, sorted member dirs, unsorted
        for search_dir in [era_dir / "raw", era_dir]:
            if search_dir.exists():
                for img_file in sorted(search_dir.glob("*")):
                    if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        # Determine member from path
                        member = "unknown"
                        for m in ["sakura", "chaewon", "yunjin", "kazuha", "eunchae"]:
                            if m in str(img_file.parent):
                                member = m
                                break
                        images.append({
                            "path": str(img_file),
                            "name": img_file.stem,
                            "member": member,
                            "era": era_name
                        })
        
        if images:
            era_images[era_name] = images
    
    return era_images

def run_blade_inference(era_images):
    """Run BLADE on each image using the API."""
    import sys
    sys.path.insert(0, str(BLADE_REPO))
    
    try:
        from api.BLADE_API import BLADE_API
    except ImportError:
        print("ERROR: Cannot import BLADE_API. Has BLADE been installed correctly?")
        return {}
    
    # Load config
    config_path = BLADE_REPO / "blade" / "configs" / "blade_inthewild.py"
    
    api = BLADE_API(str(config_path))
    
    results = {}
    total = sum(len(imgs) for imgs in era_images.values())
    processed = 0
    
    for era_name, images in era_images.items():
        era_output = OUTPUT_DIR / era_name
        era_output.mkdir(parents=True, exist_ok=True)
        
        for img_info in images:
            img_path = img_info["path"]
            member = img_info["member"]
            img_name = img_info["name"]
            
            # Output paths
            mesh_dir = era_output / member
            mesh_dir.mkdir(parents=True, exist_ok=True)
            output_prefix = str(mesh_dir / img_name)
            
            if (mesh_dir / f"{img_name}_smplx.ply").exists():
                print(f"  [{processed+1}/{total}] {era_name}/{member}/{img_name} -> already processed, skipping")
                processed += 1
                continue
            
            print(f"  [{processed+1}/{total}] Processing {era_name}/{member}/{img_name}...")
            
            try:
                api.process_image(
                    img_path=img_path,
                    output_prefix=output_prefix,
                    save_mesh=True,
                    save_params=True,
                )
                outfile = f"{output_prefix}_smplx.ply"
                if os.path.exists(outfile):
                    print(f"    -> Mesh saved: {outfile}")
                else:
                    print(f"    -> Warning: Output not found")
                    
            except Exception as e:
                print(f"    -> ERROR: {e}")
            
            processed += 1
            
            # Store result reference
            key = f"{era_name}/{member}/{img_name}"
            results[key] = {
                "era": era_name,
                "member": member,
                "image": img_info["path"],
                "mesh_dir": str(mesh_dir),
            }
    
    return results

def download_blade_checkpoint():
    """Download BLADE checkpoint if not present."""
    ckpt = BLADE_REPO / "pretrained" / "epoch_2.pth"
    if ckpt.exists():
        print("BLADE checkpoint already present")
        return True
    
    print("Downloading BLADE checkpoint from HuggingFace...")
    os.makedirs(ckpt.parent, exist_ok=True)
    result = subprocess.run(
        ["huggingface-cli", "download", "McMvMc/BLADE", "epoch_2.pth", 
         "--local-dir", str(ckpt.parent)],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("Downloaded successfully")
        return True
    else:
        print("Download failed:", result.stderr)
        return False

def download_supporting_models():
    """Download supporting models from HuggingFace."""
    models = [
        ("depth-anything/Depth-Anything-V2-Metric-Hypersim-Large", 
         "depth_anything_v2_metric_hypersim_vitl.pth", 
         "pretrained/model_init_weights"),
        ("ttxskk/AiOS", "aios_checkpoint.pth", "pretrained/model_init_weights"),
        ("facebook/sapiens-pose-bbox-detector", 
         "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth", 
         "pretrained/rtmpose"),
        ("facebook/sapiens-pose-1b", 
         "sapiens_1b_goliath_best_goliath_AP_639.pth", 
         "pretrained/pose"),
    ]
    
    for repo, fname, local_dir in models:
        fpath = BLADE_REPO / local_dir / fname
        if fpath.exists():
            print(f"  {fname} already present")
            continue
        
        print(f"  Downloading {fname}...")
        os.makedirs(fpath.parent, exist_ok=True)
        result = subprocess.run(
            ["huggingface-cli", "download", repo, fname, "--local-dir", str(fpath.parent)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"    Done")
        else:
            print(f"    Failed: {result.stderr[:200]}")

if __name__ == "__main__":
    print("=" * 60)
    print("BLADE Inference Pipeline - LE SSERAFIM Thigh Analysis")
    print("=" * 60)
    
    # Step 1: Check prerequisites
    print("\n[Step 1] Checking prerequisites...")
    if not check_prerequisites():
        sys.exit(1)
    
    # Step 2: Download models if needed
    print("\n[Step 2] Downloading models...")
    download_blade_checkpoint()
    download_supporting_models()
    
    # Step 3: Collect images
    print("\n[Step 3] Collecting images...")
    era_images = collect_images()
    total_imgs = sum(len(imgs) for imgs in era_images.values())
    print(f"  Found {total_imgs} images across {len(era_images)} eras")
    for era, imgs in era_images.items():
        print(f"    {era}: {len(imgs)} images")
    
    if total_imgs == 0:
        print("  No images found! Download photos first.")
        sys.exit(1)
    
    # Step 4: Run BLADE
    print("\n[Step 4] Running BLADE inference...")
    results = run_blade_inference(era_images)
    
    # Step 5: Save results manifest
    manifest_path = BASE / "output" / "inference_manifest.json"
    manifest = {
        "total_processed": len(results),
        "results": results,
        "timestamp": "2026-05-16",
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest saved: {manifest_path}")
    print("Done!")
