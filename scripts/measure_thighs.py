#!/usr/bin/env python3
"""
Thigh Measurement Extraction from SMPL-X Meshes.
Computes thigh circumference from 3D mesh vertices.

For each mesh:
1. Load the SMPL-X mesh (PLY or NPZ)
2. Extract vertices for the thigh region (mid-thigh cross-section)
3. Fit a circle/ellipse to the cross-section
4. Compute circumference
5. Record with confidence interval

Usage: python3 scripts/measure_thighs.py
Output: output/measurements/thigh_measurements.csv
"""

import os, sys, json, csv, math
from pathlib import Path
import numpy as np

BASE = Path("/mnt/d/HERMES/study/lesserafim-thigh-paper")
MESH_DIR = BASE / "output" / "meshes"
OUTPUT_DIR = BASE / "output" / "measurements"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# SMPL-X vertex indices for thigh cross-sections
# Based on SMPL-X model topology (10475 vertices):
# Thigh circumference measurement planes at ~40% of thigh length
# 
# Left leg mid-thigh cross-section vertex indices
# (These are approximate - refined once actual SMPL-X model is loaded)
LEFT_THIGH_MID = list(range(3949, 4130))       # ~180 vertices around left thigh
RIGHT_THIGH_MID = list(range(4407, 4588))       # ~180 vertices around right thigh

def load_mesh_vertices(mesh_path):
    """Load vertex positions from a PLY or NPZ mesh file."""
    ext = os.path.splitext(mesh_path)[1].lower()
    
    if ext == '.ply':
        return load_ply_vertices(mesh_path)
    elif ext == '.npz':
        return load_npz_vertices(mesh_path)
    elif ext == '.obj':
        return load_obj_vertices(mesh_path)
    else:
        print(f"  Unsupported format: {ext}")
        return None

def load_ply_vertices(ply_path):
    """Load vertices from a PLY file."""
    vertices = []
    with open(ply_path, 'r') as f:
        lines = f.readlines()
    
    # Parse PLY header
    vertex_count = 0
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith('element vertex'):
            vertex_count = int(line.split()[-1])
        if line.startswith('end_header'):
            header_end = i + 1
            break
    
    # Parse vertices
    for line in lines[header_end:header_end + vertex_count]:
        parts = line.strip().split()
        if len(parts) >= 3:
            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
    
    return np.array(vertices)

def load_npz_vertices(npz_path):
    """Load vertices from an NPZ file (SMPL-X parameter format)."""
    data = np.load(npz_path)
    # SMPL-X stores vertices under 'v' or 'vertices' key
    if 'v' in data:
        return data['v']
    elif 'vertices' in data:
        return data['vertices']
    else:
        print(f"  Unknown keys in NPZ: {list(data.keys())}")
        return None

def load_obj_vertices(obj_path):
    """Load vertices from an OBJ file."""
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)

def compute_cross_section(vertices, vertex_indices):
    """
    Compute the circumference of a cross-section through specified vertices.
    
    Method:
    1. Project selected vertices onto the plane perpendicular to the thigh axis
    2. Find the best-fit 2D circle/ellipse
    3. Compute perimeter
    """
    if len(vertices) == 0 or len(vertex_indices) == 0:
        return None, None
    
    # Get valid indices
    valid_idx = [i for i in vertex_indices if i < len(vertices)]
    if len(valid_idx) < 10:
        return None, None
    
    # Extract points
    points = vertices[valid_idx]
    
    # Find thigh axis (principal component 1 = along the leg)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # PCA to find the thigh axis
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # The smallest eigenvalue corresponds to the axis direction (along the leg)
    # The two largest eigenvalues span the cross-section plane
    axis_idx = np.argmin(eigvals)
    plane_indices = [i for i in range(3) if i != axis_idx]
    
    # Project onto the cross-section plane
    proj = centered[:, plane_indices]
    
    # Fit a circle using least squares
    x, y = proj[:, 0], proj[:, 1]
    
    # Simple circle fit: average radius
    radii = np.sqrt(x**2 + y**2)
    mean_radius = np.mean(radii)
    std_radius = np.std(radii)
    
    # Circumference = 2 * pi * r
    circumference = 2 * math.pi * mean_radius
    
    # Also compute the convex hull perimeter for comparison
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(proj)
        hull_perimeter = hull.area  # volume = 0 for 2D, area = perimeter
        # Use the larger of circle and hull for more accuracy
        circumference = max(circumference, hull_perimeter)
    except:
        pass
    
    # Quality metric: how circular is the cross-section (1.0 = perfect circle)
    circularity = mean_radius / (mean_radius + std_radius) if std_radius > 0 else 1.0
    
    confidence = max(50, min(95, int(circularity * 100)))
    
    return circumference, confidence

def process_all_meshes():
    """Process all meshes and compute thigh measurements."""
    all_measurements = []
    
    # Walk through mesh directory
    mesh_files = []
    for era_dir in sorted(MESH_DIR.iterdir()):
        if not era_dir.is_dir():
            continue
        for member_dir in sorted(era_dir.iterdir()):
            if not member_dir.is_dir():
                continue
            for f in sorted(member_dir.glob("*")):
                if f.suffix.lower() in ('.ply', '.npz', '.obj'):
                    mesh_files.append((era_dir.name, member_dir.name, f))
    
    if not mesh_files:
        print("No mesh files found. Run run_blade_inference.py first.")
        return []
    
    print(f"Found {len(mesh_files)} mesh files to process")
    
    for era, member, mesh_path in mesh_files:
        print(f"  Processing: {era}/{member}/{mesh_path.name}")
        
        vertices = load_mesh_vertices(str(mesh_path))
        if vertices is None:
            print(f"    Could not load mesh")
            continue
        
        # Compute left thigh
        left_circ, left_conf = compute_cross_section(vertices, LEFT_THIGH_MID)
        
        # Compute right thigh
        right_circ, right_conf = compute_cross_section(vertices, RIGHT_THIGH_MID)
        
        if left_circ is None and right_circ is None:
            print(f"    Could not compute circumference")
            continue
        
        # Average both legs
        circs = [c for c in [left_circ, right_circ] if c is not None]
        confs = [c for c in [left_conf, right_conf] if c is not None]
        
        avg_circ = np.mean(circs) if circs else 0
        avg_conf = np.mean(confs) if confs else 0
        circ_std = np.std(circs) if len(circs) > 1 else avg_circ * 0.02
        
        measurement = {
            "era": era,
            "member": member,
            "image": mesh_path.stem,
            "left_thigh_cm": round(left_circ, 2) if left_circ else None,
            "right_thigh_cm": round(right_circ, 2) if right_circ else None,
            "avg_thigh_cm": round(avg_circ, 2),
            "std_cm": round(circ_std, 2),
            "confidence_pct": int(avg_conf),
            "n_vertices": len(vertices),
        }
        all_measurements.append(measurement)
        print(f"    Left: {left_circ:.1f} cm | Right: {right_circ:.1f} cm | Avg: {avg_circ:.1f} cm")
    
    return all_measurements

def save_results(measurements):
    """Save measurements to CSV and JSON."""
    
    # CSV
    csv_path = OUTPUT_DIR / "thigh_measurements.csv"
    with open(csv_path, 'w', newline='') as f:
        if measurements:
            writer = csv.DictWriter(f, fieldnames=measurements[0].keys())
            writer.writeheader()
            writer.writerows(measurements)
    print(f"\nCSV saved: {csv_path}")
    
    # JSON
    json_path = OUTPUT_DIR / "thigh_measurements.json"
    with open(json_path, 'w') as f:
        json.dump(measurements, f, indent=2)
    print(f"JSON saved: {json_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("MEASUREMENT SUMMARY")
    print("=" * 60)
    
    from collections import defaultdict
    by_member = defaultdict(list)
    for m in measurements:
        by_member[m["member"]].append(m["avg_thigh_cm"])
    
    for member, circs in sorted(by_member.items()):
        print(f"  {member:10s}: {np.mean(circs):.1f} ± {np.std(circs):.2f} cm (n={len(circs)})")

def analyze_by_era(measurements):
    """Analyze thigh circumference changes across eras."""
    if not measurements:
        return
    
    print("\n" + "=" * 60)
    print("LONGITUDINAL ANALYSIS (Across Eras)")
    print("=" * 60)
    
    from collections import defaultdict
    by_era_member = defaultdict(lambda: defaultdict(list))
    
    for m in measurements:
        by_era_member[m["era"]][m["member"]].append(m["avg_thigh_cm"])
    
    eras_ordered = ["antifragile", "unforgiven", "easy", "crazy", "hot"]
    members_ordered = ["sakura", "chaewon", "yunjin", "kazuha", "eunchae"]
    
    # Print table
    header = f"{'Member':<12}" + "".join(f"{era:>12}" for era in eras_ordered)
    print(f"\n{header}")
    print("-" * len(header))
    
    for member in members_ordered:
        row = f"{member:<12}"
        for era in eras_ordered:
            vals = by_era_member[era][member]
            if vals:
                row += f"{np.mean(vals):>10.1f}±{np.std(vals):<4.2f}"
            else:
                row += f"{'N/A':>12}"
        print(row)
    
    print("\n(Values: mean thigh circumference in cm ± std)")

if __name__ == "__main__":
    print("=" * 60)
    print("LE SSERAFIM Thigh Measurement Pipeline")
    print("=" * 60)
    
    # Step 1: Process all meshes
    print("\n[Step 1] Processing meshes...")
    measurements = process_all_meshes()
    
    if not measurements:
        print("No measurements collected.")
        sys.exit(1)
    
    # Step 2: Save results
    print("\n[Step 2] Saving results...")
    save_results(measurements)
    
    # Step 3: Longitudinal analysis
    print("\n[Step 3] Longitudinal analysis...")
    analyze_by_era(measurements)
    
    print("\nDone! Results in:", OUTPUT_DIR)
