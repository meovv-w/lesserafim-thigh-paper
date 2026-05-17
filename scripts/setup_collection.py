#!/usr/bin/env python3
"""
LE SSERAFIM Multi-Epoch Image Collector
Collects concept photos and fancam frames for each comeback era.
Output: organized by era/member for BLADE inference.

Eras covered:
  1. ANTIFRAGILE (2022.10) 
  2. UNFORGIVEN  (2023.05)
  3. EASY        (2024.02)
  4. CRAZY       (2024.08)
  5. HOT         (2025.03)

Members (current 5):
  Sakura, Kim Chaewon, Huh Yunjin, Kazuha, Hong Eunchae
"""

import os, json, sys
from pathlib import Path

BASE = Path("/mnt/d/HERMES/study/lesserafim-thigh-paper/data")

ERAS = {
    "antifragile": {"date": "2022-10", "album": "ANTIFRAGILE", "type": "mini"},
    "unforgiven":  {"date": "2023-05", "album": "UNFORGIVEN",  "type": "full"},
    "easy":        {"date": "2024-02", "album": "EASY",        "type": "mini"},
    "crazy":       {"date": "2024-08", "album": "CRAZY",       "type": "mini"},
    "hot":         {"date": "2025-03", "album": "HOT",         "type": "mini"},
}

MEMBERS = ["sakura", "chaewon", "yunjin", "kazuha", "eunchae"]

def create_dirs():
    """Create the directory structure for collected images."""
    for era in ERAS:
        for member in MEMBERS:
            path = BASE / "concept_photos" / era / member
            path.mkdir(parents=True, exist_ok=True)
        # Also create a group folder for group shots
        (BASE / "concept_photos" / era / "group").mkdir(parents=True, exist_ok=True)
    # Fancam frames
    for era in ERAS:
        (BASE / "fancam_frames" / era).mkdir(parents=True, exist_ok=True)
    
    print("Directory structure created:")
    for era in ERAS:
        print(f"  data/concept_photos/{era}/")
        for m in MEMBERS:
            print(f"    {m}/")
        print(f"    group/")
    print()

def create_era_info():
    """Save era metadata."""
    info = {
        "description": "LE SSERAFIM Multi-Epoch Image Collection for BLADE Inference",
        "created": "2026-05-16",
        "members": MEMBERS,
        "eras": ERAS,
        "image_count": {}
    }
    # Count actual files if any exist
    for era in ERAS:
        info["image_count"][era] = {}
        for member in MEMBERS:
            folder = BASE / "concept_photos" / era / member
            if folder.exists():
                files = list(folder.glob("*"))
                info["image_count"][era][member] = len(files)
    
    with open(BASE / "collection_manifest.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"Manifest saved: data/collection_manifest.json")

def print_collection_guide():
    """Print instructions on what to collect."""
    print("=" * 60)
    print("IMAGE COLLECTION GUIDE")
    print("=" * 60)
    print()
    print("For each era, collect for each member:")
    print()
    for era, meta in ERAS.items():
        print(f"── {era.upper()} ({meta['date']}) ──")
        print(f"  Album: {meta['album']}")
        print(f"  Needs:")
        print(f"    - 5-10 solo concept photos (full body, legs visible)")
        print(f"    - 3-5 group photos (for cross-validation)")
        print(f"    - 1-2 fancam videos or keyframes")
        print()
    print("Sources:")
    print("  - kpopping.com (high-res concept photos)")
    print("  - Weverse (official photos)")
    print("  - YouTube fancams (extract frames)")
    print()

if __name__ == "__main__":
    create_dirs()
    create_era_info()
    print_collection_guide()
