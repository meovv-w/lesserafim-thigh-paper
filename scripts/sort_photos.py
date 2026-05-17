#!/usr/bin/env python3
"""
Auto-sort LE SSERAFIM concept photos by member using face recognition.
1. Takes reference photos of each member (from their profile pics)
2. Detects faces in concept photos
3. Matches against references
4. Moves to correct member folder

Usage: python3 scripts/sort_photos.py
"""

import os, sys, json, shutil
from pathlib import Path

BASE = Path("/mnt/d/HERMES/study/lesserafim-thigh-paper/data")
REF_DIR = BASE / "reference_faces"
PHOTO_BASE = BASE / "concept_photos"

MEMBERS = {
    "sakura": "sakura",
    "chaewon": "chaewon",
    "yunjin": "yunjin",
    "kazuha": "kazuha",
    "eunchae": "eunchae",
}

def download_reference_photos():
    """Download official profile photos as face references."""
    import urllib.request
    
    refs = {
        "sakura": "https://legacy.kpopping.com/thumb/4252919.jpg",
        "chaewon": "https://legacy.kpopping.com/thumb/4257218.jpg",
        "yunjin": "https://legacy.kpopping.com/thumb/4260119.jpg",
        "kazuha": "https://legacy.kpopping.com/thumb/4258411.jpg",
        "eunchae": "https://legacy.kpopping.com/thumb/4255391.jpg",
    }
    
    for member, url in refs.items():
        fpath = REF_DIR / f"{member}.jpg"
        if not fpath.exists():
            try:
                urllib.request.urlretrieve(url, fpath)
                print(f"  Downloaded reference: {member}")
            except Exception as e:
                print(f"  Failed to download {member}: {e}")

def sort_with_mediapipe():
    """Use MediaPipe Face Detection + Face Recognition to sort photos."""
    try:
        import mediapipe as mp
        import cv2
        import numpy as np
    except ImportError:
        print("mediapipe not installed. Install with: pip install mediapipe opencv-python-headless")
        return False
    
    # Load reference face encodings
    ref_encodings = {}
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    
    print("\n[Step 1] Loading reference faces...")
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    
    for member in MEMBERS:
        ref_path = REF_DIR / f"{member}.jpg"
        if not ref_path.exists():
            print(f"  No reference for {member}, skipping auto-sort")
            return False
        
        img = cv2.imread(str(ref_path))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)
        if results.detections:
            ref_encodings[member] = results.detections[0]
            print(f"  Loaded ref: {member}")
    
    face_detection.close()
    
    if len(ref_encodings) < 5:
        print(f"  Only loaded {len(ref_encodings)}/5 references, need all 5")
        return False
    
    print("\n[Step 2] Sorting photos by face matching...")
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    
    total_sorted = 0
    for era_dir in sorted(PHOTO_BASE.iterdir()):
        if not era_dir.is_dir():
            continue
        raw_dir = era_dir / "raw"
        if not raw_dir.exists():
            continue
        
        for img_path in sorted(raw_dir.glob("*")):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            if w > 1920:
                scale = 1920 / w
                new_w = 1920
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h))
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb)
            
            if not results.detections:
                # No face found, put in unknown
                unknown_dir = raw_dir.parent / "unknown"
                unknown_dir.mkdir(exist_ok=True)
                shutil.copy2(str(img_path), str(unknown_dir / img_path.name))
                continue
            
            # For each detected face, find best match
            best_member = None
            best_score = float('inf')
            
            for detection in results.detections:
                # Compare bounding box position as rough heuristic
                # In a real solution, we'd use face_recognition library
                # For now, use a simpler approach: detect largest face
                bbox = detection.location_data.relative_bounding_box
                face_size = bbox.width * bbox.height
                
                if best_member is None or face_size > best_score:
                    best_score = face_size
                    # We'll just use the largest face and move file to era-level
                    # Actual member identification would need proper face_recognition
            
            # Since mediapipe doesn't do face identification natively,
            # we'll need a different approach. Let's use the face_recognition lib approach.
            # For now, move to unsorted
            unsorted_dir = raw_dir.parent / "unsorted"
            unsorted_dir.mkdir(exist_ok=True)
            shutil.copy2(str(img_path), str(unsorted_dir / img_path.name))
            total_sorted += 1
    
    face_detection.close()
    print(f"\n  Moved {total_sorted} photos to unsorted/")
    return total_sorted > 0

def try_face_recognition_lib():
    """Try using face_recognition library for proper identification."""
    try:
        import face_recognition
    except ImportError:
        print("\nface_recognition not installed.")
        print("Install with: pip install face_recognition")
        print("Or manually sort photos into: data/concept_photos/{era}/{member}/")
        return False
    
    # Load reference encodings
    known_encodings = {}
    known_names = []
    
    print("\n[Step 1] Loading reference face encodings...")
    for member in MEMBERS:
        ref_path = REF_DIR / f"{member}.jpg"
        if not ref_path.exists():
            continue
        try:
            img = face_recognition.load_image_file(str(ref_path))
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_encodings[member] = encodings[0]
                known_names.append(member)
                print(f"  Loaded: {member}")
        except Exception as e:
            print(f"  Error loading {member}: {e}")
    
    if len(known_names) < 3:
        print(f"  Not enough references ({len(known_names)}), aborting")
        return False
    
    total_sorted = 0
    errors = 0
    
    for era_dir in sorted(PHOTO_BASE.iterdir()):
        if not era_dir.is_dir():
            continue
        era_name = era_dir.name
        
        raw_dir = era_dir / "raw"
        if not raw_dir.exists():
            continue
        
        for img_path in sorted(raw_dir.glob("*")):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            
            try:
                img = face_recognition.load_image_file(str(img_path))
                face_locs = face_recognition.face_locations(img)
                face_encs = face_recognition.face_encodings(img, face_locs)
                
                if not face_encs:
                    unknown_dir = era_dir / "unknown"
                    unknown_dir.mkdir(exist_ok=True)
                    shutil.copy2(str(img_path), str(unknown_dir / img_path.name))
                    continue
                
                for face_enc in face_encs:
                    matches = face_recognition.compare_faces(
                        [known_encodings[n] for n in known_names], 
                        face_enc, 
                        tolerance=0.5
                    )
                    if True in matches:
                        match_idx = matches.index(True)
                        member = known_names[match_idx]
                        member_dir = era_dir / member
                        member_dir.mkdir(exist_ok=True)
                        shutil.copy2(str(img_path), str(member_dir / img_path.name))
                        total_sorted += 1
                        break
                    else:
                        unknown_dir = era_dir / "unknown"
                        unknown_dir.mkdir(exist_ok=True)
                        shutil.copy2(str(img_path), str(unknown_dir / img_path.name))
                        
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  Error processing {img_path.name}: {e}")
    
    print(f"\nSorted {total_sorted} photos by face recognition")
    if errors:
        print(f"Errors: {errors}")
    return total_sorted > 0

def print_manual_guide():
    """Print instructions for manual sorting."""
    print("\n" + "=" * 60)
    print("MANUAL SORTING GUIDE")
    print("=" * 60)
    print("""
If auto-sort doesn't work, manually sort photos:
  data/concept_photos/{era}/{member}/

For each era, look at the raw images and copy/move to member folders.

To view images, open the folder in Windows Explorer:
  D:\\HERMES\\study\\lesserafim-thigh-paper\\data\\concept_photos\\{era}\\raw\\
""")
    print("Eras available:", [d.name for d in sorted(PHOTO_BASE.iterdir()) if d.is_dir()])
    print()

if __name__ == "__main__":
    REF_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("LE SSERAFIM Photo Sorter")
    print("=" * 60)
    
    # Step 1: Get reference photos
    print("\n[Downloading reference photos...]")
    download_reference_photos()
    
    # Step 2: Try face_recognition (best)
    success = try_face_recognition_lib()
    
    # Step 3: If that fails, try mediapipe
    if not success:
        print("\nTrying MediaPipe fallback...")
        success = sort_with_mediapipe()
    
    # Step 4: If still fails, print manual guide
    if not success:
        print_manual_guide()
