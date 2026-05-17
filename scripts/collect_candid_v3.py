#!/usr/bin/env python3
"""
LE SSERAFIM Candid Photo Collector v3
Directly downloads from specific kpopping event album pages.
These pages work with curl (static HTML).
"""

import json, os, subprocess, sys, time, re
from pathlib import Path
from urllib.parse import unquote

BASE = Path("/mnt/d/HERMES/study/lesserafim-thigh-paper/data")
CANDID_DIR = BASE / "candid_photos"
CANDID_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Known kpopping event album URLs, organized by era and member
# Format: { era: { member: [(date, url, event_name), ...] } }
# ============================================================
ALBUMS = {
    "antifragile": {
        "sakura": [
            ("221017", "https://kpopping.com/kpics/221017-LE-SSERAFIM-Sakura-ANTIFRAGILE-Showcase", "ANTIFRAGILE_Showcase"),
        ],
        "chaewon": [
        ],
        "yunjin": [],
        "kazuha": [],
        "eunchae": [],
        "group": [
        ],
    },
    "unforgiven": {
        "sakura": [
            ("230501", "https://kpopping.com/kpics/230501-LE-SSERAFIM-Sakura-UNFORGIVEN-Press-Showcase", "UNFORGIVEN_PressShowcase"),
        ],
        "chaewon": [
        ],
        "yunjin": [],
        "kazuha": [],
        "eunchae": [],
        "group": [
        ],
    },
    "easy": {
        "sakura": [
            ("240314", "https://kpopping.com/kpics/240314-LE-SSERAFIM-Sakura-Music-Plant-Fansign-Event", "MusicPlant_Fansign"),
        ],
        "chaewon": [],
        "yunjin": [],
        "kazuha": [],
        "eunchae": [
            ("230210", "https://kpopping.com/kpics/230210-Hong-Eunchae-Music-Bank-MC", "MusicBank_MC"),
        ],
        "group": [],
    },
    "crazy": {
        "sakura": [
            ("240830", "https://kpopping.com/kpics/240830-LE-SSERAFIM-Sakura-Crazy-at-MusicBank", "Crazy_MusicBank"),
            ("240901", "https://kpopping.com/kpics/240901-LE-SSERAFIM-Sakura-Crazy-1-800-hot-n-fun-at-Inkigayo", "Crazy_Inkigayo"),
        ],
        "chaewon": [
            ("240901", "https://kpopping.com/kpics/240901-LE-SSERAFIM-Chaewon-Crazy-1-800-hot-n-fun-at-Inkigayo", "Crazy_Inkigayo"),
        ],
        "yunjin": [
            ("240901", "https://kpopping.com/kpics/240901-LE-SSERAFIM-Yunjin-Crazy-1-800-hot-n-fun-at-Inkigayo", "Crazy_Inkigayo"),
        ],
        "kazuha": [
            ("240901", "https://kpopping.com/kpics/240901-LE-SSERAFIM-Kazuha-Crazy-1-800-hot-n-fun-at-Inkigayo", "Crazy_Inkigayo"),
            ("240905", "https://kpopping.com/kpics/240905-LE-SSERAFIM-Kazuha-Crazy-1-800-hot-n-fun-at-Mcountdown", "Crazy_Mcountdown"),
        ],
        "eunchae": [
            ("240901", "https://kpopping.com/kpics/240901-LE-SSERAFIM-Eunchae-Crazy-1-800-hot-n-fun-at-Inkigayo", "Crazy_Inkigayo"),
        ],
        "group": [
            ("240905", "https://kpopping.com/kpics/240905-LE-SSERAFIM-CRAZY-1-800-hot-n-fun-at-M-COUNTDOWN", "Crazy_Mcountdown"),
            ("240830", "https://kpopping.com/kpics/240830-LE-SSERAFIM-Crazy-at-MusicBank", "Crazy_MusicBank"),
        ],
    },
    "hot": {
        "sakura": [
            ("251026", "https://kpopping.com/kpics/251026-LE-SSERAFIM-Sakura-at-Fansign-Event", "Fansign_251026"),
            ("251129", "https://kpopping.com/kpics/251129-LE-SSERAFIM-Sakura-at-MUSICART-Fansign-Event", "MUSICART_Fansign"),
        ],
        "chaewon": [
            ("251026", "https://kpopping.com/kpics/251026-LE-SSERAFIM-Chaewon-at-Fansign-Event", "Fansign_251026"),
            ("251129", "https://kpopping.com/kpics/251129-LE-SSERAFIM-Chaewon-at-MUSICART-Fansign-Event", "MUSICART_Fansign"),
        ],
        "yunjin": [
            ("251026", "https://kpopping.com/kpics/251026-LE-SSERAFIM-Yunjin-at-Fansign-Event", "Fansign_251026"),
            ("251129", "https://kpopping.com/kpics/251129-LE-SSERAFIM-Yunjin-at-MUSICART-Fansign-Event", "MUSICART_Fansign"),
        ],
        "kazuha": [
            ("251026", "https://kpopping.com/kpics/251026-LE-SSERAFIM-Kazuha-at-Fansign-Event", "Fansign_251026"),
            ("251129", "https://kpopping.com/kpics/251129-LE-SSERAFIM-Kazuha-at-MUSICART-Fansign-Event", "MUSICART_Fansign"),
            ("251108", "https://kpopping.com/kpics/251108-KAZUHA-at-Fansign-Event", "Fansign_251108"),
        ],
        "eunchae": [
            ("251026", "https://kpopping.com/kpics/251026-LE-SSERAFIM-Eunchae-at-Fansign-Event", "Fansign_251026"),
            ("251129", "https://kpopping.com/kpics/251129-LE-SSERAFIM-Eunchae-at-MUSICART-Fansign-Event", "MUSICART_Fansign"),
        ],
        "group": [
            ("251109", "https://kpopping.com/kpics/251109-LE-SSERAFIM-at-Inkigayo-Mini-Fanmeeting", "Inkigayo_MiniFanmeet"),
            ("251129", "https://kpopping.com/kpics/251129-LE-SSERAFIM-at-MUSICART-Fansign-Event", "MUSICART_Fansign_Group"),
        ],
    },
}

def extract_images_from_album(album_url):
    """Extract legacy.kpopping.com image URLs from an album page."""
    cmd = f"""curl -sL '{album_url}' 2>/dev/null | grep -oP 'url=https?%3A%2F%2Flegacy\\.kpopping\\.com[^"&]+' | sed 's/url=//' """
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return []
    
    urls = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            decoded = unquote(line)
            if decoded.startswith('https://legacy.kpopping.com/'):
                urls.append(decoded)
        except:
            pass
    
    return list(set(urls))

def download_image(img_url, output_path):
    """Download a single image."""
    if output_path.exists() and output_path.stat().st_size > 5000:
        return True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = f"curl -sL -o '{output_path}' '{img_url}'"
    subprocess.run(cmd, shell=True, capture_output=True, timeout=30)
    return output_path.exists() and output_path.stat().st_size > 5000

if __name__ == "__main__":
    print("=" * 60)
    print("LE SSERAFIM Candid Photo Collector v3")
    print("=" * 60)
    
    total_downloaded = 0
    total_albums = sum(len(album_list) for era_data in ALBUMS.values() for member_list in era_data.values() for album_list in [member_list])
    
    # Flatten and count
    album_count = 0
    for era, members in ALBUMS.items():
        for member, album_list in members.items():
            album_count += len(album_list)
    
    processed = 0
    for era, members in ALBUMS.items():
        for member, album_list in members.items():
            for date_str, url, event_name in album_list:
                processed += 1
                print(f"\n[{processed}/{album_count}] {era}/{member} ({date_str})")
                
                # Extract image URLs
                img_urls = extract_images_from_album(url)
                if not img_urls:
                    print(f"  No images found (might be wrong URL)")
                    continue
                
                print(f"  Found {len(img_urls)} images")
                
                # Download
                event_dir = CANDID_DIR / era / member / f"{date_str}_{event_name}"
                downloaded = 0
                for i, img_url in enumerate(img_urls):
                    ext = img_url.split('.')[-1].split('?')[0]
                    out_path = event_dir / f"img_{i+1:03d}.{ext}"
                    if download_image(img_url, out_path):
                        downloaded += 1
                
                print(f"  Downloaded {downloaded}/{len(img_urls)} images")
                total_downloaded += downloaded
                time.sleep(0.3)  # Rate limiting
    
    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    
    for era in sorted(os.listdir(CANDID_DIR)):
        era_dir = CANDID_DIR / era
        if not era_dir.is_dir():
            continue
        count = sum(1 for _ in era_dir.rglob("*") if _.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        print(f"  {era:15s}: {count:4d} images")
    
    print(f"\n  Total: {total_downloaded} images")
    print(f"  Location: data/candid_photos/")
