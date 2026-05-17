#!/usr/bin/env python3
"""
LE SSERAFIM Candid Photo Collector v2
Collects fan-taken event photos from kpopping member albums.
These are more natural than official concept photos.

Strategy: Use each member's album page to find dated event photo sets
with legs visible (fansigns, music shows, airports, concerts).
"""

import json, os, re, subprocess, sys, time
from pathlib import Path
from urllib.parse import unquote
from collections import defaultdict

BASE = Path("/mnt/d/HERMES/study/lesserafim-thigh-paper/data")
CANDID_DIR = BASE / "candid_photos"
CANDID_DIR.mkdir(parents=True, exist_ok=True)

MEMBERS = {
    "sakura": "https://kpopping.com/kpics/gender-all/category-any/idol-Sakura/group-any/order-new",
    "chaewon": "https://kpopping.com/kpics/gender-all/category-any/idol-Kim-Chaewon/group-any/order-new",
    "yunjin": "https://kpopping.com/kpics/gender-all/category-any/idol-Huh-Yunjin/group-any/order-new",
    "kazuha": "https://kpopping.com/kpics/gender-all/category-any/idol-Kazuha/group-any/order-new",
    "eunchae": "https://kpopping.com/kpics/gender-all/category-any/idol-Hong-Eunchae/group-any/order-new",
}

# Era date ranges for temporal assignment
ERAS = {
    "antifragile": ("20221001", "20230430"),
    "unforgiven": ("20230501", "20240131"),
    "easy": ("20240201", "20240731"),
    "crazy": ("20240801", "20250228"),
    "hot": ("20250301", "20260516"),
}

# Event types that likely show legs
EVENT_KEYWORDS = [
    "inkigayo", "mcountdown", "musicbank", "music bank", "showcase",
    "fansign", "concert", "tour", "festival", "gayo", "airport",
    "departure", "arrival", "summer", "university", "outdoor",
]

def extract_page_urls(page_url):
    """Extract all unique kpopping album URLs from a member page."""
    cmd = f"""curl -sL '{page_url}' 2>/dev/null | grep -oP 'href="/kpics/[^"]+"' | sed 's/href="//;s/"//' | sort -u"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return []
    
    full_urls = []
    for path in result.stdout.strip().split('\n'):
        path = path.strip()
        if path and not path.startswith('http'):
            path = f"https://kpopping.com{path}"
        if path and path not in full_urls:
            full_urls.append(path)
    return full_urls

def extract_album_urls(album_url):
    """Extract legacy.kpopping.com image URLs from an album page."""
    cmd = f"""curl -sL '{album_url}' 2>/dev/null | grep -oP 'url=https?%3A%2F%2Flegacy\\.kpopping\\.com[^"&]+' | sed 's/url=//' """
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return []
    
    urls = []
    for line in result.stdout.strip().split('\n'):
        try:
            decoded = unquote(line.strip())
            if decoded.startswith('https://legacy.kpopping.com/'):
                urls.append(decoded)
        except:
            pass
    
    return list(set(urls))

def parse_date_from_url(url):
    """Extract date from kpopping URL (YYMMDD format in the path)."""
    m = re.search(r'/(\d{6})-', url)
    if not m:
        m = re.search(r'/(20\d{2})(\d{2})(\d{2})', url)
        if m:
            return f"{m.group(1)}{m.group(2)}{m.group(3)}"
        return None
    date_str = m.group(1)
    return f"20{date_str}" if len(date_str) == 6 else date_str

def assign_era(date_str):
    """Assign a date string to an era."""
    if not date_str:
        return "unknown"
    for era_name, (start, end) in ERAS.items():
        if start <= date_str <= end:
            return era_name
    return "unknown"

def is_relevant_album(album_url, keyword_filter=True):
    """Check if an album likely has visible legs."""
    if keyword_filter:
        url_lower = album_url.lower()
        return any(kw in url_lower for kw in EVENT_KEYWORDS)
    return True

def download_single_image(img_url, output_path):
    """Download a single image if it doesn't exist."""
    if output_path.exists() and output_path.stat().st_size > 5000:
        return True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = f"curl -sL -o '{output_path}' '{img_url}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, timeout=30)
    return output_path.exists() and output_path.stat().st_size > 5000

def collect_for_member(member_name, member_url, limit_albums=30):
    """Collect candid photos for a specific member."""
    print(f"\n{'='*60}")
    print(f"Collecting for: {member_name}")
    print(f"{'='*60}")
    
    # Step 1: Get album URLs from member page
    print("  [1/3] Scanning member album page...")
    all_albums = extract_page_urls(member_url)
    print(f"    Found {len(all_albums)} albums")
    
    # Filter relevant ones
    relevant = [a for a in all_albums[:limit_albums*3] if is_relevant_album(a)]
    print(f"    {len(relevant)} relevant (event/music show/airport)")
    
    # Step 2: For each album, download images
    downloaded = 0
    albums_processed = 0
    for album_url in relevant[:limit_albums]:
        date_str = parse_date_from_url(album_url)
        era = assign_era(date_str)
        
        if era == "unknown":
            continue
        
        # Extract image URLs
        img_urls = extract_album_urls(album_url)
        if not img_urls:
            continue
        
        # Create output path: candid_photos/{era}/{member}/{date}_{event}/
        event_name = album_url.split('/')[-1].split('?')[0][:60]
        event_dir = CANDID_DIR / era / member / f"{date_str}_{event_name}" if date_str else CANDID_DIR / era / member / f"unknown_{event_name}"
        
        for i, img_url in enumerate(img_urls[:5]):  # Max 5 per album
            ext = img_url.split('.')[-1].split('?')[0]
            out_path = event_dir / f"img_{i+1:03d}.{ext}"
            if download_single_image(img_url, out_path):
                downloaded += 1
        
        albums_processed += 1
        if albums_processed % 5 == 0:
            print(f"    Processed {albums_processed}/{min(len(relevant), limit_albums)} albums, {downloaded} images")
        
        time.sleep(0.2)  # Rate limiting
    
    print(f"  Total: {downloaded} images downloaded for {member_name}")
    return downloaded

if __name__ == "__main__":
    print("=" * 60)
    print("LE SSERAFIM Candid Photo Collector v2")
    print("Collects fan-taken photos with visible legs")
    print("=" * 60)
    
    total = 0
    for member_name, member_url in MEMBERS.items():
        count = collect_for_member(member_name, member_url, limit_albums=20)
        total += count
    
    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    
    for era in ERAS:
        era_dir = CANDID_DIR / era
        if era_dir.exists():
            count = sum(1 for _ in era_dir.rglob("*") if _.suffix.lower() in ('.jpg', '.jpeg', '.png'))
            print(f"  {era:15s}: {count:4d} images")
    
    print(f"\n  Total candid photos: {total}")
    print(f"  Location: data/candid_photos/{{era}}/{{member}}/{{date}}_{{event}}/")
