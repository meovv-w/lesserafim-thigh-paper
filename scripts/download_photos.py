#!/usr/bin/env python3
"""
Batch download all LE SSERAFIM concept photos from kpopping.
Usage: python3 scripts/download_photos.py
"""
import subprocess, json, os, sys
from pathlib import Path

BASE = Path("/mnt/d/HERMES/study/lesserafim-thigh-paper/data")
DOWNLOAD_DIR = BASE / "concept_photos"

ERAS = {
    "antifragile": {
        "name": "ANTIFRAGILE",
        "urls_file": str(BASE / "antifragile_urls.txt"),
        "pages": [
            "https://kpopping.com/kpics/LE-SSERAFIM-2nd-Mini-Album-ANTIFRAGILE-Concept-Teasers"
        ]
    },
    "unforgiven": {
        "name": "UNFORGIVEN",
        "urls_file": str(BASE / "unforgiven_urls.txt"),
        "pages": [
            "https://kpopping.com/kpics/LE-SSERAFIM-1st-Album-UNFORGIVEN-Concept-Photos"
        ]
    },
    "easy": {
        "name": "EASY",
        "urls_file": str(BASE / "easy_urls.txt"),
        "pages": [
            "https://kpopping.com/kpics/LE-SSERAFIM-3rd-Mini-Album-EASY-Concept-Photo"
        ]
    },
    "crazy": {
        "name": "CRAZY",
        "urls_file": str(BASE / "crazy_urls.txt"),
        "pages": [
            "https://kpopping.com/kpics/le-sserafim-4th-mini-album-crazy-concept-photos"
        ]
    },
    "hot": {
        "name": "HOT",
        "urls_file": str(BASE / "hot_urls.txt"),
        "pages": [
            "https://kpopping.com/kpics/le-sserafim-5th-mini-album-hot-concept-photos"
        ]
    }
}

def download_urls_from_page(page_url, output_file):
    """Extract image URLs from a kpopping page using the Next.js image pattern."""
    print(f"  Scraping: {page_url}")
    # Use curl + grep to extract image URLs from the HTML
    cmd = f"""curl -sL '{page_url}' | grep -oP 'url=https?%3A%2F%2Flegacy\\.kpopping\\.com[^"&]+' | sed 's/url=//' | python3 -c "
import sys, urllib.parse
for line in sys.stdin:
    print(urllib.parse.unquote(line.strip()))
" | sort -u > "{output_file}" """
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    count = 0
    if os.path.exists(output_file):
        with open(output_file) as f:
            count = len(f.readlines())
    print(f"    Found {count} URLs -> {output_file}")
    return count

def download_images(era_key, urls_file):
    """Download all images for an era into the raw folder."""
    if not os.path.exists(urls_file):
        print(f"  No URL file for {era_key}, skipping download")
        return 0
    
    era_dir = DOWNLOAD_DIR / era_key / "raw"
    era_dir.mkdir(parents=True, exist_ok=True)
    
    with open(urls_file) as f:
        urls = [line.strip() for line in f if line.strip()]
    
    count = 0
    for i, url in enumerate(urls):
        ext = url.split('.')[-1] if '.' in url else 'jpeg'
        fname = f"img_{i+1:03d}.{ext}"
        fpath = era_dir / fname
        if fpath.exists():
            continue
        cmd = f"curl -sL -o '{fpath}' '{url}'"
        subprocess.run(cmd, shell=True, capture_output=True)
        if fpath.exists() and os.path.getsize(fpath) > 1000:
            count += 1
        else:
            if fpath.exists():
                os.remove(fpath)
    
    return count

if __name__ == "__main__":
    print("=" * 60)
    print("LE SSERAFIM Concept Photo Downloader")
    print("=" * 60)
    
    # Step 1: Scrape URLs from kpopping pages
    print("\n[Step 1] Scraping image URLs...")
    for era_key, era_info in ERAS.items():
        urls_file = era_info["urls_file"]
        if os.path.exists(urls_file) and os.path.getsize(urls_file) > 50:
            with open(urls_file) as f:
                count = len(f.readlines())
            print(f"  {era_key}: URLs already saved ({count} URLs)")
        else:
            for page_url in era_info["pages"]:
                download_urls_from_page(page_url, urls_file)
    
    # Step 2: Download all images
    print("\n[Step 2] Downloading images...")
    totals = {}
    for era_key in ERAS:
        print(f"\n  --- {era_key.upper()} ---")
        count = download_images(era_key, ERAS[era_key]["urls_file"])
        totals[era_key] = count
        print(f"  Downloaded: {count} new images")
    
    # Step 3: Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for era_key, count in totals.items():
        era_dir = DOWNLOAD_DIR / era_key / "raw"
        existing = len(list(era_dir.glob("*"))) if era_dir.exists() else 0
        print(f"  {era_key:15s} {existing:4d} images in data/concept_photos/{era_key}/raw/")
    
    print("\nNext step: Sort raw images into member folders.")
    print("Run: python3 scripts/sort_photos.py")
