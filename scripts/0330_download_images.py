"""Download all images from CountBench.json into data/"""
import json
import os
import socket
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

with open("labels/CountBench.json") as f:
    data = json.load(f)

for i, item in enumerate(data):
    url = item["image_url"]
    ext = Path(urlparse(url).path).suffix or ".jpg"
    dest = DATA_DIR / f"{i:04d}{ext}"
    if dest.exists():
        print(f"[{i:04d}] already exists, skipping")
        continue
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            dest.write_bytes(resp.read())
        print(f"[{i:04d}] downloaded {dest.name}")
    except Exception as e:
        print(f"[{i:04d}] FAILED {url}: {e}")
