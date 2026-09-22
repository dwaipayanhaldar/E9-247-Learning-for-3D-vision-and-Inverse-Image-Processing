"""
Download COLMAP's "South Building" benchmark (real, non-planar, moving-camera
scene) to use as the real-world "general/non-planar" pair in the geometric
verification experiment, alongside the synthetic planar/panoramic/watermark
pairs. ~174MB, 128 images, 3072x2048.

Usage:
    python download_south_building.py
"""

import subprocess
import zipfile
from pathlib import Path

URL = "https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip"
DATA_DIR = Path(__file__).parent / "data"
ZIP_PATH = DATA_DIR / "south-building.zip"
EXTRACT_DIR = DATA_DIR / "south-building"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.rglob("*.JPG")):
        print(f"Already extracted at {EXTRACT_DIR}")
        return

    if not ZIP_PATH.exists():
        print(f"Downloading {URL} -> {ZIP_PATH}")
        subprocess.run(["curl", "-L", "-C", "-", "-o", str(ZIP_PATH), URL], check=True)
    else:
        print(f"Zip already present at {ZIP_PATH}, skipping download.")

    print(f"Extracting to {EXTRACT_DIR}")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)

    images = list(EXTRACT_DIR.rglob("*.JPG")) + list(EXTRACT_DIR.rglob("*.jpg"))
    print(f"Done. {len(images)} images under {EXTRACT_DIR}")


if __name__ == "__main__":
    main()
