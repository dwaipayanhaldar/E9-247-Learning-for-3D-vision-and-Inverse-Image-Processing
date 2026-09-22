"""
Loaders for the real datasets used in the Problem 2 experiment:

  data/south-building/images/*.JPG
      COLMAP's own "South Building" benchmark. Real, uncalibrated (no focal
      prior), non-planar, moving-camera scene -> exercises
      EstimateUncalibratedTwoViewGeometry (F vs H only).

  data/Images_All/3D_general/images/{im0,im1}.png (+ calib.txt)
      A Middlebury-style stereo pair with EXACT known intrinsics and
      baseline -> exercises EstimateCalibratedTwoViewGeometry (E, F, H all
      three, N_E/N_F check).

  data/Images_All/Planar/v_wall/{1..6}.ppm (+ H_1_2 .. H_1_6)
      An HPatches viewpoint sequence of a (near-)planar wall, with
      ground-truth homographies -> lets us check our RANSAC-fit H against
      the true H_gt, not just against COLMAP's ratio test.

  data/Images_All/Pure_rotation/images/*.png (+ PureRot_Mid_On_GT.txt)
      A real captured sequence from a camera rotated in place (rig capture,
      1384 frames) -> genuine zero-translation / panoramic pairs, no
      synthetic warp required.
"""

import re
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
IMAGES_ALL = DATA_DIR / "Images_All"


def load_gray(path, max_dim=1600):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


# ---------------------------------------------------------------------------
# South Building (general, uncalibrated)
# ---------------------------------------------------------------------------

def south_building_images():
    sb_dir = DATA_DIR / "south-building"
    return sorted(sb_dir.rglob("*.JPG")) + sorted(sb_dir.rglob("*.jpg"))


# ---------------------------------------------------------------------------
# Middlebury-style stereo pair (general, CALIBRATED)
# ---------------------------------------------------------------------------

def load_middlebury_pair():
    d = IMAGES_ALL / "3D_general" / "images"
    calib_text = (d / "calib.txt").read_text()
    nums = re.findall(r"[-+]?\d*\.?\d+", calib_text.split("cam0=")[1].split("]")[0])
    fx, _, cx, _, fy, cy = [float(n) for n in nums[:6]]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    img0 = load_gray(d / "im0.png")
    img1 = load_gray(d / "im1.png")
    return img0, img1, K


# ---------------------------------------------------------------------------
# HPatches v_wall (planar, with ground-truth H)
# ---------------------------------------------------------------------------

def load_planar_pair(idx2=2):
    """idx2 in {2,3,4,5,6}: pairs image 1 with image `idx2`, for which
    H_1_{idx2} is the ground-truth homography 1 -> idx2."""
    d = IMAGES_ALL / "Planar" / "v_wall"
    img1 = load_gray(d / "1.ppm")
    img2 = load_gray(d / f"{idx2}.ppm")
    H_gt = np.loadtxt(d / f"H_1_{idx2}")
    return img1, img2, H_gt


def homography_error(H_est, H_gt, image_shape, n_samples=200, rng=None):
    """Mean pixel reprojection error of H_est vs. ground-truth H_gt over a
    grid of points in image 1, both normalized to the same scale."""
    rng = rng or np.random.default_rng(0)
    h, w = image_shape[:2]
    pts = np.stack([rng.uniform(0, w, n_samples), rng.uniform(0, h, n_samples), np.ones(n_samples)])
    H_est = H_est / H_est[2, 2]
    H_gt = H_gt / H_gt[2, 2]
    p_est = H_est @ pts
    p_gt = H_gt @ pts
    p_est = (p_est[:2] / p_est[2]).T
    p_gt = (p_gt[:2] / p_gt[2]).T
    return float(np.linalg.norm(p_est - p_gt, axis=1).mean())


# ---------------------------------------------------------------------------
# Pure rotation rig capture (panoramic)
# ---------------------------------------------------------------------------

def load_panoramic_pair(frame_i=150, frame_j=650):
    d = IMAGES_ALL / "Pure_rotation" / "images"
    files = sorted(d.glob("*.png"))
    img1 = load_gray(files[frame_i])
    img2 = load_gray(files[frame_j])
    return img1, img2
