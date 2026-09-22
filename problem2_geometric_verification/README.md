# Problem 2 — Geometric Verification & Scene Graph Augmentation

Reimplements COLMAP's multi-model verification decision tree (`EstimateCalibratedTwoViewGeometry`
/ `EstimateUncalibratedTwoViewGeometry` / `DetectWatermarkMatches` in
`colmap/src/colmap/estimators/two_view_geometry.cc`) in Python/OpenCV, using COLMAP's real
default thresholds, and runs it on real data.

## Setup

```
pip install opencv-python pillow pycolmap pandas matplotlib numpy
```

## Data

Already in `data/`:
- `south-building/` — COLMAP's own benchmark (real, uncalibrated, non-planar). If missing,
  run `python download_south_building.py` (needs normal internet access — GitHub's release-asset
  CDN was blocked from the agent sandbox that built this, so it was fetched manually).
- `Images_All/3D_general/` — Middlebury-style stereo pair with exact calibration (`calib.txt`).
- `Images_All/Planar/v_wall/` — HPatches `v_wall` sequence with ground-truth homographies (`H_1_2`...`H_1_6`).
- `Images_All/Pure_rotation/` — real rig capture of a camera rotated in place (1384 frames + GT pose log).

## Run

```
python run_experiment.py
```

Writes to `results/`:
- `pair_classification_table.{csv,png}` — **Experiment option 1**: N_F, N_E, N_H and their
  ratios across general/calibrated, general/uncalibrated, planar, panoramic, and watermark pairs.
- `plane_fraction_sweep.{csv,png}` — **Experiment option 2a**: a synthetic two-plane 3D scene
  (`synthetic_scene.py`) gives exact, continuous control over N_H/N_F via the fraction of points
  on a single plane, showing the classification flip on both sides of epsilon_HF = 0.8.
- `threshold_sweep.{csv,png}` — **Experiment option 2b**: fixes the pair whose ratio landed
  closest to 0.8 and sweeps the threshold itself, showing the exact flip point.

## Files

- `two_view_geometry.py` — the reusable core: F/H/E RANSAC fits, COLMAP's classification
  decision tree, and the watermark/WTF detector. Every function docstring cites the exact
  `two_view_geometry.cc` line range it mirrors.
- `real_datasets.py` — loaders for the four real datasets above.
- `synthesize_pairs.py` — fallback synthetic planar/panoramic/watermark pair generators
  (used automatically for the watermark stamp; useful if you don't have real data on hand).
- `synthetic_scene.py` — the two-plane synthetic 3D scene for the borderline/threshold experiments.

## Noteworthy results to highlight in the presentation

1. The Middlebury stereo pair (`general/calibrated`) — a REAL scene with ground-truth depth
   variation (`disp0GT.pfm` proves it's non-planar) — gets classified **PLANAR_OR_PANORAMIC**
   by COLMAP's own ratio test (H_F_ratio ≈ 0.98, H_E_ratio ≈ 0.98), because the stereo baseline
   (59.6mm) is narrow relative to scene depth, so a homography fits almost as well as the true
   epipolar geometry at a 4px threshold. This is a genuine, data-backed answer to "can a valid H
   fit occur for a pair that isn't actually safe to triangulate broadly" — no synthetic
   construction needed, it happened on real data.

2. `pure_rotation_self_calibration.py` (run standalone) — the `Pure_rotation` row's `N_E` is
   intentionally left blank. Its `PureRot_Mid_On_GT.txt` gives real per-frame rotations, so
   self-calibrating K from `H = K R K^-1` looks doable in principle, but every pair's relative
   rotation has axis ≈ `[0,0,±1]`: the rig does a pure ROLL about the camera's own optical axis,
   not a pan/tilt. For a roll about the optical axis, `K·Rz(θ)·K⁻¹ = Rz(θ)` exactly for *every*
   focal length — the image motion carries zero information about K, verified both algebraically
   and by a self-calibration fit that swings wildly / hugs its bounds depending on which frame
   pairs are used. This is a second, independent, data-backed reason (beyond scale/parallax)
   the paper treats panoramic pairs as a dead end for triangulation (Sec 4.1).
