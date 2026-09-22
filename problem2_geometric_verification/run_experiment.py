"""
Main entry point for the Problem 2 (Geometric Verification & Scene Graph
Augmentation) experiments.

Produces, under results/:
  - pair_classification_table.csv / .png   (Experiment option 1: N_F, N_E, N_H
        and their ratios across general / planar / panoramic / watermark pairs)
  - plane_fraction_sweep.csv / .png        (Experiment option 2a: continuous
        control of N_H/N_F via a synthetic two-plane scene)
  - threshold_sweep.csv / .png             (Experiment option 2b: fix a
        borderline pair, sweep epsilon_HF, show the classification flip)

Run:
    python run_experiment.py
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from two_view_geometry import analyze_pair, MAX_H_INLIER_RATIO, MIN_E_F_INLIER_RATIO
from synthesize_pairs import default_K, make_watermark_pair
import real_datasets as rd
from synthetic_scene import sweep_plane_fraction, sweep_threshold, two_plane_correspondences

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def find_general_pair(image_paths, offsets=(5, 10, 15, 20, 30, 40)):
    """Try a few frame offsets in the South Building sequence and return the
    first pair that COLMAP-style verification calls CALIBRATED/UNCALIBRATED
    (i.e. a genuine, non-degenerate, non-planar general pair)."""
    for off in offsets:
        for start in (0, len(image_paths) // 3, 2 * len(image_paths) // 3):
            i, j = start, start + off
            if j >= len(image_paths):
                continue
            img1 = rd.load_gray(image_paths[i])
            img2 = rd.load_gray(image_paths[j])
            result = analyze_pair(img1, img2, label=f"general/uncalibrated (South Building #{i}-#{j})")
            if result["config"] in ("UNCALIBRATED", "CALIBRATED") and result["num_matches"] >= 30:
                return result, image_paths[i], image_paths[j]
    return None, None, None


# ---------------------------------------------------------------------------
# Experiment 1: category table, built entirely from real captured data
# ---------------------------------------------------------------------------

def run_category_table():
    rows = []

    # 1) General / non-planar, UNCALIBRATED path: real South Building photos.
    sb_images = rd.south_building_images()
    if sb_images:
        print(f"South Building: {len(sb_images)} images.")
        gen_result, gi, gj = find_general_pair(sb_images)
        if gen_result is not None:
            print(f"  general pair: {Path(gi).name} <-> {Path(gj).name}")
            rows.append(gen_result)
        else:
            print("  WARNING: no clean general pair found in the tried offsets.")
    else:
        print("South Building not found under data/south-building -- skipping that row "
              "(run download_south_building.py, or download it yourself and unzip there).")

    # 2) General / non-planar, CALIBRATED path: Middlebury stereo pair with
    #    exact known K -> this is the row that actually exercises N_E/N_F.
    img0, img1, K_mb = rd.load_middlebury_pair()
    rows.append(analyze_pair(img0, img1, K=K_mb, label="general/calibrated (Middlebury stereo pair)"))

    # 3) Planar: HPatches v_wall, with ground-truth H to sanity-check our fit.
    p1, p2, H_gt = rd.load_planar_pair(idx2=3)
    K_planar = default_K(p1.shape[1], p1.shape[0])
    planar_result = analyze_pair(p1, p2, K=K_planar, label="planar (HPatches v_wall, 1-3)")
    if planar_result["H_model"] is not None:
        gt_err = rd.homography_error(planar_result["H_model"], H_gt, p1.shape)
        print(f"  planar pair: RANSAC-fit H vs. ground-truth H_1_3, mean pixel error = {gt_err:.2f}px")
    rows.append(planar_result)

    # 4) Panoramic / pure rotation: real rig capture, zero translation by construction.
    r1, r2 = rd.load_panoramic_pair(frame_i=150, frame_j=650)
    rows.append(analyze_pair(r1, r2, label="panoramic (Pure_rotation rig capture)"))

    # 5) Watermark/WTF: two genuinely unrelated real photos, identical stamped corner text.
    wm_src_a = rd.load_gray(rd.IMAGES_ALL / "3D_general" / "images" / "im0.png")
    wm_src_b = rd.load_gray(sb_images[0]) if sb_images else rd.load_gray(rd.IMAGES_ALL / "Planar" / "v_wall" / "1.ppm")
    wm_src_b = cv2.resize(wm_src_b, (wm_src_a.shape[1], wm_src_a.shape[0]))
    wm_a, wm_b = make_watermark_pair(wm_src_a, wm_src_b)
    rows.append(analyze_pair(wm_a, wm_b, label="watermark/WTF (real images, synthetic stamp)"))

    df = pd.DataFrame(rows)
    model_cols = [c for c in ("F_model", "H_model", "E_model") if c in df.columns]
    df = df.drop(columns=model_cols)  # 3x3 matrices don't belong in a CSV/plot table

    csv_path = RESULTS_DIR / "pair_classification_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}\n")
    print(df.to_string(index=False))

    # Plot: H_F_ratio (and E_F_ratio where available) per category vs thresholds.
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    ax.bar(x - 0.18, df["H_F_ratio"].astype(float), width=0.35, label="N_H / N_F")
    if df["E_F_ratio"].notna().any():
        ax.bar(x + 0.18, df["E_F_ratio"].fillna(0).astype(float), width=0.35, label="N_E / N_F")
    ax.axhline(MAX_H_INLIER_RATIO, color="red", linestyle="--", linewidth=1,
               label=f"epsilon_HF = {MAX_H_INLIER_RATIO}")
    ax.axhline(MIN_E_F_INLIER_RATIO, color="green", linestyle="--", linewidth=1,
               label=f"epsilon_EF = {MIN_E_F_INLIER_RATIO}")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=20, ha="right")
    ax.set_ylabel("inlier ratio")
    ax.set_title("Geometric verification ratios by scene configuration")
    ax.legend()
    fig.tight_layout()
    fig_path = RESULTS_DIR / "pair_classification_table.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path}")
    return df


# ---------------------------------------------------------------------------
# Experiment 2a: plane-fraction sweep (continuous control of N_H/N_F)
# ---------------------------------------------------------------------------

def run_plane_fraction_sweep():
    fractions = np.linspace(0.0, 1.0, 21)
    rows = sweep_plane_fraction(fractions)
    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "plane_fraction_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["plane_fraction"], df["H_F_ratio"], marker="o")
    ax.axhline(MAX_H_INLIER_RATIO, color="red", linestyle="--", label=f"epsilon_HF = {MAX_H_INLIER_RATIO}")
    # The curve is a symmetric U-shape (fraction=0 and fraction=1 are BOTH
    # trivially single-plane scenes; only a genuine mix of the two depths in
    # between is a true general/non-planar scene), so there are two
    # crossings, not one -- annotate both sign changes of (ratio - eps_HF).
    above = df["H_F_ratio"] >= MAX_H_INLIER_RATIO
    crossings = df["plane_fraction"][above != above.shift(fill_value=above.iloc[0])]
    for i, frac in enumerate(crossings):
        ax.axvline(frac, color="gray", linestyle=":",
                   label="classification flips" if i == 0 else None)
    ax.set_xlabel("fraction of correspondences on the foreground plane")
    ax.set_ylabel("N_H / N_F")
    ax.set_title("Synthetic two-plane scene: ratio vs. planar fraction")
    ax.legend()
    fig.tight_layout()
    fig_path = RESULTS_DIR / "plane_fraction_sweep.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path}")
    return df


# ---------------------------------------------------------------------------
# Experiment 2b: fix a borderline pair, sweep epsilon_HF itself
# ---------------------------------------------------------------------------

def run_threshold_sweep(plane_fraction_df):
    # Pick the plane_fraction whose H_F_ratio is closest to the real default
    # threshold, i.e. the genuinely "borderline" synthetic pair.
    idx = (plane_fraction_df["H_F_ratio"] - MAX_H_INLIER_RATIO).abs().idxmin()
    frac = float(plane_fraction_df.loc[idx, "plane_fraction"])
    print(f"\nUsing plane_fraction={frac:.2f} as the borderline pair "
          f"(H_F_ratio={plane_fraction_df.loc[idx, 'H_F_ratio']:.3f}).")

    pts1, pts2, _ = two_plane_correspondences(plane_fraction=frac, rng=np.random.default_rng(0))
    thresholds = np.linspace(0.5, 0.98, 25)
    rows, F_report, H_report = sweep_threshold(pts1, pts2, thresholds)
    df = pd.DataFrame(rows)
    df["N_F"] = F_report.num_inliers
    df["N_H"] = H_report.num_inliers
    csv_path = RESULTS_DIR / "threshold_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4))
    is_planar = (df["config"] == "PLANAR_OR_PANORAMIC").astype(int)
    ax.step(df["epsilon_HF"], is_planar, where="post")
    ax.axvline(MAX_H_INLIER_RATIO, color="red", linestyle="--", label=f"COLMAP default epsilon_HF = {MAX_H_INLIER_RATIO}")
    ax.set_xlabel("epsilon_HF threshold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["UNCALIBRATED/general", "PLANAR_OR_PANORAMIC"])
    ax.set_title(f"Classification flip for a borderline pair (plane_fraction={frac:.2f})")
    ax.legend()
    fig.tight_layout()
    fig_path = RESULTS_DIR / "threshold_sweep.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path}")
    return df


if __name__ == "__main__":
    print("=" * 70)
    print("Experiment 1: N_F / N_E / N_H across general, planar, panoramic, WTF")
    print("=" * 70)
    cat_df = run_category_table()

    print("\n" + "=" * 70)
    print("Experiment 2a: synthetic two-plane scene, sweep planar fraction")
    print("=" * 70)
    frac_df = run_plane_fraction_sweep()

    print("\n" + "=" * 70)
    print("Experiment 2b: fix borderline pair, sweep epsilon_HF threshold")
    print("=" * 70)
    run_threshold_sweep(frac_df)

    print(f"\nAll results written to {RESULTS_DIR}")
