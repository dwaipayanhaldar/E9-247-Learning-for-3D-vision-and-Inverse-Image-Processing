"""
Synthetic two-plane 3D scene for the "borderline pair" experiment (option 2
in the assignment: find/construct a pair near the N_H/N_F decision boundary
and show the classification flip as the threshold is perturbed).

Rather than fighting with SIFT noise on a real photo to land near
epsilon_HF = 0.8 by luck, we generate EXACT 2D correspondences directly:
  - `plane_fraction` of points lie on a single fronto-parallel plane at
    depth `plane_depth` -> these satisfy ONE global homography exactly.
  - the remaining points lie on a second plane at a different depth
    `bg_depth` -> these satisfy the epipolar constraint (same F) but, in
    general position, NOT the same H as the foreground plane.
This gives direct, continuous control over N_H/N_F: as plane_fraction -> 1,
the pair becomes indistinguishable from a truly planar scene; as it -> 0,
it's a clean general/non-planar scene. Sweeping plane_fraction sweeps
N_H/N_F across the epsilon_HF = 0.8 decision boundary.
"""

import numpy as np
import cv2

from two_view_geometry import fit_fundamental, fit_homography, classify_two_view_geometry


def camera_matrix(w=1600, h=1200, focal=1400.0):
    return np.array([[focal, 0, w / 2.0],
                      [0, focal, h / 2.0],
                      [0, 0, 1.0]])


def two_plane_correspondences(plane_fraction=0.7, n_points=400,
                               plane_depth=6.0, bg_depth=20.0,
                               baseline=0.6, yaw_deg=4.0,
                               K=None, image_size=(1600, 1200),
                               noise_px=0.3, rng=None):
    """Project random 3D points on two fronto-parallel planes into two
    pinhole views related by a translation `baseline` (+ small rotation),
    and return (pts1, pts2, on_plane_mask)."""
    rng = rng or np.random.default_rng(0)
    w, h = image_size
    K = K if K is not None else camera_matrix(w, h)

    n_plane = int(round(plane_fraction * n_points))
    n_bg = n_points - n_plane

    def random_points(depth, n):
        # Points spread across roughly the first camera's field of view at `depth`.
        half_fov_x = depth * (w / 2.0) / K[0, 0] * 0.9
        half_fov_y = depth * (h / 2.0) / K[1, 1] * 0.9
        X = rng.uniform(-half_fov_x, half_fov_x, n)
        Y = rng.uniform(-half_fov_y, half_fov_y, n)
        Z = np.full(n, depth)
        return np.stack([X, Y, Z], axis=1)

    pts3d = np.vstack([random_points(plane_depth, n_plane),
                        random_points(bg_depth, n_bg)])
    on_plane = np.concatenate([np.ones(n_plane, dtype=bool), np.zeros(n_bg, dtype=bool)])

    yaw = np.radians(yaw_deg)
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                  [0, 1, 0],
                  [-np.sin(yaw), 0, np.cos(yaw)]])
    t = np.array([baseline, 0.0, 0.0])  # camera 2 shifted along +X relative to camera 1

    def project(P, R_c, t_c):
        Pc = (R_c @ P.T).T + t_c
        proj = (K @ Pc.T).T
        return proj[:, :2] / proj[:, 2:3]

    pts1 = project(pts3d, np.eye(3), np.zeros(3))
    # Camera 2 pose relative to camera 1: cam2_from_cam1 = [R | t]
    pts2 = project(pts3d, R, t)

    pts1 += rng.normal(0, noise_px, pts1.shape)
    pts2 += rng.normal(0, noise_px, pts2.shape)

    in_bounds = (
        (pts1[:, 0] >= 0) & (pts1[:, 0] < w) & (pts1[:, 1] >= 0) & (pts1[:, 1] < h) &
        (pts2[:, 0] >= 0) & (pts2[:, 0] < w) & (pts2[:, 1] >= 0) & (pts2[:, 1] < h)
    )
    return (pts1[in_bounds].astype(np.float32),
            pts2[in_bounds].astype(np.float32),
            on_plane[in_bounds])


def sweep_plane_fraction(fractions, **kwargs):
    """For each plane_fraction, fit F and H and return N_F, N_H, H_F_ratio."""
    rows = []
    for frac in fractions:
        pts1, pts2, _ = two_plane_correspondences(plane_fraction=frac, rng=np.random.default_rng(0), **kwargs)
        F_report = fit_fundamental(pts1, pts2)
        H_report = fit_homography(pts1, pts2)
        result = classify_two_view_geometry(F_report, H_report)
        rows.append(dict(plane_fraction=frac, N_F=F_report.num_inliers,
                          N_H=H_report.num_inliers, H_F_ratio=result["H_F_ratio"],
                          config=result["config"]))
    return rows


def sweep_threshold(pts1, pts2, thresholds):
    """Fix a pair (pts1, pts2), vary epsilon_HF, and report the classification
    at each threshold value -- directly answers "how does the ambiguity get
    resolved (or not)" for a borderline pair."""
    F_report = fit_fundamental(pts1, pts2)
    H_report = fit_homography(pts1, pts2)
    rows = []
    for eps in thresholds:
        result = classify_two_view_geometry(F_report, H_report, max_H_inlier_ratio=eps)
        rows.append(dict(epsilon_HF=eps, H_F_ratio=result["H_F_ratio"], config=result["config"]))
    return rows, F_report, H_report
