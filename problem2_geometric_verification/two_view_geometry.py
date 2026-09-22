"""
Reimplementation of COLMAP's multi-model geometric verification, mirrored
line-for-line against the production C++ so the numbers this script produces
can be checked against the real pipeline.

Primary reference (from the cloned colmap/ repo, src/colmap/estimators/two_view_geometry.cc):
  - TwoViewGeometryOptions defaults ........... two_view_geometry.h:18-111
  - EstimateCalibratedTwoViewGeometry ......... two_view_geometry.cc:1075-1235
      (fits E, F, H jointly; computes E_F/H_F/H_E inlier ratios; classifies)
  - EstimateUncalibratedTwoViewGeometry ....... two_view_geometry.cc:377-466
      (F vs H only, used when no focal-length prior is available)
  - DetectWatermarkMatches .................... two_view_geometry.cc:1550-1615
      (border-region check + pure-translation RANSAC -> WTF pairs)

Paper reference: Schoenberger & Frahm, "Structure-from-Motion Revisited",
CVPR 2016, Section 4.1 "Scene Graph Augmentation".
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# COLMAP defaults, copied verbatim from TwoViewGeometryOptions
# (src/colmap/estimators/two_view_geometry.h:18-111)
# ---------------------------------------------------------------------------
MIN_NUM_INLIERS = 15                 # options.min_num_inliers
MIN_E_F_INLIER_RATIO = 0.95          # options.min_E_F_inlier_ratio  (paper's epsilon_EF)
MAX_H_INLIER_RATIO = 0.80            # options.max_H_inlier_ratio   (paper's epsilon_HF)
WATERMARK_MIN_INLIER_RATIO = 0.70    # options.watermark_min_inlier_ratio
WATERMARK_BORDER_SIZE = 0.10         # options.watermark_border_size (fraction of diagonal)
RANSAC_MAX_ERROR = 4.0               # options.ransac_options.max_error (px)
WATERMARK_MAX_ERROR = 4.0            # options.watermark_detection_max_error (px)
RANSAC_CONFIDENCE = 0.999            # options.ransac_options.confidence


@dataclass
class ModelReport:
    name: str
    model: Optional[np.ndarray]
    num_inliers: int
    inlier_mask: np.ndarray  # bool, shape (N,)


def fit_fundamental(pts1, pts2, max_error=RANSAC_MAX_ERROR):
    """cv2 analogue of the LO-RANSAC fundamental-matrix fit in
    EstimateUncalibratedTwoViewGeometry / EstimateCalibratedTwoViewGeometry."""
    n = len(pts1)
    if n < 8:
        return ModelReport("F", None, 0, np.zeros(n, dtype=bool))
    F, mask = cv2.findFundamentalMat(
        pts1, pts2, cv2.FM_RANSAC, max_error, RANSAC_CONFIDENCE, maxIters=10000
    )
    if F is None or mask is None:
        return ModelReport("F", None, 0, np.zeros(n, dtype=bool))
    if F.shape[0] > 3:  # OpenCV can return multiple stacked solutions; keep the best
        F = F[:3]
    mask = mask.ravel().astype(bool)
    return ModelReport("F", F, int(mask.sum()), mask)


def fit_homography(pts1, pts2, max_error=RANSAC_MAX_ERROR):
    """cv2 analogue of HomographyMatrixEstimator + LO-RANSAC."""
    n = len(pts1)
    if n < 4:
        return ModelReport("H", None, 0, np.zeros(n, dtype=bool))
    H, mask = cv2.findHomography(
        pts1, pts2, cv2.RANSAC, max_error, maxIters=10000, confidence=RANSAC_CONFIDENCE
    )
    if H is None or mask is None:
        return ModelReport("H", None, 0, np.zeros(n, dtype=bool))
    mask = mask.ravel().astype(bool)
    return ModelReport("H", H, int(mask.sum()), mask)


def fit_essential(pts1, pts2, K, max_error=RANSAC_MAX_ERROR):
    """cv2 analogue of EssentialMatrixTangentSampsonEstimator + LO-RANSAC.
    Requires known intrinsics K (both images assumed to share K here, for
    simplicity -- COLMAP handles per-image cameras via CamRayFromImg)."""
    n = len(pts1)
    if n < 5:
        return ModelReport("E", None, 0, np.zeros(n, dtype=bool))
    E, mask = cv2.findEssentialMat(
        pts1, pts2, cameraMatrix=K, method=cv2.RANSAC,
        prob=RANSAC_CONFIDENCE, threshold=max_error, maxIters=10000
    )
    if E is None or mask is None:
        return ModelReport("E", None, 0, np.zeros(n, dtype=bool))
    if E.shape[0] > 3:
        E = E[:3]
    mask = mask.ravel().astype(bool)
    return ModelReport("E", E, int(mask.sum()), mask)


def fit_translation_ransac(pts1, pts2, max_error=WATERMARK_MAX_ERROR,
                            confidence=RANSAC_CONFIDENCE, max_iters=10000, rng=None):
    """Manual RANSAC for a pure 2D translation model, mirroring
    TranslationTransformEstimator<2> as used inside DetectWatermarkMatches
    (two_view_geometry.cc:1607-1609). OpenCV has no built-in for this."""
    rng = rng or np.random.default_rng(0)
    n = len(pts1)
    if n == 0:
        return ModelReport("T", None, 0, np.zeros(0, dtype=bool))
    disp = np.asarray(pts2) - np.asarray(pts1)
    best_mask = np.zeros(n, dtype=bool)
    best_count = 0
    trials, trials_needed = 0, max_iters
    while trials < min(trials_needed, max_iters):
        t = disp[rng.integers(0, n)]
        err = np.linalg.norm(disp - t, axis=1)
        mask = err <= max_error
        count = int(mask.sum())
        if count > best_count:
            best_count, best_mask = count, mask
            inlier_ratio = max(count / n, 1e-6)
            denom = np.log(max(1 - inlier_ratio, 1e-12))
            trials_needed = int(np.log(1 - confidence) / denom) if denom < 0 else max_iters
        trials += 1
    t_final = disp[best_mask].mean(axis=0) if best_count else None
    return ModelReport("T", t_final, best_count, best_mask)


def classify_two_view_geometry(F_report, H_report, E_report=None,
                                min_num_inliers=MIN_NUM_INLIERS,
                                min_E_F_inlier_ratio=MIN_E_F_INLIER_RATIO,
                                max_H_inlier_ratio=MAX_H_INLIER_RATIO):
    """Reproduces the classification decision tree exactly:
      - with E_report:    EstimateCalibratedTwoViewGeometry, cc:1148-1207
      - without E_report: EstimateUncalibratedTwoViewGeometry, cc:425-441
    Returns a dict with config, winning inlier count/mask, and the three
    ratios the paper calls N_E/N_F and N_H/N_F (Sec 4.1)."""

    def passed(r):
        return r is not None and r.model is not None and r.num_inliers >= min_num_inliers

    def ratio(a, b):
        if b is None or b.num_inliers == 0:
            return float("inf")
        return a.num_inliers / b.num_inliers

    if not (passed(F_report) or passed(H_report) or passed(E_report)):
        return dict(config="DEGENERATE", num_inliers=0, inlier_mask=None,
                    E_F_ratio=None, H_F_ratio=None, H_E_ratio=None)

    E_F_ratio = ratio(E_report, F_report) if E_report is not None else None
    H_F_ratio = ratio(H_report, F_report)
    H_E_ratio = ratio(H_report, E_report) if E_report is not None else None

    num_inliers, mask, config = 0, None, "DEGENERATE"

    if E_report is not None and passed(E_report) and E_F_ratio > min_E_F_inlier_ratio:
        # Calibrated configuration: E is consistent with the given focal length.
        if E_report.num_inliers >= F_report.num_inliers:
            num_inliers, mask = E_report.num_inliers, E_report.inlier_mask
        else:
            num_inliers, mask = F_report.num_inliers, F_report.inlier_mask
        if H_E_ratio > max_H_inlier_ratio:
            config = "PLANAR_OR_PANORAMIC"
            if H_report.num_inliers > num_inliers:
                num_inliers, mask = H_report.num_inliers, H_report.inlier_mask
        else:
            config = "CALIBRATED"
    elif passed(F_report):
        num_inliers, mask = F_report.num_inliers, F_report.inlier_mask
        if H_F_ratio > max_H_inlier_ratio:
            config = "PLANAR_OR_PANORAMIC"
            if H_report.num_inliers > num_inliers:
                num_inliers, mask = H_report.num_inliers, H_report.inlier_mask
        else:
            config = "UNCALIBRATED"
    elif passed(H_report):
        num_inliers, mask, config = H_report.num_inliers, H_report.inlier_mask, "PLANAR_OR_PANORAMIC"

    return dict(config=config, num_inliers=num_inliers, inlier_mask=mask,
                E_F_ratio=E_F_ratio, H_F_ratio=H_F_ratio, H_E_ratio=H_E_ratio)


def detect_watermark(pts1, pts2, inlier_mask, shape1, shape2,
                      border_size=WATERMARK_BORDER_SIZE,
                      min_inlier_ratio=WATERMARK_MIN_INLIER_RATIO,
                      max_error=WATERMARK_MAX_ERROR, rng=None):
    """Mirrors DetectWatermarkMatches (two_view_geometry.cc:1550-1615): a pair
    is WTF if >= min_inlier_ratio of the WINNING model's inliers sit outside a
    `border_size`-of-diagonal margin on BOTH images, AND that same fraction is
    explained by a pure 2D translation."""
    h1, w1 = shape1[:2]
    h2, w2 = shape2[:2]
    diag1, diag2 = np.hypot(w1, h1), np.hypot(w2, h2)
    b1, b2 = border_size * diag1, border_size * diag2

    idx = np.where(inlier_mask)[0] if inlier_mask is not None else np.array([], dtype=int)
    num_inliers = len(idx)
    if num_inliers == 0:
        return dict(is_watermark=False, border_ratio=0.0, translation_inlier_ratio=None)

    p1, p2 = np.asarray(pts1)[idx], np.asarray(pts2)[idx]
    in_center1 = (p1[:, 0] > b1) & (p1[:, 0] < w1 - b1) & (p1[:, 1] > b1) & (p1[:, 1] < h1 - b1)
    in_center2 = (p2[:, 0] > b2) & (p2[:, 0] < w2 - b2) & (p2[:, 1] > b2) & (p2[:, 1] < h2 - b2)
    in_border = (~in_center1) & (~in_center2)
    border_ratio = float(in_border.mean())

    if border_ratio < min_inlier_ratio:
        return dict(is_watermark=False, border_ratio=border_ratio, translation_inlier_ratio=None)

    t_report = fit_translation_ransac(p1, p2, max_error=max_error, rng=rng)
    translation_inlier_ratio = t_report.num_inliers / num_inliers
    is_watermark = translation_inlier_ratio >= min_inlier_ratio
    return dict(is_watermark=is_watermark, border_ratio=border_ratio,
                translation_inlier_ratio=translation_inlier_ratio)


# ---------------------------------------------------------------------------
# End-to-end pipeline: SIFT -> ratio test -> F/H(/E) -> classify -> WTF check
# ---------------------------------------------------------------------------

def extract_sift(img_gray, n_features=8000):
    sift = cv2.SIFT_create(nfeatures=n_features)
    kp, desc = sift.detectAndCompute(img_gray, None)
    return kp, desc


def match_lowe_ratio(desc1, desc2, ratio=0.8):
    """Lowe's ratio test, matching COLMAP's default SiftMatchingOptions.max_ratio=0.8."""
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    return good


def analyze_pair(img1_gray, img2_gray, K=None, ratio_test=0.8, label=""):
    """Full pipeline for one image pair. Returns a flat dict of results
    ready to be appended to a results table (see run_experiment.py)."""
    kp1, desc1 = extract_sift(img1_gray)
    kp2, desc2 = extract_sift(img2_gray)
    matches = match_lowe_ratio(desc1, desc2, ratio=ratio_test)

    if len(matches) < 8:
        return dict(label=label, num_matches=len(matches), config="DEGENERATE",
                    N_F=0, N_E=None, N_H=0, E_F_ratio=None, H_F_ratio=None, H_E_ratio=None,
                    is_watermark=False, border_ratio=None, translation_inlier_ratio=None,
                    F_model=None, H_model=None, E_model=None)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    F_report = fit_fundamental(pts1, pts2)
    H_report = fit_homography(pts1, pts2)
    E_report = fit_essential(pts1, pts2, K) if K is not None else None

    result = classify_two_view_geometry(F_report, H_report, E_report)

    wtf = detect_watermark(pts1, pts2, result["inlier_mask"], img1_gray.shape, img2_gray.shape)

    return dict(
        label=label,
        num_matches=len(matches),
        config=result["config"],
        N_F=F_report.num_inliers,
        N_E=(E_report.num_inliers if E_report is not None else None),
        N_H=H_report.num_inliers,
        E_F_ratio=result["E_F_ratio"],
        H_F_ratio=result["H_F_ratio"],
        H_E_ratio=result["H_E_ratio"],
        is_watermark=wtf["is_watermark"],
        border_ratio=wtf["border_ratio"],
        translation_inlier_ratio=wtf["translation_inlier_ratio"],
        F_model=F_report.model,
        H_model=H_report.model,
        E_model=(E_report.model if E_report is not None else None),
    )
