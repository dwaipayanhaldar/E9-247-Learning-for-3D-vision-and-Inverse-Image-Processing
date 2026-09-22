"""
Build image pairs with KNOWN ground-truth geometric configuration, so the
classification experiment has a verifiable answer key instead of just "this
looked planar to me".

- Planar pair:     img2 = homography-warp(img1) for a randomly perturbed
                    4-point projective transform. Any two images related by a
                    single homography are, by definition, images of a plane
                    (or a pure rotation) -- see Hartley & Zisserman Ch. 8 (you
                    have this book in the same folder as this script).
- Panoramic pair:  img2 = warp(img1, K @ R @ inv(K)) for a pure rotation R
                    about the camera center and zero translation -- the
                    textbook "pure rotation induces a homography" identity
                    (H&Z eq. 8.9). This is the exact degenerate case the
                    paper calls out for triangulation.
- Watermark pair:  two UNRELATED images get an identical text/timestamp
                    stamped at the same corner, mimicking the classic
                    watermark/timestamp WTF case from Sec. 4.1.

None of these need a camera or a capture session, which matters if you're
short on time before the deadline -- but you should still capture 1-2 REAL
photo pairs (planar poster, phone rotated in place) to show in the deck that
the synthetic and real cases agree.
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


def load_gray(path, max_dim=1200):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def default_K(w, h, focal_scale=1.2):
    """Heuristic intrinsics (focal = 1.2 * max(w,h), principal point = center),
    the same rule of thumb commonly used when no EXIF/calibration is available."""
    f = focal_scale * max(w, h)
    return np.array([[f, 0, w / 2.0],
                      [0, f, h / 2.0],
                      [0, 0, 1.0]], dtype=np.float64)


def make_planar_pair(img, max_jitter_frac=0.12, rng=None):
    """Warp `img` by a random projective homography of a fronto-parallel
    plane. Returns (img2, H_gt) where x2 = H_gt @ x1 EXACTLY for every pixel."""
    rng = rng or np.random.default_rng(0)
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = max_jitter_frac * min(w, h)
    dst = src + rng.uniform(-jitter, jitter, size=src.shape).astype(np.float32)
    H_gt = cv2.getPerspectiveTransform(src, dst)
    img2 = cv2.warpPerspective(img, H_gt, (w, h))
    return img2, H_gt


def rotation_matrix(yaw_deg, pitch_deg, roll_deg=0.0):
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def make_panoramic_pair(img, yaw_deg=12.0, pitch_deg=4.0, K=None):
    """Warp `img` by H = K @ R @ inv(K), i.e. the image a camera would see if
    it rotated about its own center by (yaw, pitch) with ZERO translation.
    Returns (img2, H_gt, K)."""
    h, w = img.shape[:2]
    if K is None:
        K = default_K(w, h)
    R = rotation_matrix(yaw_deg, pitch_deg)
    H_gt = K @ R @ np.linalg.inv(K)
    H_gt /= H_gt[2, 2]
    img2 = cv2.warpPerspective(img, H_gt, (w, h))
    return img2, H_gt, K


def make_watermark_pair(img_a, img_b, text="2024:01:15 09:41:03",
                         corner="br", margin_frac=0.03, font_size_frac=0.035):
    """Stamp the SAME text at the SAME corner of two otherwise unrelated
    grayscale images, mimicking a camera timestamp burned into the frame."""
    out = []
    for img in (img_a, img_b):
        h, w = img.shape[:2]
        pil = Image.fromarray(img).convert("L")
        draw = ImageDraw.Draw(pil)
        font_size = max(12, int(font_size_frac * min(w, h)))
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        mx, my = int(margin_frac * w), int(margin_frac * h)
        pos = {
            "br": (w - tw - mx, h - th - my),
            "bl": (mx, h - th - my),
            "tr": (w - tw - mx, my),
            "tl": (mx, my),
        }[corner]
        draw.rectangle([pos[0] - 4, pos[1] - 2, pos[0] + tw + 4, pos[1] + th + 6], fill=0)
        draw.text(pos, text, fill=255, font=font)
        out.append(np.array(pil))
    return out[0], out[1]
