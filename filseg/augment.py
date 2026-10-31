"""Crop-first augmentation: every per-pixel operation runs on the output crop only.

Brightness/contrast/gamma are pointwise, so they are applied as a 256-entry lookup table on the
uint8 crop, and the full-frame normalisation statistics are computed exactly from the image
histogram pushed through the same table. Rotation, scaling, flips and cropping are a single
affine warp evaluated only at the output pixels.
"""

import cv2
import numpy as np

IDENTITY_LUT = np.arange(256, dtype=np.float32) / 255.0


def intensity_lut(rng: np.random.Generator) -> np.ndarray:
    x = IDENTITY_LUT.copy()
    if rng.random() < 0.8:
        x = (x - 0.5) * rng.uniform(0.75, 1.3) + 0.5 + rng.uniform(-0.1, 0.1)
    if rng.random() < 0.5:
        x = np.clip(x, 0, 1) ** rng.uniform(0.7, 1.5)
    return np.clip(x, 0, 1).astype(np.float32)


def lut_stats(img: np.ndarray, lut: np.ndarray) -> tuple[float, float]:
    """Mean and std of `lut[img]` over the whole frame, without materialising it."""
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    n = hist.sum()
    mean = (hist * lut).sum() / n
    var = (hist * lut.astype(np.float64) ** 2).sum() / n - mean ** 2
    return float(mean), float(np.sqrt(max(var, 0.0)))


def blur_and_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.2:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    if rng.random() < 0.2:
        img = img + rng.normal(0, rng.uniform(0.01, 0.04), img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def crop_origin(
    target: np.ndarray, size: int, fg_prob: float, rng: np.random.Generator
) -> tuple[int, int]:
    """Top-left corner (x0, y0) of a size x size crop, centred on a filament pixel with probability fg_prob."""
    h, w = target.shape
    if size >= h:
        return 0, 0
    fg = np.flatnonzero(target > 0.5) if rng.random() < fg_prob else np.empty(0)
    if fg.size:
        cy, cx = divmod(int(rng.choice(fg)), w)
        jitter = size // 4
        y0 = cy - size // 2 + int(rng.integers(-jitter, jitter + 1))
        x0 = cx - size // 2 + int(rng.integers(-jitter, jitter + 1))
    else:
        y0, x0 = int(rng.integers(0, h - size + 1)), int(rng.integers(0, w - size + 1))
    return int(np.clip(x0, 0, w - size)), int(np.clip(y0, 0, h - size))


def crop_transform(
    x0: int, y0: int, size: int, angle_deg: float = 0.0, scale: float = 1.0, flip_x: bool = False, flip_y: bool = False
) -> np.ndarray:
    """2x3 matrix mapping output-crop pixel (x, y) to source pixel, for cv2.warpAffine with WARP_INVERSE_MAP.

    Rotation/scale/flip act about the crop centre; with the defaults this is an exact integer crop.
    """
    t = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    flip = np.diag([-1.0 if flip_x else 1.0, -1.0 if flip_y else 1.0])
    a = rot @ flip / scale
    c_out = np.full(2, (size - 1) / 2)
    c_src = np.array([x0, y0]) + c_out
    return np.hstack([a, (c_src - a @ c_out)[:, None]])


def random_crop_transform(
    target: np.ndarray, size: int, fg_prob: float, rng: np.random.Generator, augment: bool, affine_prob: float = 0.7
) -> np.ndarray:
    x0, y0 = crop_origin(target, size, fg_prob, rng)
    if not augment:
        return crop_transform(x0, y0, size)
    rotate = rng.random() < affine_prob
    # Flips swap filament chirality, which is fine because classes are merged.
    return crop_transform(
        x0, y0, size,
        angle_deg=rng.uniform(-180, 180) if rotate else 0.0,
        scale=rng.uniform(0.85, 1.15) if rotate else 1.0,
        flip_x=rng.random() < 0.5,
        flip_y=rng.random() < 0.5,
    )


def warp(img: np.ndarray, m: np.ndarray, size: int) -> np.ndarray:
    return cv2.warpAffine(
        img, m, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
