"""Numpy/OpenCV augmentations. Images are float32 in [0, 1]; targets are float32 masks in [0, 1]."""

import cv2
import numpy as np


def augment_intensity(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.8:
        img = (img - 0.5) * rng.uniform(0.75, 1.3) + 0.5 + rng.uniform(-0.1, 0.1)
    if rng.random() < 0.5:
        img = np.clip(img, 0, 1) ** rng.uniform(0.7, 1.5)
    if rng.random() < 0.2:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    if rng.random() < 0.2:
        img = img + rng.normal(0, rng.uniform(0.01, 0.04), img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


def augment_geometric(
    img: np.ndarray, target: np.ndarray, rng: np.random.Generator, affine_prob: float = 0.7
) -> tuple[np.ndarray, np.ndarray]:
    # Flips swap filament chirality, which is fine because classes are merged.
    if rng.random() < 0.5:
        img, target = img[:, ::-1], target[:, ::-1]
    if rng.random() < 0.5:
        img, target = img[::-1], target[::-1]
    if rng.random() < affine_prob:
        h, w = img.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-180, 180), rng.uniform(0.85, 1.15))
        warp = dict(dsize=(w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        img = cv2.warpAffine(np.ascontiguousarray(img), m, flags=cv2.INTER_LINEAR, **warp)
        target = cv2.warpAffine(np.ascontiguousarray(target), m, flags=cv2.INTER_LINEAR, **warp)
    return np.ascontiguousarray(img), np.ascontiguousarray(target)


def random_crop(
    img: np.ndarray, target: np.ndarray, size: int, fg_prob: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape
    if size >= h:
        return img, target
    fg = np.flatnonzero(target > 0.5) if rng.random() < fg_prob else np.empty(0)
    if fg.size:
        cy, cx = divmod(int(rng.choice(fg)), w)
        jitter = size // 4
        y0 = cy - size // 2 + int(rng.integers(-jitter, jitter + 1))
        x0 = cx - size // 2 + int(rng.integers(-jitter, jitter + 1))
    else:
        y0, x0 = int(rng.integers(0, h - size + 1)), int(rng.integers(0, w - size + 1))
    y0, x0 = int(np.clip(y0, 0, h - size)), int(np.clip(x0, 0, w - size))
    return img[y0:y0 + size, x0:x0 + size], target[y0:y0 + size, x0:x0 + size]
