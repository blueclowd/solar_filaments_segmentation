"""Group U-Net filament pixels into instances using YOLO boxes.

The U-Net decides *which pixels* are filament; YOLO decides *which pixels belong together*:
  * fragments lying mostly inside one box become one filament (fixes over-fragmentation);
  * a component with at least `split_min_frac` of its pixels in a second box is split pixel-wise
    between the boxes (fixes merged neighbours; 1.0 disables splitting);
  * components outside every box are kept as their own filaments or dropped (`keep_unmatched`).
"""

import cv2
import numpy as np


def combine_instances(
    prob: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
    close_radius: int = 3,
    conf: float = 0.25,
    min_cover: float = 0.5,
    split_min_frac: float = 0.2,
    keep_unmatched: bool = True,
    fragment_min_area: int = 30,
    min_area: int = 200,
) -> np.ndarray:
    """Return an int32 label map (0 = background). `boxes` are xyxy pixels in `prob`'s frame."""
    h, w = prob.shape
    binary = (prob > threshold).astype(np.uint8)
    if close_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1,) * 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    n, comp, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]

    keep = scores >= conf
    boxes = boxes[keep][np.argsort(-scores[keep])]
    nb = len(boxes)
    # owner[y, x] = 1 + index of the most confident box covering the pixel (0 = no box)
    owner = np.zeros((h, w), np.int32)
    for j in range(nb - 1, -1, -1):
        x0, y0, x1, y1 = np.clip(np.round(boxes[j]).astype(int), 0, [w, h, w, h])
        owner[y0:y1, x0:x1] = j + 1
    cover = np.bincount((comp.astype(np.int64) * (nb + 1) + owner).ravel(), minlength=n * (nb + 1)).reshape(n, nb + 1)

    lut = np.zeros(n, np.int64)
    split, next_label = [], nb + 1
    for c in range(1, n):
        if area[c] < fragment_min_area:
            continue
        inside = cover[c, 1:]
        second = np.partition(inside, -2)[-2] if nb > 1 else 0
        if nb and inside.sum() >= min_cover * area[c] and second >= split_min_frac * area[c]:
            split.append(c)
        elif nb and inside.max() >= min_cover * area[c]:
            lut[c] = inside.argmax() + 1
        elif nb and inside.sum() >= min_cover * area[c]:
            if split_min_frac < 1:
                split.append(c)
            else:
                lut[c] = inside.argmax() + 1
        elif keep_unmatched and area[c] >= min_area:
            lut[c] = next_label
            next_label += 1
    labels = lut[comp]
    for c in split:
        m = comp == c
        o = owner[m]
        o[o == 0] = cover[c, 1:].argmax() + 1
        labels[m] = o

    sizes = np.bincount(labels.ravel())
    valid = np.flatnonzero(sizes >= min_area)
    valid = valid[valid > 0]
    remap = np.zeros(len(sizes), np.int32)
    remap[valid] = np.arange(1, len(valid) + 1)
    return remap[labels]
