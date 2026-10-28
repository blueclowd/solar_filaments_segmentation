"""Turn a filament probability map into individual filament instances."""

import cv2
import numpy as np

from .data import ORIG_SIZE


def upsample(prob: np.ndarray, size: int = ORIG_SIZE) -> np.ndarray:
    if prob.shape[0] == size:
        return prob
    return cv2.resize(prob.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)


def _disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def probs_to_instances(
    prob: np.ndarray,
    threshold: float = 0.5,
    close_radius: int = 3,
    min_area: int = 200,
    merge_distance: int = 12,
) -> np.ndarray:
    """Return an int32 label map (0 = background, 1..N = filaments).

    PQ penalises fragmentation, so nearby components (within `merge_distance` px) are grouped into one
    filament; the grouping only changes labels, never the predicted pixels.
    """
    binary = (prob > threshold).astype(np.uint8)
    if close_radius > 0:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _disk(close_radius))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    binary = keep[labels].astype(np.uint8)

    if merge_distance > 0:
        grown = cv2.dilate(binary, _disk(merge_distance))
        _, groups = cv2.connectedComponents(grown, connectivity=8)
        labels = groups * binary
    else:
        _, labels = cv2.connectedComponents(binary, connectivity=8)

    ids = np.unique(labels)
    ids = ids[ids > 0]
    lut = np.zeros(labels.max() + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[labels]
