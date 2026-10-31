"""Box-conditioned single-filament segmentation (YOLO boxes -> one mask per filament).

Each sample is one filament. A window around its box is cut from the full-resolution 2048 image and
resized to `crop_size`; a second input channel marks the box, so the model segments only that filament
even when neighbours share the window. At inference the boxes come from a YOLO detector and the
per-box masks are pasted back into the full frame by `assemble_instances`.
"""

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import IDENTITY_LUT, blur_and_noise, hist_stats, intensity_lut
from .data import ORIG_SIZE, Filament, Record, load_gray


@dataclass(frozen=True)
class Window:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class WindowParams:
    crop_size: int = 256
    margin: float = 0.25      # added on each side, as a fraction of the box side
    min_window: int = 48      # smallest window side in original pixels
    max_aspect: float = 2.0   # windows are widened so long thin boxes keep some context


def filament_xyxy(fil: Filament) -> np.ndarray:
    pts = np.concatenate([np.asarray(p, np.float64).reshape(-1, 2) for p in fil])
    return np.concatenate([pts.min(0), pts.max(0)])


def jitter_box(box: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Perturb each edge by N(0, sigma * box side) to mimic detector error."""
    x0, y0, x1, y1 = box
    w, h = max(x1 - x0, 2.0), max(y1 - y0, 2.0)
    d = rng.normal(0, sigma, 4) * np.array([w, h, w, h])
    nx0, nx1 = sorted((x0 + d[0], x1 + d[2]))
    ny0, ny1 = sorted((y0 + d[1], y1 + d[3]))
    return np.array([nx0, ny0, max(nx1, nx0 + 2), max(ny1, ny0 + 2)])


def make_window(box: np.ndarray, p: WindowParams) -> Window:
    x0, y0, x1, y1 = box
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    cw, ch = max(w * (1 + 2 * p.margin), p.min_window), max(h * (1 + 2 * p.margin), p.min_window)
    cw = max(cw, ch / p.max_aspect)
    ch = max(ch, cw / p.max_aspect)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return Window(int(np.floor(cx - cw / 2)), int(np.floor(cy - ch / 2)),
                  int(np.ceil(cx + cw / 2)), int(np.ceil(cy + ch / 2)))


def _clip(win: Window, h: int, w: int) -> tuple[int, int, int, int]:
    return max(win.y0, 0), min(win.y1, h), max(win.x0, 0), min(win.x1, w)


def crop_resize(img: np.ndarray, win: Window, size: int) -> np.ndarray:
    """Window of `img` (zero-padded outside the frame) resized to size x size."""
    out = np.zeros((win.h, win.w), img.dtype)
    ys0, ys1, xs0, xs1 = _clip(win, *img.shape)
    if ys1 > ys0 and xs1 > xs0:
        out[ys0 - win.y0:ys1 - win.y0, xs0 - win.x0:xs1 - win.x0] = img[ys0:ys1, xs0:xs1]
    interp = cv2.INTER_AREA if max(win.h, win.w) > size else cv2.INTER_LINEAR
    return cv2.resize(out, (size, size), interpolation=interp)


def _to_crop(pts: np.ndarray, win: Window, size: int) -> np.ndarray:
    return (pts - [win.x0, win.y0]) * [size / win.w, size / win.h]


def box_channel(box: np.ndarray, win: Window, size: int) -> np.ndarray:
    m = np.zeros((size, size), np.float32)
    (bx0, by0), (bx1, by1) = np.clip(np.round(_to_crop(np.reshape(box, (2, 2)), win, size)), 0, size).astype(int)
    m[by0:by1, bx0:bx1] = 1
    return m


def filament_mask(fil: Filament, win: Window, size: int, shift_bits: int = 4) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    pts = [np.round(_to_crop(np.asarray(p, np.float64).reshape(-1, 2), win, size) * (1 << shift_bits)).astype(np.int32)
           for p in fil]
    cv2.fillPoly(mask, pts, 1, lineType=cv2.LINE_8, shift=shift_bits)
    return mask.astype(np.float32)


def paste_prob(prob: np.ndarray, win: Window, frame: int = ORIG_SIZE) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Resize a crop probability back to its window, clipped to the frame; returns (prob, (y0, y1, x0, x1))."""
    full = cv2.resize(prob.astype(np.float32), (win.w, win.h), interpolation=cv2.INTER_LINEAR)
    ys0, ys1, xs0, xs1 = _clip(win, frame, frame)
    return full[ys0 - win.y0:ys1 - win.y0, xs0 - win.x0:xs1 - win.x0], (ys0, ys1, xs0, xs1)


def image_hist(img: np.ndarray) -> np.ndarray:
    return np.bincount(img.ravel(), minlength=256)


def instance_inputs(img: np.ndarray, boxes: np.ndarray, p: WindowParams) -> tuple[np.ndarray, list[Window]]:
    """Model inputs (N, 2, S, S) for every box of one full-resolution image, plus their windows."""
    mean, std = hist_stats(image_hist(img), IDENTITY_LUT)
    wins = [make_window(b, p) for b in boxes]
    xs = [np.stack([(IDENTITY_LUT[crop_resize(img, w, p.crop_size)] - mean) / (std + 1e-6),
                    box_channel(b, w, p.crop_size)]) for b, w in zip(boxes, wins)]
    return (np.stack(xs).astype(np.float32) if xs else np.zeros((0, 2, p.crop_size, p.crop_size), np.float32)), wins


class InstanceDataset(Dataset):
    """One item per annotated filament (every annotator); returns ((2, S, S) input, (1, S, S) target)."""

    def __init__(self, records: list[Record], image_dir: Path, params: WindowParams, box_jitter: float = 0.0,
                 augment: bool = False, cache_images: bool = True):
        self.records, self.params = records, params
        self.box_jitter, self.augment = box_jitter, augment
        self.paths = [image_dir / r.file_name for r in records]
        self.items = [(ri, ai, fi) for ri, r in enumerate(records)
                      for ai, a in enumerate(r.annotations) for fi in range(len(a))]
        self.cache = [self._load(i) for i in range(len(records))] if cache_images else None

    def _load(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        img = load_gray(self.paths[i])
        return img, image_hist(img)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        ri, ai, fi = self.items[idx]
        fil = self.records[ri].annotations[ai][fi]
        img, hist = self.cache[ri] if self.cache is not None else self._load(ri)
        rng = np.random.default_rng(random.getrandbits(64))
        size = self.params.crop_size

        box = filament_xyxy(fil)
        if self.box_jitter:
            box = jitter_box(box, self.box_jitter, rng)
        win = make_window(box, self.params)
        lut = intensity_lut(rng) if self.augment else IDENTITY_LUT
        mean, std = hist_stats(hist, lut)
        x = lut[crop_resize(img, win, size)]
        target, b = filament_mask(fil, win, size), box_channel(box, win, size)
        if self.augment:
            x = blur_and_noise(x, rng)
            k, flip = int(rng.integers(4)), rng.random() < 0.5
            x, target, b = (np.rot90(a, k) for a in (x, target, b))
            if flip:
                x, target, b = x[:, ::-1], target[:, ::-1], b[:, ::-1]
        x = (x - mean) / (std + 1e-6)
        inp = np.ascontiguousarray(np.stack([x, b]).astype(np.float32))
        return torch.from_numpy(inp), torch.from_numpy(np.ascontiguousarray(target))[None]


def assemble_instances(
    probs: np.ndarray,
    windows: list[Window],
    scores: np.ndarray,
    conf: float = 0.25,
    threshold: float = 0.5,
    min_area: int = 200,
    nms_iou: float = 0.5,
    frame: int = ORIG_SIZE,
) -> np.ndarray:
    """Paste per-box masks into an int32 label map, most confident box first.

    A mask overlapping an accepted one with IoU > `nms_iou` is a duplicate detection and dropped;
    otherwise contested pixels go to the instance with the higher probability.
    """
    labels = np.zeros((frame, frame), np.int32)
    best = np.zeros((frame, frame), np.float32)
    areas = [0]
    for i in np.argsort(-scores):
        if scores[i] < conf:
            break
        prob, (y0, y1, x0, x1) = paste_prob(probs[i], windows[i], frame)
        m = prob > threshold
        n = int(m.sum())
        if n < min_area:
            continue
        lab_view, best_view = labels[y0:y1, x0:x1], best[y0:y1, x0:x1]
        inter = np.bincount(lab_view[m], minlength=len(areas))[1:]
        if inter.size:
            iou = inter / (n + np.asarray(areas[1:]) - inter)
            if iou.max() > nms_iou:
                continue
        claim = m & (prob > best_view)
        new_id = len(areas)
        lab_view[claim] = new_id
        best_view[claim] = prob[claim]
        areas.append(n)

    sizes = np.bincount(labels.ravel(), minlength=len(areas))
    valid = np.flatnonzero(sizes >= min_area)
    valid = valid[valid > 0]
    remap = np.zeros(len(sizes), np.int32)
    remap[valid] = np.arange(1, len(valid) + 1)
    return remap[labels]
