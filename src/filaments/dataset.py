"""Training / validation datasets.

Training samples are random crops. Each crop is produced by ONE affine warp from the raw 2048²
image (rotation, flip, scale jitter, translation and the resolution change all at once), and the
disk normalisation is evaluated only at the sampled pixels, so a crop costs crop-sized work.

label_mode
  "separate": every annotation set is its own sample (Kaggle's suggestion; label noise as augmentation)
  "soft":     one sample per observation; target = mean of its annotators' masks (consensus in [0,1])
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data import AnnotationSet, image_path, load_image
from .preprocess import LIMB_MARGIN, ProfileStore, full_image, load_labels, normalise


@dataclass
class CropConfig:
    crop: int = 512            # output crop size (pixels at model resolution)
    scale: float = 0.5         # model resolution / native resolution (0.5 -> disk seen at 1024²)
    pos_frac: float = 0.7      # probability that a crop is centred on a filament pixel
    rotate: bool = True        # free rotation (a solar disk has no preferred orientation)
    scale_jitter: float = 0.1  # +-10 % zoom
    photometric: bool = True   # contrast/brightness/blur/noise (site-to-site variation, EDA §6)


def _affine(center_xy, out_size, scale, angle_deg, flip):
    """2x3 matrix mapping *source* pixel coords -> output crop coords."""
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    R = np.array([[c, -s], [s, c]]) * scale
    if flip:
        R = R @ np.diag([-1.0, 1.0])
    t = np.array([out_size / 2, out_size / 2]) - R @ np.asarray(center_xy)
    return np.c_[R, t].astype(np.float32)


def _photometric(x: np.ndarray, rng: random.Random) -> np.ndarray:
    x = x * rng.uniform(0.8, 1.25) + rng.uniform(-0.3, 0.3)
    if rng.random() < 0.3:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.5))      # poor seeing
    if rng.random() < 0.3:
        x = x + np.random.normal(0, rng.uniform(0.05, 0.25), x.shape).astype(np.float32)
    return x


class CropDataset(Dataset):
    def __init__(self, sets: list[AnnotationSet], cfg: CropConfig, label_mode: str = "separate",
                 samples_per_epoch: int | None = None, seed: int = 0):
        self.cfg, self.label_mode = cfg, label_mode
        by_obs: dict[str, list[AnnotationSet]] = {}
        for s in sets:
            by_obs.setdefault(s.observation, []).append(s)
        if label_mode == "separate":
            self.items = [(s.observation, [s.set_id]) for s in sets]
        elif label_mode == "soft":
            self.items = [(o, [s.set_id for s in g]) for o, g in by_obs.items()]
        else:
            raise ValueError(label_mode)
        self.n = samples_per_epoch or len(self.items)
        self.seed = seed
        self.profiles = None  # opened lazily per worker

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.profiles is None:
            self.profiles = ProfileStore()
        rng = random.Random((self.seed, i, torch.initial_seed()).__hash__())
        obs, set_ids = self.items[rng.randrange(len(self.items))]
        cfg = self.cfg
        raw = load_image(image_path(obs))
        disk, profile, spread = self.profiles(obs)
        labels = [load_labels(sid) for sid in set_ids]

        if rng.random() < cfg.pos_frac:
            lab = labels[rng.randrange(len(labels))]
            ys, xs = np.nonzero(lab)
            k = rng.randrange(len(ys))
            jitter = cfg.crop / cfg.scale / 4
            center = (xs[k] + rng.uniform(-jitter, jitter), ys[k] + rng.uniform(-jitter, jitter))
        else:  # anywhere on the disk
            cx, cy, r = disk
            rr, th = r * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
            center = (cx + rr * math.cos(th), cy + rr * math.sin(th))

        scale = cfg.scale * (1 + rng.uniform(-cfg.scale_jitter, cfg.scale_jitter))
        angle = rng.uniform(0, 360) if cfg.rotate else rng.choice([0, 90, 180, 270])
        M = _affine(center, cfg.crop, scale, angle, flip=rng.random() < 0.5)
        size = (cfg.crop, cfg.crop)

        if scale < 0.75:  # anti-alias before a strong downsample
            raw = cv2.GaussianBlur(raw, (0, 0), 0.5 / scale)
        img = cv2.warpAffine(raw, M, size, flags=cv2.INTER_LINEAR, borderValue=0)
        # rho at each output pixel, via the inverse map (only crop-sized work)
        Minv = cv2.invertAffineTransform(M)
        oy, ox = np.indices(size, dtype=np.float32)
        sx = Minv[0, 0] * ox + Minv[0, 1] * oy + Minv[0, 2]
        sy = Minv[1, 0] * ox + Minv[1, 1] * oy + Minv[1, 2]
        rho = np.hypot(sx - disk[0], sy - disk[1]) / disk[2]
        x = normalise(img, rho, profile, spread)

        masks = [cv2.warpAffine(lab.astype(np.float32), M, size, flags=cv2.INTER_NEAREST) for lab in labels]
        instance = masks[0].astype(np.int32)
        target = np.mean([m > 0 for m in masks], axis=0).astype(np.float32)

        if cfg.photometric:
            x = _photometric(x, rng)
            x[rho > LIMB_MARGIN] = 0.0
        return {
            "image": torch.from_numpy(x[None].astype(np.float32)),
            "mask": torch.from_numpy(target[None]),
            "instances": torch.from_numpy(instance),   # for instance-aware targets (enhancement work)
        }


def full_disk_input(observation: str, scale: float, profiles: ProfileStore | None = None) -> np.ndarray:
    """Whole normalised disk at model resolution: float32 (H*scale, W*scale). Used for val/test."""
    profiles = profiles or ProfileStore()
    raw = load_image(image_path(observation))
    disk, profile, spread = profiles(observation)
    x = full_image(raw, disk, profile, spread)
    if scale != 1.0:
        size = int(round(raw.shape[1] * scale))
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    return x


def load_folds(path) -> pd.DataFrame:
    return pd.read_csv(path)
