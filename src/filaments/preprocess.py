"""Disk-aware preprocessing and the on-disk cache it needs.

Per image we store the fitted disk (cx, cy, r) and a robust radial intensity profile (median per
radial bin = limb darkening). An image is normalised as

    flat = raw / profile(rho)            # removes centre-to-limb darkening (rho = r / R_sun)
    x    = (flat - 1) / spread           # spread = robust std of flat on the disk
    x    = 0 outside the limb            # the off-limb halo carries no filaments

so filaments are negative, plage positive, and all sites share one intensity scale (EDA §6).
Label maps (instance ids per annotation set) are rasterised once and cached as PNG.

Build the cache with:  python -m filaments.preprocess
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from .data import (PROJECT_ROOT, TEST_IMAGES, TRAIN_IMAGES, AnnotationSet, fit_disk, load_annotation_sets,
                   load_image)

# Must be writable (the Kaggle input dir is read-only).
CACHE_DIR = Path(os.environ.get("FILAMENT_CACHE", PROJECT_ROOT / "data" / "processed"))
LABEL_DIR = CACHE_DIR / "labels"
PROFILE_FILE = CACHE_DIR / "profiles.npz"

N_BINS = 256          # radial bins over rho in [0, 1]
LIMB_MARGIN = 0.99    # cut just inside the limb: the JPEG limb has a bright/dark ring artifact
CLIP = 8.0            # |x| clip in robust-std units


def radial_profile(img: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    """Median intensity per radial bin, lightly smoothed. Medians ignore filaments/plage."""
    yy, xx = np.indices(img.shape)
    rho = np.hypot(xx - cx, yy - cy) / r
    on = rho < 1.0
    bins = np.minimum((rho[on] * N_BINS).astype(int), N_BINS - 1)
    prof = np.asarray(ndimage.median(img[on].astype(np.float32), labels=bins, index=np.arange(N_BINS)))
    prof = np.nan_to_num(prof, nan=np.nanmedian(prof))
    k = np.ones(5) / 5
    smooth = np.convolve(np.pad(prof, 2, mode="edge"), k, mode="valid")
    return np.maximum(smooth, 1.0).astype(np.float32)


def normalise(raw: np.ndarray, rho: np.ndarray, profile: np.ndarray, spread: float) -> np.ndarray:
    """raw: sampled intensities; rho: r/R at the same pixels (any shape). Returns float32 x."""
    idx = np.clip(rho * N_BINS, 0, N_BINS - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, N_BINS - 1)
    w = (idx - lo).astype(np.float32)
    prof = profile[lo] * (1 - w) + profile[hi] * w
    x = (raw.astype(np.float32) / prof - 1.0) / spread
    x[rho > LIMB_MARGIN] = 0.0
    return np.clip(x, -CLIP, CLIP)


def full_image(raw: np.ndarray, disk: tuple[float, float, float], profile: np.ndarray, spread: float) -> np.ndarray:
    cx, cy, r = disk
    yy, xx = np.indices(raw.shape, dtype=np.float32)
    rho = np.hypot(xx - cx, yy - cy) / r
    return normalise(raw, rho, profile, spread)


# ---------------------------------------------------------------- cache building

def _image_entry(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    raw = load_image(path)
    disk = fit_disk(raw)
    prof = radial_profile(raw, *disk)
    flat = full_image(raw, disk, prof, spread=1.0)  # x with spread 1 = flat - 1
    cx, cy, r = disk
    yy, xx = np.indices(raw.shape)
    inner = np.hypot(xx - cx, yy - cy) < 0.95 * r
    v = flat[inner]
    spread = float(1.4826 * np.median(np.abs(v - np.median(v))))
    return path.stem, np.array([*disk, spread], np.float32), prof


def _label_entry(s: AnnotationSet) -> None:
    out = LABEL_DIR / f"{s.set_id}.png"
    if not out.exists():
        lab = s.label_map()
        assert lab.max() < 2**16
        cv2.imwrite(str(out), lab.astype(np.uint16))


def build_cache(workers: int = 8) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    if not PROFILE_FILE.exists():
        paths = sorted(TRAIN_IMAGES.glob("*.jpeg")) + sorted(TEST_IMAGES.glob("*.jpeg"))
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_image_entry, paths, chunksize=8))
        np.savez(PROFILE_FILE, names=np.array([r[0] for r in rows]),
                 geom=np.stack([r[1] for r in rows]), profiles=np.stack([r[2] for r in rows]))
    sets = load_annotation_sets()
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_label_entry, sets, chunksize=8))
    (CACHE_DIR / "README.json").write_text(json.dumps({
        "profiles": "names, geom=[cx, cy, r, spread], profiles=[N, N_BINS] median radial intensity",
        "labels": "uint16 instance label maps per annotation set (0 = background)",
    }, indent=2))


class ProfileStore:
    """Lookup of per-image disk geometry, spread and radial profile."""

    def __init__(self, path: Path = PROFILE_FILE):
        z = np.load(path)
        self.index = {str(n): i for i, n in enumerate(z["names"])}
        self.geom, self.profiles = z["geom"], z["profiles"]

    def __call__(self, observation: str) -> tuple[tuple[float, float, float], np.ndarray, float]:
        i = self.index[observation]
        cx, cy, r, spread = self.geom[i]
        return (float(cx), float(cy), float(r)), self.profiles[i], float(spread)


def load_labels(set_id: str) -> np.ndarray:
    lab = cv2.imread(str(LABEL_DIR / f"{set_id}.png"), cv2.IMREAD_UNCHANGED)
    if lab is None:
        raise FileNotFoundError(set_id)
    return lab.astype(np.int32)


if __name__ == "__main__":
    build_cache()
    print("cache ready:", CACHE_DIR)
