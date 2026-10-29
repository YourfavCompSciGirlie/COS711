"""MAGFiLO (Kaggle 2026) loading and leakage-safe train/validation splitting.

Naming:
  observation  = image file stem, e.g. 20160920230134Lh  (YYYYMMDDHHMMSS + 2-letter GONG site)
  annotation set = COCO image entry, e.g. 010401-20160920230134Lh (annotator batch + observation)
One observation can carry several independent annotation sets. Splits are always made over
observations (never annotation sets), grouped further by time so near-duplicate days don't leak.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .rle import polygon_to_mask

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# On Kaggle: FILAMENT_DATA=/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026
DATA_ROOT = Path(os.environ.get("FILAMENT_DATA", PROJECT_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"))
TRAIN_JSON = DATA_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_IMAGES = DATA_ROOT / "train" / "train_images"
TEST_IMAGES = DATA_ROOT / "test" / "test_images"

SITES = {"Bh": "Big Bear", "Mh": "Mauna Loa", "Lh": "Learmonth", "Uh": "Udaipur",
         "Th": "El Teide", "Ch": "Cerro Tololo"}


def parse_observation(stem: str) -> tuple[datetime, str]:
    """'20160920230134Lh' -> (datetime, 'Lh')."""
    return datetime.strptime(stem[:14], "%Y%m%d%H%M%S"), stem[14:16]


@dataclass
class AnnotationSet:
    set_id: str            # COCO image id, e.g. 010401-20160920230134Lh
    observation: str       # 20160920230134Lh
    file_name: str
    height: int
    width: int
    polygons: list[list[float]]
    categories: list[int]

    @property
    def time(self) -> datetime:
        return parse_observation(self.observation)[0]

    @property
    def site(self) -> str:
        return parse_observation(self.observation)[1]

    def instance_masks(self) -> list[np.ndarray]:
        return [polygon_to_mask(p, self.height, self.width) for p in self.polygons]

    def label_map(self) -> np.ndarray:
        """Instance label map; on overlaps the later polygon wins."""
        labels = np.zeros((self.height, self.width), dtype=np.int32)
        for k, m in enumerate(self.instance_masks(), start=1):
            labels[m] = k
        return labels


def load_annotation_sets(path: Path = TRAIN_JSON) -> list[AnnotationSet]:
    coco = json.loads(Path(path).read_text())
    by_image: dict = {img["id"]: img for img in coco["images"]}
    anns: dict = {k: [] for k in by_image}
    for a in coco["annotations"]:
        # The data page lists annotation.image_id as int but image.id as str; normalise both.
        key = a["image_id"] if a["image_id"] in by_image else str(a["image_id"])
        anns.setdefault(key, []).append(a)
    sets = []
    for img_id, img in by_image.items():
        stem = Path(img["file_name"]).stem
        sets.append(AnnotationSet(
            set_id=str(img_id), observation=stem, file_name=img["file_name"],
            height=img["height"], width=img["width"],
            polygons=[a["segmentation"][0] for a in anns.get(img_id, [])],
            categories=[a["category_id"] for a in anns.get(img_id, [])],
        ))
    return sets


def load_image(path: Path | str) -> np.ndarray:
    """Grayscale uint8 HxW (the JPEGs are single-channel; never treat them as RGB)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def image_path(observation: str) -> Path:
    for root in (TRAIN_IMAGES, TEST_IMAGES):
        p = root / f"{observation}.jpeg"
        if p.exists():
            return p
    raise FileNotFoundError(observation)


def _circle_lsq(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) least-squares circle fit."""
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    (cx, cy, c), *_ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
    return cx, cy, np.sqrt(c + cx**2 + cy**2)


# GONG H-alpha frames are pre-centred and scaled: ~80% of images fit to this disk within a few px.
NOMINAL_DISK = (1023.0, 1024.0, 903.5)


def fit_disk(img: np.ndarray, n_rays: int = 360, search: float = 0.06) -> tuple[float, float, float]:
    """Solar disk (cx, cy, r) in pixels.

    The JPEGs contain a grey off-limb halo (with prominences) around the disk whose outer edge is
    often a stronger edge than the limb itself, and the halo is not always concentric. So: start
    from the nominal GONG disk, find the steepest intensity drop along radial rays within
    +-`search` of the nominal radius (the limb), and fit a circle to those edge points with outlier
    rejection (prominences / dark filaments at the limb produce stray edges)."""
    cx, cy, r = NOMINAL_DISK
    smooth = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 2)
    theta = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    for _ in range(2):  # second pass re-samples rays from the refined centre
        radii = np.arange((1 - search) * r, (1 + search) * r)
        xs = cx + np.outer(np.cos(theta), radii)
        ys = cy + np.outer(np.sin(theta), radii)
        prof = cv2.remap(smooth, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        drop = np.argmin(np.diff(prof, axis=1), axis=1)
        ex, ey = xs[np.arange(n_rays), drop], ys[np.arange(n_rays), drop]
        keep = np.ones(n_rays, bool)
        for _ in range(3):
            cx, cy, r = _circle_lsq(ex[keep], ey[keep])
            resid = np.abs(np.hypot(ex - cx, ey - cy) - r)
            keep = resid < max(3 * 1.4826 * np.median(resid[keep]), 2.0)
    return float(cx), float(cy), float(r)


def time_group(observation: str, days: int = 27) -> int:
    """Bucket observations into blocks of `days` (default ~one solar rotation) so filaments seen
    on consecutive days, which are near-duplicates, land in the same fold."""
    t, _ = parse_observation(observation)
    return (t - datetime(2000, 1, 1)).days // days


def make_folds(observations: list[str], n_folds: int = 5, days: int = 27, seed: int = 0) -> dict[str, int]:
    """Assign each observation to a fold. Groups = time blocks; blocks are distributed greedily
    (largest first) to balance fold sizes. Returns {observation: fold}."""
    rng = np.random.default_rng(seed)
    groups: dict[int, list[str]] = {}
    for obs in sorted(set(observations)):
        groups.setdefault(time_group(obs, days), []).append(obs)
    keys = list(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))
    sizes = np.zeros(n_folds, dtype=int)
    fold_of: dict[str, int] = {}
    for k in keys:
        f = int(sizes.argmin())
        sizes[f] += len(groups[k])
        for obs in groups[k]:
            fold_of[obs] = f
    return fold_of
