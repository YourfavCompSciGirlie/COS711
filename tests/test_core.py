import numpy as np

from filaments import metrics, rle
from filaments.data import make_folds


def _box(y0, y1, x0, x1, size=64):
    m = np.zeros((size, size), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_rle_roundtrip():
    m = np.zeros((rle.IMAGE_SIZE, rle.IMAGE_SIZE), bool)
    m[100:300, 50:900] = True
    m[1500:1510, 2000:] = True
    assert (rle.decode(rle.encode(m)) == m).all()


def test_pq_perfect():
    g = [_box(0, 10, 0, 10), _box(20, 40, 20, 40)]
    s = metrics.image_stats(g, g)
    assert s.tp == 2 and s.fp == 0 and s.fn == 0 and np.isclose(s.pq, 1.0)


def test_pq_fragmentation_is_penalised():
    g = [_box(0, 10, 0, 40)]
    halves = [_box(0, 10, 0, 20), _box(0, 10, 20, 40)]  # each half has IoU 0.5 -> not a match
    s = metrics.image_stats(g, halves)
    assert s.tp == 0 and s.fn == 1 and s.fp == 2 and s.pq == 0
    assert s.one_to_many == 1


def test_pq_merging_is_penalised():
    g = [_box(0, 10, 0, 10), _box(0, 10, 30, 40)]
    merged = [_box(0, 10, 0, 40)]
    s = metrics.image_stats(g, merged)
    assert s.tp == 0 and s.many_to_one == 1


def test_pq_known_value():
    g = [_box(0, 10, 0, 10)]
    p = [_box(0, 10, 0, 8), _box(50, 60, 50, 60)]  # IoU 0.8 match + 1 FP
    s = metrics.image_stats(g, p)
    assert np.isclose(s.pq, 0.8 / (1 + 0.5))


def test_label_map_input():
    lab = np.zeros((64, 64), int)
    lab[0:10, 0:10] = 1
    lab[20:40, 20:40] = 7
    s = metrics.image_stats(lab, lab)
    assert s.tp == 2


def test_folds_keep_observations_and_nearby_days_together():
    obs = ["20150101120000Bh", "20150102120000Mh", "20150601120000Lh", "20160101120000Bh"]
    folds = make_folds(obs, n_folds=2, days=27)
    assert folds["20150101120000Bh"] == folds["20150102120000Mh"]
