"""Data generator tests: determinism, shapes, and mask consistency."""

import numpy as np
import pytest

from qav.data import (
    DEFECT_KINDS,
    MASK_DELTA_THRESHOLD,
    DataConfig,
    _base_texture,
    inject_defect,
    make_dataset,
)

TINY = DataConfig(n_train=8, n_test_clean=6, n_test_defective=9, seed=11)


def test_dataset_is_deterministic():
    d1 = make_dataset(TINY)
    d2 = make_dataset(TINY)
    assert np.array_equal(d1.train, d2.train)
    assert np.array_equal(d1.test_images, d2.test_images)
    assert np.array_equal(d1.test_masks, d2.test_masks)
    assert d1.test_types == d2.test_types
    # A different seed must actually change the data.
    d3 = make_dataset(DataConfig(n_train=8, n_test_clean=6, n_test_defective=9, seed=12))
    assert not np.array_equal(d1.train, d3.train)


def test_dataset_shapes_labels_types():
    d = make_dataset(TINY)
    s = TINY.image_size
    assert d.train.shape == (8, s, s) and d.train.dtype == np.float32
    assert d.test_images.shape == (15, s, s)
    assert d.test_masks.shape == (15, s, s) and d.test_masks.dtype == bool
    assert int(d.test_labels.sum()) == 9 and len(d.test_types) == 15
    assert d.test_types.count("clean") == 6
    for kind in DEFECT_KINDS:
        assert d.test_types.count(kind) == 3
    assert float(d.train.min()) >= 0.0 and float(d.train.max()) <= 1.0


@pytest.mark.parametrize("kind", DEFECT_KINDS)
def test_inject_defect_changes_masked_pixels_only(kind):
    rng = np.random.default_rng(21)
    clean = _base_texture(rng, 64)
    before = clean.copy()
    defective, mask = inject_defect(clean, kind, rng)
    assert np.array_equal(clean, before), "injection must not mutate its input"
    assert mask.dtype == bool and mask.any(), "ground-truth mask must be non-empty"
    delta = np.abs(defective.astype(np.float64) - clean.astype(np.float64))
    assert delta[mask].mean() > MASK_DELTA_THRESHOLD
    # Outside the mask the image is (by construction) within the threshold.
    assert delta[~mask].max() <= MASK_DELTA_THRESHOLD + 1e-6
    assert float(defective.min()) >= 0.0 and float(defective.max()) <= 1.0


def test_dataset_masks_consistent_with_labels():
    d = make_dataset(TINY)
    for i, label in enumerate(d.test_labels):
        if label == 0:
            assert not d.test_masks[i].any()
            assert d.test_types[i] == "clean"
        else:
            assert d.test_masks[i].sum() >= 10, "defect masks should cover a visible region"
            assert d.test_types[i] in DEFECT_KINDS


def test_inject_defect_rejects_unknown_kind():
    rng = np.random.default_rng(0)
    clean = _base_texture(rng, 64)
    with pytest.raises(ValueError):
        inject_defect(clean, "dent", rng)
