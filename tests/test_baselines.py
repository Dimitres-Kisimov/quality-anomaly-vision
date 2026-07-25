"""Classical-baseline tests on tiny (fast) configurations."""

import numpy as np
import pytest

from qav.baselines import LocalStatsDetector, PCAReconstruction
from qav.data import DataConfig, make_dataset

CFG = DataConfig(n_train=80, n_test_clean=30, n_test_defective=30, seed=13)


@pytest.fixture(scope="module")
def dataset():
    return make_dataset(CFG)


def test_pca_error_higher_on_defective(dataset):
    pca = PCAReconstruction(n_components=24).fit(dataset.train)
    scores = pca.scores(dataset.test_images)
    labels = dataset.test_labels.astype(bool)
    assert scores[labels].mean() > scores[~labels].mean()
    # Heatmaps have image shape and are non-negative.
    hm = pca.heatmaps(dataset.test_images[:4])
    assert hm.shape == dataset.test_images[:4].shape and (hm >= 0).all()


def test_pca_reconstructs_clean_far_better_than_noise(dataset):
    pca = PCAReconstruction(n_components=24).fit(dataset.train)
    train_scores = pca.scores(dataset.train[:5])
    rng = np.random.default_rng(0)
    noise = rng.uniform(0.0, 1.0, size=(5, 64, 64)).astype(np.float32)
    assert pca.scores(noise).mean() > 3.0 * train_scores.mean()
    assert pca.explained_variance_ratio_ is not None
    assert 0.0 < float(pca.explained_variance_ratio_.sum()) <= 1.0 + 1e-9
    with pytest.raises(RuntimeError):
        PCAReconstruction().heatmaps(dataset.train[:1])


def test_local_stats_flags_injected_blobs(dataset):
    det = LocalStatsDetector().fit(dataset.train)
    types = np.array(dataset.test_types)
    scores = det.scores(dataset.test_images)
    assert scores[types == "blob"].mean() > scores[types == "clean"].mean()
    # Localization: the heatmap is hotter inside the ground-truth mask than
    # outside for the clear majority of blob images.
    heatmaps = det.heatmaps(dataset.test_images)
    blob_idx = [i for i, t in enumerate(dataset.test_types) if t == "blob"]
    hotter_inside = [
        heatmaps[i][dataset.test_masks[i]].mean() > heatmaps[i][~dataset.test_masks[i]].mean()
        for i in blob_idx
    ]
    assert np.mean(hotter_inside) >= 0.7
    with pytest.raises(RuntimeError):
        LocalStatsDetector().heatmaps(dataset.test_images[:1])
