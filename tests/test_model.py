"""Autoencoder tests (skipped cleanly when torch is not installed)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qav.data import DataConfig, make_dataset  # noqa: E402
from qav.model import ConvAutoencoderDetector, TrainConfig  # noqa: E402


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    labels = labels.astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def test_autoencoder_trains_separates_and_is_deterministic():
    data = make_dataset(DataConfig(n_train=48, n_test_clean=16, n_test_defective=15, seed=13))
    cfg = TrainConfig(epochs=3, batch_size=16, seed=3)

    det = ConvAutoencoderDetector(cfg).fit(data.train)
    assert 50_000 <= det.history["param_count"] <= 400_000, "should stay a small model"
    losses = det.history["epoch_losses"]
    assert len(losses) == 3 and all(np.isfinite(losses))
    assert losses[-1] <= losses[0] * 1.05, "training must not diverge"

    scores = det.scores(data.test_images)
    heatmaps = det.heatmaps(data.test_images[:4])
    assert heatmaps.shape == data.test_images[:4].shape and (heatmaps >= 0).all()
    auc = _rank_auc(data.test_labels, scores)
    assert auc > 0.55, f"AE should separate defective from clean even on a tiny config (auc={auc:.3f})"

    # Same seeds, same machine -> bit-identical scores on refit.
    refit_scores = ConvAutoencoderDetector(cfg).fit(data.train).scores(data.test_images)
    assert np.array_equal(scores, refit_scores)

    with pytest.raises(RuntimeError):
        ConvAutoencoderDetector(cfg).heatmaps(data.test_images[:1])
