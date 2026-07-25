"""Metric math against hand-computed values, localization vs chance, and the
recommendation rule."""

from types import SimpleNamespace

import numpy as np
import pytest

from qav.baselines import PCAReconstruction
from qav.data import DataConfig, make_dataset
from qav.evaluate import (
    average_precision,
    localization_ious,
    pick_recommendation,
    pr_curve,
    random_localization_iou,
    roc_auc,
    roc_curve,
    tpr_at_fpr,
)


def test_roc_and_pr_match_hand_computed_values():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    # Pairs: (0.1 vs 0.35, 0.8) both correct, (0.4 vs 0.8) correct,
    # (0.4 vs 0.35) wrong -> 3/4.
    assert roc_auc(labels, scores) == pytest.approx(0.75)
    # Descending: 0.8 (P=1, R=0.5), 0.4 (neg), 0.35 (P=2/3, R=1)
    # AP = 0.5 * 1 + 0.5 * 2/3 = 5/6.
    assert average_precision(labels, scores) == pytest.approx(5.0 / 6.0)

    # Perfect separation and fully tied scores.
    assert roc_auc(labels, np.array([1, 2, 3, 4])) == pytest.approx(1.0)
    assert average_precision(labels, np.array([1, 2, 3, 4])) == pytest.approx(1.0)
    assert roc_auc(labels, np.full(4, 0.5)) == pytest.approx(0.5)
    assert average_precision(labels, np.full(4, 0.5)) == pytest.approx(0.5)

    fpr, tpr = roc_curve(labels, scores)
    assert fpr[0] == tpr[0] == 0.0 and fpr[-1] == tpr[-1] == 1.0
    assert (np.diff(fpr) >= 0).all() and (np.diff(tpr) >= 0).all()
    recall, precision = pr_curve(labels, scores)
    assert recall[0] == 0.0 and precision[0] == 1.0 and recall[-1] == 1.0
    assert tpr_at_fpr(labels, np.array([1, 2, 3, 4]), 0.05) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        roc_auc(np.zeros(4), scores)


def test_heatmap_localization_beats_random():
    data = make_dataset(DataConfig(n_train=80, n_test_clean=30, n_test_defective=30, seed=13))
    defective = data.test_labels.astype(bool)
    pca = PCAReconstruction(n_components=24).fit(data.train)
    heatmaps = pca.heatmaps(data.test_images)
    ious = localization_ious(heatmaps[defective], data.test_masks[defective])
    chance = random_localization_iou(data.test_masks[defective])
    assert ious.mean() > 3.0 * chance, (
        f"PCA localization ({ious.mean():.4f}) should clearly beat random ({chance:.4f})"
    )
    assert (ious >= 0.1).mean() >= 0.3


def test_recommendation_rule_prefers_simplest_within_margin():
    def fake(name, rank, auc):
        return SimpleNamespace(name=name, complexity_rank=rank, roc_auc=auc)

    # AE best but PCA within the 0.02 margin -> the simpler PCA wins.
    rec = pick_recommendation(
        [fake("Local statistics", 1, 0.80), fake("PCA", 2, 0.94), fake("AE", 3, 0.95)]
    )
    assert rec.method == "PCA" and rec.best_method == "AE"

    # AE beats every simpler method by more than the margin -> AE wins.
    rec = pick_recommendation(
        [fake("Local statistics", 1, 0.89), fake("PCA", 2, 0.90), fake("AE", 3, 0.95)]
    )
    assert rec.method == "AE"

    # The simplest method is itself the best -> chosen, and says so.
    rec = pick_recommendation(
        [fake("Local statistics", 1, 0.96), fake("PCA", 2, 0.95), fake("AE", 3, 0.90)]
    )
    assert rec.method == "Local statistics" and rec.chosen_auc == pytest.approx(0.96)
    assert "best overall" in rec.rationale
