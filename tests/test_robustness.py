"""Robustness stress-test: perturbation determinism/purity, sweep math and
delta bookkeeping, the measured degradation direction, and byte-identical CSV.

None of these tests need torch: the sweep runs on any dict of fitted detectors,
and the report used here is the torch-free PCA + local-statistics study.
"""

import numpy as np
import pytest

from qav.data import DataConfig, make_dataset
from qav.evaluate import average_precision, roc_auc, run_full_evaluation, tpr_at_fpr
from qav.robustness import (
    PERTURBATIONS,
    perturb,
    plain_language_read,
    robustness_csv,
    robustness_for_report,
    robustness_sweep,
    write_robustness_csv,
)

# Moderate torch-free config: big enough that the measured degradation
# direction is stable, small enough to stay fast.
CFG = DataConfig(n_train=150, n_test_clean=60, n_test_defective=60, seed=13)


@pytest.fixture(scope="module")
def report():
    return run_full_evaluation(CFG, include_autoencoder=False)


@pytest.mark.parametrize("kind", list(PERTURBATIONS))
def test_perturb_is_deterministic_pure_and_clipped(kind):
    imgs = make_dataset(DataConfig(n_train=4, n_test_clean=4, n_test_defective=3, seed=5)).test_images
    before = imgs.copy()
    severity = PERTURBATIONS[kind][-1]  # strongest level

    a = perturb(imgs, kind, severity, seed=123)
    b = perturb(imgs, kind, severity, seed=123)
    assert np.array_equal(a, b), "same seed + severity must be bit-identical"
    assert np.array_equal(imgs, before), "perturb must not mutate its input"
    assert a.shape == imgs.shape and a.dtype == np.float32
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0, "output must stay in [0, 1]"
    # The strongest severity actually changes the image.
    assert not np.array_equal(a, imgs)


def test_gaussian_noise_grows_with_severity():
    imgs = make_dataset(DataConfig(n_train=4, n_test_clean=6, n_test_defective=3, seed=7)).test_images
    mild = np.abs(perturb(imgs, "gaussian_noise", 0.02, seed=1) - imgs).mean()
    strong = np.abs(perturb(imgs, "gaussian_noise", 0.10, seed=1) - imgs).mean()
    assert strong > mild > 0.0


def test_perturb_rejects_unknown_kind():
    imgs = make_dataset(DataConfig(n_train=2, n_test_clean=2, n_test_defective=3, seed=1)).test_images
    with pytest.raises(ValueError):
        perturb(imgs, "jpeg", 0.5)


def test_baselines_reuse_report_scores_and_deltas_are_exact(report):
    robust = robustness_for_report(report)
    # Every retained detector is stress-tested; nothing is re-fit.
    assert set(robust.baselines) == set(report.detectors)
    n_points = len(report.methods) * sum(len(s) for s in PERTURBATIONS.values())
    assert len(robust.points) == n_points

    labels = report.dataset.test_labels
    for m in report.methods:
        base = robust.baselines[m.name]
        # Baseline metrics reuse the study's already-computed scores exactly.
        assert base.roc_auc == pytest.approx(roc_auc(labels, m.scores))
        assert base.pr_auc == pytest.approx(average_precision(labels, m.scores))
        assert base.tpr_at_5pct_fpr == pytest.approx(tpr_at_fpr(labels, m.scores, 0.05))

    # Deltas are exactly perturbed-minus-baseline, and a recomputed point matches.
    det = report.detectors["PCA reconstruction"]
    base = robust.baselines["PCA reconstruction"]
    pt = next(
        p for p in robust.points
        if p.method == "PCA reconstruction" and p.perturbation == "blur" and p.severity == 1.6
    )
    from qav.robustness import _point_seed  # deterministic per-point seed

    corrupted = perturb(report.dataset.test_images, "blur", 1.6, _point_seed(robust.seed, "blur", 2))
    recomputed = average_precision(labels, det.scores(corrupted))
    assert pt.pr_auc == pytest.approx(recomputed)
    assert pt.d_pr_auc == pytest.approx(pt.pr_auc - base.pr_auc)
    assert pt.d_roc_auc == pytest.approx(pt.roc_auc - base.roc_auc)


def test_measured_degradation_direction_is_honest(report):
    robust = robustness_for_report(report)

    def point(method, kind, sev):
        return next(
            p for p in robust.points
            if p.method == method and p.perturbation == kind and p.severity == sev
        )

    for name in robust.baselines:
        # Heavy defocus and heavy noise wash out small defects -> PR-AUC falls.
        assert point(name, "blur", 1.6).d_pr_auc < 0
        assert point(name, "gaussian_noise", 0.10).d_pr_auc < 0
        # A global brightness shift breaks the clean-trained detectors -> ROC-AUC falls.
        assert point(name, "brightness", 0.20).d_roc_auc < 0

    # PCA re-centers each image, so a mild contrast change barely moves it: a
    # genuine differential-robustness finding, asserted as a bounded delta.
    assert abs(point("PCA reconstruction", "contrast", 0.85).d_pr_auc) < 0.03

    # The worst-PR-drop helper points at a real (negative) degradation.
    worst = robust.worst_pr_drop("PCA reconstruction")
    assert worst.d_pr_auc < 0
    assert worst.d_pr_auc <= min(p.d_pr_auc for p in robust.points if p.method == "PCA reconstruction")

    read = plain_language_read(robust)
    assert "Robustness stress-test" in read
    assert "PCA reconstruction" in read


def test_robustness_csv_is_byte_identical_and_well_shaped(report, tmp_path):
    robust = robustness_for_report(report)
    text_a = robustness_csv(robust)
    text_b = robustness_csv(robustness_for_report(report))
    assert text_a == text_b, "CSV must be byte-identical across re-runs"

    lines = text_a.splitlines()
    assert lines[0] == (
        "method,perturbation,severity,roc_auc,pr_auc,tpr_at_5pct_fpr,"
        "d_roc_auc,d_pr_auc,d_tpr_at_5pct_fpr"
    )
    assert len(lines) == 1 + len(robust.baselines) + len(robust.points)
    # Baseline rows carry perturbation "none" and exactly-zero deltas.
    baseline_lines = [ln for ln in lines[1:] if ln.split(",")[1] == "none"]
    assert len(baseline_lines) == len(robust.baselines)
    for ln in baseline_lines:
        assert ln.endswith("0.000000,0.000000,0.000000")

    out = tmp_path / "robustness.csv"
    n = write_robustness_csv(robust, out)
    assert n > 0 and out.read_text(encoding="utf-8") == text_a
    # Re-writing yields the exact same bytes.
    write_robustness_csv(robust, out)
    assert out.read_text(encoding="utf-8") == text_a


def test_robustness_sweep_accepts_bare_detectors():
    # The sweep does not require a full report: a name->detector dict is enough,
    # and it computes baselines itself when none are supplied.
    dataset = make_dataset(CFG)
    from qav.baselines import PCAReconstruction

    pca = PCAReconstruction(n_components=24).fit(dataset.train)
    robust = robustness_sweep({"pca": pca}, dataset)
    assert set(robust.baselines) == {"pca"}
    assert len(robust.points) == sum(len(s) for s in PERTURBATIONS.values())
