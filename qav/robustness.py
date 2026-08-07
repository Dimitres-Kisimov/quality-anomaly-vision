"""Robustness stress-test: how much do the detectors degrade under
camera-realistic image corruptions they never saw during fitting?

The README states plainly that this study omits "lighting, perspective or focus
variation -- the classic failure modes of real vision systems". This module
measures exactly that omission instead of hand-waving past it. Each detector is
fit **once** on the clean, undistorted training images (as before); then the
**test** images are corrupted with four physically-motivated perturbations and
**re-scored with the already-fitted detector** -- no re-training, no re-fitting.
That mirrors deployment: a model tuned on today's clean camera feed meets a
line whose lighting drifts, whose focus slips, whose sensor adds noise.

The four perturbations (each swept over three increasing severities):

- ``gaussian_noise``  additive sensor noise, std in image units ([0, 1]).
- ``brightness``      a global illumination shift added to every pixel.
- ``blur``            Gaussian defocus (``scipy.ndimage.gaussian_filter``).
- ``contrast``        multiplicative contrast reduction about each image's mean
                      (gain < 1 = washed-out exposure).

For every (method, perturbation, severity) we report ROC-AUC, PR-AUC and
TPR at 5% FPR -- computed by the *same* from-scratch estimators used everywhere
else in the project (:mod:`qav.evaluate`) -- plus the **delta** against that
method's clean-test baseline. PR-AUC is the headline metric (defects are the
rare positive class), consistent with the rest of the study.

Honesty scope, stated plainly:

- Every number here is **measured** on the synthetic test set; nothing is
  assumed. The perturbations are deterministic (seeded once, reused), so the
  CSV this module emits is byte-identical across re-runs.
- The corruptions are *synthetic stand-ins* for real camera degradation, not a
  substitute for a pilot on real hardware. A negative delta says "this method
  loses this much separability under this much of this corruption on this data"
  -- it is a relative fragility signal, not a production guarantee.

Pure numpy + scipy + the standard library: no wall-clock, no plotting, and it
re-uses fitted detectors rather than training anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from qav.evaluate import average_precision, roc_auc, tpr_at_fpr

# Perturbation -> increasing severities. Chosen so the mildest level is a
# barely-visible corruption and the strongest is a clearly degraded but still
# plausible camera fault. Fixed a priori, identical for every method.
PERTURBATIONS: dict[str, tuple[float, ...]] = {
    "gaussian_noise": (0.02, 0.05, 0.10),  # additive sensor-noise std (image units)
    "brightness": (0.05, 0.10, 0.20),  # global illumination shift (added to every pixel)
    "blur": (0.6, 1.0, 1.6),  # defocus: Gaussian filter sigma in pixels
    "contrast": (0.85, 0.70, 0.55),  # multiplicative contrast gain about the mean (<1 = washout)
}

# One base seed for the (only) stochastic perturbation, Gaussian noise. Per
# (perturbation, severity) we derive a distinct fixed seed so the whole sweep is
# reproducible and byte-identical, yet different severities use different noise.
ROBUSTNESS_SEED = 4242
TPR_MAX_FPR = 0.05  # the fixed operating point shared with the rest of the study


def _point_seed(base_seed: int, kind: str, severity_index: int) -> int:
    """Deterministic per-(kind, severity) seed. No use of the builtin ``hash``
    (which is salted per process) -- the perturbation order fixes the index."""
    kind_index = list(PERTURBATIONS).index(kind)
    return base_seed + 97 * kind_index + severity_index


# --------------------------------------------------------------------------
# Perturbations (pure: never mutate the input; always clip back into [0, 1])
# --------------------------------------------------------------------------


def _add_gaussian_noise(images: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=images.shape)
    return np.clip(images.astype(np.float64) + noise, 0.0, 1.0).astype(np.float32)


def _shift_brightness(images: np.ndarray, delta: float, seed: int) -> np.ndarray:
    return np.clip(images.astype(np.float64) + delta, 0.0, 1.0).astype(np.float32)


def _blur(images: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    out = np.stack([ndimage.gaussian_filter(im.astype(np.float64), sigma) for im in images])
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _reduce_contrast(images: np.ndarray, gain: float, seed: int) -> np.ndarray:
    x = images.astype(np.float64)
    mean = x.mean(axis=(1, 2), keepdims=True)
    return np.clip(mean + gain * (x - mean), 0.0, 1.0).astype(np.float32)


_PERTURBERS = {
    "gaussian_noise": _add_gaussian_noise,
    "brightness": _shift_brightness,
    "blur": _blur,
    "contrast": _reduce_contrast,
}


def perturb(images: np.ndarray, kind: str, severity: float, seed: int = ROBUSTNESS_SEED) -> np.ndarray:
    """Apply one perturbation to a stack of images, deterministically.

    ``images`` is left untouched; a new ``float32`` array in ``[0, 1]`` with the
    same shape is returned. Only ``gaussian_noise`` consumes ``seed``; the other
    three are fully determined by ``severity``.
    """
    if kind not in _PERTURBERS:
        raise ValueError(f"unknown perturbation {kind!r}; expected one of {tuple(PERTURBATIONS)}")
    return _PERTURBERS[kind](np.asarray(images), float(severity), seed)


# --------------------------------------------------------------------------
# Sweep + report
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodBaseline:
    """A method's metrics on the clean (unperturbed) test set."""

    method: str
    roc_auc: float
    pr_auc: float
    tpr_at_5pct_fpr: float


@dataclass(frozen=True)
class RobustnessPoint:
    """One (method, perturbation, severity) result and its deltas vs baseline."""

    method: str
    perturbation: str
    severity: float
    roc_auc: float
    pr_auc: float
    tpr_at_5pct_fpr: float
    d_roc_auc: float  # perturbed minus baseline (negative = degraded)
    d_pr_auc: float
    d_tpr_at_5pct_fpr: float


@dataclass(frozen=True)
class RobustnessReport:
    baselines: dict[str, MethodBaseline]
    points: list[RobustnessPoint]
    perturbations: dict[str, tuple[float, ...]]
    seed: int

    def worst_pr_drop(self, method: str) -> RobustnessPoint:
        """The point with the largest PR-AUC drop for ``method`` (ties -> the
        more severe corruption, then a stable perturbation order)."""
        pts = [p for p in self.points if p.method == method]
        order = list(self.perturbations)
        return min(pts, key=lambda p: (round(p.d_pr_auc, 6), -p.severity, order.index(p.perturbation)))


def _metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    return (
        roc_auc(labels, scores),
        average_precision(labels, scores),
        tpr_at_fpr(labels, scores, TPR_MAX_FPR),
    )


def robustness_sweep(
    detectors: dict,
    dataset,
    perturbations: dict[str, tuple[float, ...]] | None = None,
    seed: int = ROBUSTNESS_SEED,
    baseline_scores: dict[str, np.ndarray] | None = None,
) -> RobustnessReport:
    """Stress-test every fitted detector across the perturbation grid.

    ``detectors`` maps a method name to an object exposing ``scores(images)``
    (already fit on the clean training set). ``baseline_scores`` may supply each
    method's clean-test scores to avoid recomputing them; anything missing is
    scored on the fly. Detectors are only ever *queried*, never re-fit.
    """
    perturbations = perturbations or PERTURBATIONS
    baseline_scores = baseline_scores or {}
    labels = np.asarray(dataset.test_labels)
    images = dataset.test_images

    baselines: dict[str, MethodBaseline] = {}
    points: list[RobustnessPoint] = []
    for name, detector in detectors.items():
        base = baseline_scores.get(name)
        if base is None:
            base = detector.scores(images)
        b_roc, b_pr, b_tpr = _metrics(labels, np.asarray(base))
        baselines[name] = MethodBaseline(name, b_roc, b_pr, b_tpr)

        for kind, severities in perturbations.items():
            for sev_index, severity in enumerate(severities):
                corrupted = perturb(images, kind, severity, _point_seed(seed, kind, sev_index))
                roc, pr, tpr = _metrics(labels, detector.scores(corrupted))
                points.append(
                    RobustnessPoint(
                        method=name,
                        perturbation=kind,
                        severity=float(severity),
                        roc_auc=roc,
                        pr_auc=pr,
                        tpr_at_5pct_fpr=tpr,
                        d_roc_auc=roc - b_roc,
                        d_pr_auc=pr - b_pr,
                        d_tpr_at_5pct_fpr=tpr - b_tpr,
                    )
                )
    return RobustnessReport(
        baselines=baselines, points=points, perturbations=dict(perturbations), seed=seed
    )


def robustness_for_report(report, seed: int = ROBUSTNESS_SEED) -> RobustnessReport:
    """Run the sweep on every method in ``report`` whose fitted detector is
    retained (``report.detectors``), reusing the already-computed clean-test
    scores as baselines so nothing is scored twice or re-fit."""
    detectors = getattr(report, "detectors", {}) or {}
    baseline_scores = {m.name: m.scores for m in report.methods if m.name in detectors}
    return robustness_sweep(detectors, report.dataset, seed=seed, baseline_scores=baseline_scores)


def plain_language_read(report: RobustnessReport) -> str:
    """A short read a QA/vision engineer can act on -- all numbers from the sweep."""
    lines = [
        "Robustness stress-test (each detector fit on clean images, then re-scored on "
        "corrupted test images -- no re-fitting):",
    ]
    for name, base in report.baselines.items():
        worst = report.worst_pr_drop(name)
        lines.append(
            f"  - {name}: clean PR-AUC {base.pr_auc:.3f}; worst corruption is "
            f"{worst.perturbation} @ {worst.severity:g} -> PR-AUC {worst.pr_auc:.3f} "
            f"({worst.d_pr_auc:+.3f}), ROC-AUC {worst.d_roc_auc:+.3f}."
        )
    lines.append(
        "Reading: a large negative delta flags a corruption the method cannot absorb; a "
        "near-zero delta means that failure mode is (on this synthetic data) tolerable. These "
        "are relative fragility signals on synthetic corruptions, not production guarantees."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Byte-identical CSV
# --------------------------------------------------------------------------

_CSV_HEADER = (
    "method,perturbation,severity,roc_auc,pr_auc,tpr_at_5pct_fpr,"
    "d_roc_auc,d_pr_auc,d_tpr_at_5pct_fpr"
)


def robustness_csv(report: RobustnessReport) -> str:
    """The sweep as CSV text (LF endings). Baseline rows first (perturbation
    ``none``, severity 0, zero deltas), then one row per operating point."""
    rows = [_CSV_HEADER]

    def row(method, pert, sev, roc, pr, tpr, d_roc, d_pr, d_tpr) -> str:
        return ",".join(
            [
                method,
                pert,
                f"{sev:.4f}",
                f"{roc:.6f}",
                f"{pr:.6f}",
                f"{tpr:.6f}",
                f"{d_roc:.6f}",
                f"{d_pr:.6f}",
                f"{d_tpr:.6f}",
            ]
        )

    for name, base in report.baselines.items():
        rows.append(
            row(name, "none", 0.0, base.roc_auc, base.pr_auc, base.tpr_at_5pct_fpr, 0.0, 0.0, 0.0)
        )
    for p in report.points:
        rows.append(
            row(
                p.method,
                p.perturbation,
                p.severity,
                p.roc_auc,
                p.pr_auc,
                p.tpr_at_5pct_fpr,
                p.d_roc_auc,
                p.d_pr_auc,
                p.d_tpr_at_5pct_fpr,
            )
        )
    return "\n".join(rows) + "\n"


def write_robustness_csv(report: RobustnessReport, path: str | Path) -> int:
    text = robustness_csv(report)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return len(text.encode("utf-8"))


def _fmt(x: float) -> str:  # small helper reserved for callers that want a display string
    return "" if math.isnan(x) else f"{x:.3f}"
