"""Evaluation: ROC/PR from scratch, localization vs ground truth, and the
pre-stated recommendation rule.

No sklearn anywhere -- ROC-AUC uses the tie-aware rank (Mann-Whitney) formula
and PR-AUC is the standard step-wise average precision, both hand-checkable
and covered by tests against manually computed values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qav.baselines import (
    SCORE_TOP_QUANTILE,
    LocalStatsDetector,
    PCAReconstruction,
    heatmap_image_scores,
)
from qav.data import DEFECT_KINDS, DataConfig, SurfaceDataset, make_dataset

# Stated BEFORE looking at any results, and applied mechanically by
# ``pick_recommendation`` -- the write-up quotes this string verbatim.
RECOMMENDATION_RULE = (
    "Rank methods by implementation complexity (local statistics < PCA "
    "reconstruction < conv autoencoder). Recommend the SIMPLEST method whose "
    "overall image-level ROC-AUC is within 0.02 of the best method's. A more "
    "complex method is recommended only when it beats every simpler method by "
    "more than 0.02 ROC-AUC."
)
RECOMMENDATION_MARGIN = 0.02

# A defect counts as "localized" when predicted hot pixels overlap the ground
# truth with at least this IoU (fixed a priori, same for every method).
LOCALIZATION_IOU_HIT = 0.10


# --------------------------------------------------------------------------
# Image-level metrics, from scratch
# --------------------------------------------------------------------------


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware ROC-AUC via average ranks (Mann-Whitney U)."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc needs at least one positive and one negative")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average the ranks of tied scores.
    uniq, inverse = np.unique(scores, return_inverse=True)
    for u in range(len(uniq)):
        tied = inverse == u
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _blocks(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative TP/FP after each distinct threshold (descending scores)."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    # Last index of each tie block.
    distinct = np.nonzero(np.diff(sorted_scores))[0]
    block_ends = np.concatenate([distinct, [len(scores) - 1]])
    tp = np.cumsum(sorted_labels)[block_ends].astype(np.float64)
    fp = np.cumsum(~sorted_labels)[block_ends].astype(np.float64)
    return tp, fp, sorted_scores[block_ends]


def roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(fpr, tpr) points including the (0, 0) origin."""
    tp, fp, _ = _blocks(labels, scores)
    tpr = np.concatenate([[0.0], tp / tp[-1]])
    fpr = np.concatenate([[0.0], fp / fp[-1]])
    return fpr, tpr


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Step-wise PR-AUC: sum of precision * recall increments per threshold."""
    tp, fp, _ = _blocks(labels, scores)
    n_pos = tp[-1]
    if n_pos == 0:
        raise ValueError("average_precision needs at least one positive")
    precision = tp / (tp + fp)
    recall = tp / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def pr_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(recall, precision) points, starting at recall 0 / precision 1."""
    tp, fp, _ = _blocks(labels, scores)
    precision = np.concatenate([[1.0], tp / (tp + fp)])
    recall = np.concatenate([[0.0], tp / tp[-1]])
    return recall, precision


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.05) -> float:
    """Best achievable TPR while keeping FPR at or below ``max_fpr``."""
    fpr, tpr = roc_curve(labels, scores)
    ok = fpr <= max_fpr + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


# --------------------------------------------------------------------------
# Pixel-level localization
# --------------------------------------------------------------------------


def heatmap_to_mask(heatmap: np.ndarray, top_quantile: float = SCORE_TOP_QUANTILE) -> np.ndarray:
    """Threshold one heatmap at its own quantile (~2% hottest pixels)."""
    return heatmap >= np.quantile(heatmap, top_quantile)


def localization_ious(
    heatmaps: np.ndarray, masks: np.ndarray, top_quantile: float = SCORE_TOP_QUANTILE
) -> np.ndarray:
    """IoU between thresholded heatmap and ground-truth mask, per image."""
    ious = np.empty(len(heatmaps), dtype=np.float64)
    for i, (hm, gt) in enumerate(zip(heatmaps, masks, strict=True)):
        pred = heatmap_to_mask(hm, top_quantile)
        union = np.logical_or(pred, gt).sum()
        ious[i] = np.logical_and(pred, gt).sum() / union if union else 0.0
    return ious


def random_localization_iou(masks: np.ndarray, seed: int = 0) -> float:
    """Chance level: the same thresholding applied to pure-noise heatmaps."""
    rng = np.random.default_rng(seed)
    noise = rng.uniform(size=masks.shape)
    return float(localization_ious(noise, masks).mean())


# --------------------------------------------------------------------------
# Full study
# --------------------------------------------------------------------------


@dataclass
class MethodResult:
    name: str
    complexity_rank: int  # 1 = simplest; used by the recommendation rule
    roc_auc: float
    pr_auc: float
    tpr_at_5pct_fpr: float
    per_type_auc: dict[str, float]
    mean_iou: float
    hit_rate: float  # fraction of defective images with IoU >= LOCALIZATION_IOU_HIT
    per_type_iou: dict[str, float]
    scores: np.ndarray
    roc: tuple[np.ndarray, np.ndarray]
    pr: tuple[np.ndarray, np.ndarray]
    heatmaps: np.ndarray


@dataclass
class Recommendation:
    method: str
    rationale: str
    best_method: str
    best_auc: float
    chosen_auc: float
    margin: float = RECOMMENDATION_MARGIN


@dataclass
class EvalReport:
    dataset: SurfaceDataset
    methods: list[MethodResult]
    recommendation: Recommendation
    random_iou: float
    ae_history: dict = field(default_factory=dict)
    # name -> fitted detector, retained so downstream analyses (e.g. the
    # robustness stress-test) can re-score perturbed images without re-fitting.
    detectors: dict = field(default_factory=dict)


def evaluate_method(
    name: str,
    complexity_rank: int,
    heatmaps: np.ndarray,
    scores: np.ndarray,
    dataset: SurfaceDataset,
) -> MethodResult:
    labels = dataset.test_labels
    types = np.array(dataset.test_types)
    defective = labels.astype(bool)

    per_type_auc: dict[str, float] = {}
    for kind in DEFECT_KINDS:
        sel = (types == kind) | (types == "clean")
        per_type_auc[kind] = roc_auc(labels[sel], scores[sel])

    ious = localization_ious(heatmaps[defective], dataset.test_masks[defective])
    per_type_iou = {
        kind: float(ious[types[defective] == kind].mean()) for kind in DEFECT_KINDS
    }
    return MethodResult(
        name=name,
        complexity_rank=complexity_rank,
        roc_auc=roc_auc(labels, scores),
        pr_auc=average_precision(labels, scores),
        tpr_at_5pct_fpr=tpr_at_fpr(labels, scores, 0.05),
        per_type_auc=per_type_auc,
        mean_iou=float(ious.mean()),
        hit_rate=float((ious >= LOCALIZATION_IOU_HIT).mean()),
        per_type_iou=per_type_iou,
        scores=scores,
        roc=roc_curve(labels, scores),
        pr=pr_curve(labels, scores),
        heatmaps=heatmaps,
    )


def pick_recommendation(methods: list, margin: float = RECOMMENDATION_MARGIN) -> Recommendation:
    """Apply RECOMMENDATION_RULE mechanically to any objects exposing
    ``name``, ``complexity_rank`` and ``roc_auc``."""
    by_simplicity = sorted(methods, key=lambda m: m.complexity_rank)
    best = max(methods, key=lambda m: m.roc_auc)
    for m in by_simplicity:
        if m.roc_auc >= best.roc_auc - margin:
            chosen = m
            break
    else:  # pragma: no cover - the best method always qualifies
        chosen = best
    if chosen.name == best.name:
        rationale = (
            f"{chosen.name} has the best overall ROC-AUC ({chosen.roc_auc:.3f}) and no "
            f"simpler method comes within {margin:.2f} of it."
        )
    else:
        rationale = (
            f"{chosen.name} (ROC-AUC {chosen.roc_auc:.3f}) is within the pre-stated "
            f"{margin:.2f} margin of the best method, {best.name} "
            f"(ROC-AUC {best.roc_auc:.3f}), so the simpler method is recommended."
        )
    return Recommendation(
        method=chosen.name,
        rationale=rationale,
        best_method=best.name,
        best_auc=float(best.roc_auc),
        chosen_auc=float(chosen.roc_auc),
        margin=margin,
    )


def run_full_evaluation(
    data_config: DataConfig | None = None,
    train_config=None,
    include_autoencoder: bool = True,
) -> EvalReport:
    """Generate data, fit every method on clean training images, evaluate all."""
    dataset = make_dataset(data_config or DataConfig())

    results: list[MethodResult] = []
    detectors: dict = {}

    local = LocalStatsDetector().fit(dataset.train)
    hm = local.heatmaps(dataset.test_images)
    results.append(
        evaluate_method("Local statistics", 1, hm, heatmap_image_scores(hm), dataset)
    )
    detectors["Local statistics"] = local

    pca = PCAReconstruction(n_components=32).fit(dataset.train)
    hm = pca.heatmaps(dataset.test_images)
    results.append(
        evaluate_method("PCA reconstruction", 2, hm, heatmap_image_scores(hm), dataset)
    )
    detectors["PCA reconstruction"] = pca

    ae_history: dict = {}
    if include_autoencoder:
        from qav.model import ConvAutoencoderDetector, TrainConfig

        ae = ConvAutoencoderDetector(train_config or TrainConfig()).fit(dataset.train)
        hm = ae.heatmaps(dataset.test_images)
        results.append(
            evaluate_method("Conv autoencoder", 3, hm, heatmap_image_scores(hm), dataset)
        )
        ae_history = ae.history
        detectors["Conv autoencoder"] = ae

    defective = dataset.test_labels.astype(bool)
    return EvalReport(
        dataset=dataset,
        methods=results,
        recommendation=pick_recommendation(results),
        random_iou=random_localization_iou(dataset.test_masks[defective]),
        ae_history=ae_history,
        detectors=detectors,
    )
