"""Executive deliverables: a PDF report, an Excel workbook, and README figures.

Everything is rendered with matplotlib (Agg backend, no display needed) and
openpyxl via pandas. Colors follow one small validated categorical palette
(one fixed hue per method, never cycled); anomaly heatmaps use a single-hue
sequential ramp so magnitude reads as light -> dark.
"""

from __future__ import annotations

import datetime as _dt
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# Load pyarrow while the process is still quiet. pandas 3.0 would otherwise
# import it lazily at the first string-frame construction, which can hit a
# Windows access violation late in a busy process (torch thread pools +
# rendered figures) -- see tests/conftest.py for the full note.
try:
    import pyarrow  # noqa: F401
except ImportError:
    pass

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from qav.data import DEFECT_KINDS  # noqa: E402
from qav.evaluate import (  # noqa: E402
    LOCALIZATION_IOU_HIT,
    RECOMMENDATION_RULE,
    EvalReport,
    localization_ious,
)

# Fixed method -> hue assignment (categorical palette, validated for CVD
# separation; the magenta slot is low-contrast on white, so every figure that
# uses it also carries direct labels or a table).
METHOD_COLORS = {
    "Local statistics": "#2a78d6",
    "PCA reconstruction": "#008300",
    "Conv autoencoder": "#e87ba4",
}
FALLBACK_COLOR = "#eda100"
HEAT_CMAP = "Oranges"  # single-hue sequential: light = calm, dark = anomalous
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#d9d8d4"

DISCLAIMER = (
    "Synthetic-data study: every image in this report is procedurally generated "
    "(64x64 grayscale textures with injected defects). This is a deliberately "
    "small-scale method demonstration for a QA screening decision -- it is not a "
    "production vision system and reports no real production data."
)


def _method_color(name: str) -> str:
    return METHOD_COLORS.get(name, FALLBACK_COLOR)


def _style_axes(ax) -> None:
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def _cover_figure(report: EvalReport) -> plt.Figure:
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.text(0.5, 0.82, "Surface-Defect Screening Study", ha="center",
             fontsize=22, color=TEXT_PRIMARY, weight="bold")
    fig.text(0.5, 0.78, "Convolutional autoencoder vs classical anomaly detectors",
             ha="center", fontsize=12, color=TEXT_SECONDARY)
    fig.text(0.5, 0.75, f"Generated {_dt.date.today().isoformat()}  |  seed "
             f"{report.dataset.config.seed}  |  author: Dimitres Kisimov",
             ha="center", fontsize=9, color=TEXT_SECONDARY)

    rec = report.recommendation
    n_train = len(report.dataset.train)
    n_test = len(report.dataset.test_images)
    headline = (
        f"Recommendation: {rec.method}\n\n"
        f"Best ROC-AUC {rec.best_auc:.3f} ({rec.best_method}); recommended method "
        f"ROC-AUC {rec.chosen_auc:.3f}.\n"
        f"Decision rule (fixed before the study): simplest method within "
        f"{rec.margin:.2f} ROC-AUC of the best.\n\n"
        f"Data: {n_train} clean training images, {n_test} test images "
        f"({int(report.dataset.test_labels.sum())} defective across "
        f"{len(DEFECT_KINDS)} defect kinds)."
    )
    headline = "\n".join(
        textwrap.fill(line, width=72) for line in headline.splitlines()
    )
    fig.text(0.5, 0.58, headline, ha="center", va="center", fontsize=11,
             color=TEXT_PRIMARY, linespacing=1.7,
             bbox={"boxstyle": "round,pad=1.0", "facecolor": "#f4f3f1",
                   "edgecolor": TEXT_SECONDARY})
    fig.text(0.5, 0.30, textwrap.fill(DISCLAIMER, width=76), ha="center",
             va="center", fontsize=9, color=TEXT_SECONDARY, linespacing=1.6,
             bbox={"boxstyle": "round,pad=0.8", "facecolor": "white",
                   "edgecolor": "#b3261e"})
    return fig


def _gallery_figure(report: EvalReport) -> plt.Figure:
    """One example per class: image, ground truth, then one heatmap per method."""
    ds = report.dataset
    row_kinds = ["clean", *DEFECT_KINDS]
    example_idx = [ds.test_types.index(kind) for kind in row_kinds]
    n_cols = 2 + len(report.methods)
    fig, axes = plt.subplots(len(row_kinds), n_cols,
                             figsize=(2.1 * n_cols, 2.2 * len(row_kinds)))
    for r, (kind, idx) in enumerate(zip(row_kinds, example_idx, strict=True)):
        axes[r, 0].imshow(ds.test_images[idx], cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_ylabel(kind.replace("_", "-"), fontsize=10, color=TEXT_PRIMARY)
        axes[r, 1].imshow(ds.test_masks[idx], cmap="gray", vmin=0, vmax=1)
        for c, method in enumerate(report.methods, start=2):
            hm = method.heatmaps[idx]
            axes[r, c].imshow(hm, cmap=HEAT_CMAP, vmin=0.0, vmax=float(hm.max()) or 1.0)
            if ds.test_masks[idx].any():
                axes[r, c].contour(ds.test_masks[idx], levels=[0.5], colors=TEXT_PRIMARY,
                                   linewidths=0.7, linestyles="dashed")
        if r == 0:
            axes[r, 0].set_title("surface", fontsize=10, color=TEXT_PRIMARY)
            axes[r, 1].set_title("ground truth", fontsize=10, color=TEXT_PRIMARY)
            for c, method in enumerate(report.methods, start=2):
                axes[r, c].set_title(method.name, fontsize=10,
                                     color=_method_color(method.name), weight="bold")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Examples with per-method anomaly heatmaps "
                 "(each heatmap scaled to its own max; dashed = ground truth)",
                 fontsize=11, color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _curves_figure(report: EvalReport) -> plt.Figure:
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.6))
    for method in report.methods:
        color = _method_color(method.name)
        fpr, tpr = method.roc
        ax_roc.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{method.name} (AUC {method.roc_auc:.3f})")
        recall, precision = method.pr
        ax_pr.plot(recall, precision, color=color, linewidth=2,
                   label=f"{method.name} (AP {method.pr_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], color=GRID_COLOR, linewidth=1.5, linestyle="--")
    ax_roc.set_xlabel("False positive rate", fontsize=9, color=TEXT_PRIMARY)
    ax_roc.set_ylabel("True positive rate", fontsize=9, color=TEXT_PRIMARY)
    ax_roc.set_title("ROC -- defective vs clean (image level)", fontsize=11, color=TEXT_PRIMARY)
    ax_pr.set_xlabel("Recall", fontsize=9, color=TEXT_PRIMARY)
    ax_pr.set_ylabel("Precision", fontsize=9, color=TEXT_PRIMARY)
    ax_pr.set_title("Precision-Recall (image level)", fontsize=11, color=TEXT_PRIMARY)
    for ax in (ax_roc, ax_pr):
        _style_axes(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    return fig


def _per_type_bar_figure(report: EvalReport) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    kinds = list(DEFECT_KINDS)
    x = np.arange(len(kinds), dtype=float)
    width = 0.8 / len(report.methods)
    for i, method in enumerate(report.methods):
        vals = [method.per_type_auc[k] for k in kinds]
        pos = x + (i - (len(report.methods) - 1) / 2) * width
        ax.bar(pos, vals, width=width * 0.94, color=_method_color(method.name),
               edgecolor="white", linewidth=1, label=method.name)
        for p, v in zip(pos, vals, strict=True):
            ax.text(p, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5,
                    color=TEXT_SECONDARY)
    ax.axhline(0.5, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    ax.text(len(kinds) - 0.52, 0.512, "chance", fontsize=7.5, color=TEXT_SECONDARY)
    ax.set_xticks(x, [k.replace("_", "-") for k in kinds], fontsize=10, color=TEXT_PRIMARY)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("ROC-AUC vs clean", fontsize=9, color=TEXT_PRIMARY)
    ax.set_title("Which method catches which defect kind", fontsize=11, color=TEXT_PRIMARY)
    _style_axes(ax)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    return fig


def _table_figure(report: EvalReport) -> plt.Figure:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes((0.06, 0.55, 0.88, 0.30))
    ax.axis("off")
    columns = ["Method", "ROC-AUC", "PR-AUC", "TPR@5%FPR", "Mean IoU",
               f"Hit rate (IoU>={LOCALIZATION_IOU_HIT:.1f})"]
    rows = [
        [m.name, f"{m.roc_auc:.3f}", f"{m.pr_auc:.3f}", f"{m.tpr_at_5pct_fpr:.3f}",
         f"{m.mean_iou:.3f}", f"{m.hit_rate:.3f}"]
        for m in report.methods
    ]
    table = ax.table(cellText=rows, colLabels=columns, loc="upper center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.5)
    ax.set_title("Image-level detection and pixel-level localization",
                 fontsize=12, color=TEXT_PRIMARY, pad=18)

    ax2 = fig.add_axes((0.06, 0.28, 0.88, 0.22))
    ax2.axis("off")
    columns2 = ["Method"] + [f"{k.replace('_', '-')} AUC" for k in DEFECT_KINDS] + [
        f"{k.replace('_', '-')} IoU" for k in DEFECT_KINDS
    ]
    rows2 = [
        [m.name]
        + [f"{m.per_type_auc[k]:.3f}" for k in DEFECT_KINDS]
        + [f"{m.per_type_iou[k]:.3f}" for k in DEFECT_KINDS]
        for m in report.methods
    ]
    table2 = ax2.table(cellText=rows2, colLabels=columns2, loc="upper center",
                       cellLoc="center")
    table2.auto_set_font_size(False)
    table2.set_fontsize(7.6)
    table2.scale(1, 1.5)
    ax2.set_title("Per-defect-type breakdown", fontsize=12, color=TEXT_PRIMARY, pad=18)

    rec = report.recommendation
    box_text = "\n\n".join(
        [
            f"RECOMMENDATION: {rec.method}",
            textwrap.fill(rec.rationale, width=86),
            textwrap.fill(f"Rule (fixed before the study): {RECOMMENDATION_RULE}", width=86),
            textwrap.fill(
                "Chance-level localization for reference: random heatmaps reach "
                f"mean IoU {report.random_iou:.3f}.",
                width=86,
            ),
        ]
    )
    fig.text(0.5, 0.13, box_text, ha="center", va="center", fontsize=9,
             color=TEXT_PRIMARY, linespacing=1.6,
             bbox={"boxstyle": "round,pad=0.9", "facecolor": "#f4f3f1",
                   "edgecolor": TEXT_SECONDARY})
    return fig


# NOTE: every frame below is built from a dict of columns, never from a list
# of dicts/tuples -- the row-wise pandas constructor (nested_data_to_arrays)
# segfaulted intermittently on Python 3.14 + pandas 3.0 during development.
def _metrics_frame(report: EvalReport) -> pd.DataFrame:
    ms = report.methods
    return pd.DataFrame(
        {
            "method": [m.name for m in ms],
            "complexity_rank": [m.complexity_rank for m in ms],
            "roc_auc": [round(m.roc_auc, 4) for m in ms],
            "pr_auc": [round(m.pr_auc, 4) for m in ms],
            "tpr_at_5pct_fpr": [round(m.tpr_at_5pct_fpr, 4) for m in ms],
            "mean_iou": [round(m.mean_iou, 4) for m in ms],
            "hit_rate": [round(m.hit_rate, 4) for m in ms],
        }
    )


def _per_type_frame(report: EvalReport) -> pd.DataFrame:
    pairs = [(m, kind) for m in report.methods for kind in DEFECT_KINDS]
    return pd.DataFrame(
        {
            "method": [m.name for m, _ in pairs],
            "defect_type": [kind for _, kind in pairs],
            "roc_auc_vs_clean": [round(m.per_type_auc[kind], 4) for m, kind in pairs],
            "mean_iou": [round(m.per_type_iou[kind], 4) for m, kind in pairs],
        }
    )


def _per_image_frame(report: EvalReport) -> pd.DataFrame:
    """One row per test image: label, type, and every method's score (plus the
    localization IoU on defective images). Enough raw data for a reviewer to
    re-derive the ROC curves independently."""
    ds = report.dataset
    defective = ds.test_labels.astype(bool)
    frame = pd.DataFrame(
        {
            "image_index": np.arange(len(ds.test_images)),
            "test_type": ds.test_types,
            "label": ds.test_labels,
        }
    )
    for m in report.methods:
        key = m.name.lower().replace(" ", "_")
        frame[f"score_{key}"] = np.round(m.scores, 6)
        iou_col = np.full(len(ds.test_images), np.nan)
        iou_col[defective] = np.round(
            localization_ious(m.heatmaps[defective], ds.test_masks[defective]), 4
        )
        frame[f"iou_{key}"] = iou_col
    return frame


def _assumptions_frame(report: EvalReport) -> pd.DataFrame:
    cfg = report.dataset.config
    items = [
        ("Data provenance", "All images are synthetic (procedural textures + injected "
         "defects). No real production or camera data anywhere in the study."),
        ("Scale", "Deliberately small-scale method demonstration; results support a "
         "screening-method choice, not a production deployment."),
        ("Dataset", f"{cfg.n_train} clean train / {cfg.n_test_clean} clean + "
         f"{cfg.n_test_defective} defective test images, {cfg.image_size}x"
         f"{cfg.image_size} px, seed {cfg.seed} (fully deterministic)."),
        ("Defect kinds", "scratch (thin line), blob (Gaussian bump/pit), "
         "texture-break (rotated+shifted patch of the pattern)."),
        ("Image score", "Mean of the hottest 2% of heatmap pixels -- same rule for "
         "every method, fixed a priori (no per-method tuning)."),
        ("Localization", "Heatmap thresholded at its own 98th percentile; IoU vs "
         f"ground-truth mask; hit = IoU >= {LOCALIZATION_IOU_HIT:.2f}. Random "
         f"heatmaps reach mean IoU {report.random_iou:.4f}."),
        ("Recommendation rule", RECOMMENDATION_RULE),
        ("Autoencoder", f"~{report.ae_history.get('param_count', 'n/a')} parameters, "
         f"{len(report.ae_history.get('epoch_losses', []))} epochs, CPU, seeded "
         "(torch.manual_seed + deterministic algorithms requested with warn_only)."),
        ("Reproducibility", "Same machine + torch build reproduces bit-identical "
         "metrics; across builds/hardware small float differences are possible."),
    ]
    return pd.DataFrame(
        {
            "assumption": [name for name, _ in items],
            "detail": [detail for _, detail in items],
        }
    )


def build_deliverables(
    report: EvalReport,
    out_dir: str | Path = "deliverables",
    fig_dir: str | Path = "figures",
) -> dict[str, int]:
    """Write PDF + Excel + README figures; return {path: size_bytes}."""
    out_path = Path(out_dir)
    fig_path = Path(fig_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    pdf_file = out_path / "qa_defect_report.pdf"
    gallery = _gallery_figure(report)
    curves = _curves_figure(report)
    bars = _per_type_bar_figure(report)
    with PdfPages(pdf_file) as pdf:
        for fig in (_cover_figure(report), gallery, curves, bars, _table_figure(report)):
            pdf.savefig(fig)
    gallery.savefig(fig_path / "gallery.png", dpi=150, facecolor="white")
    curves.savefig(fig_path / "roc_pr.png", dpi=150, facecolor="white")
    bars.savefig(fig_path / "per_type_auc.png", dpi=150, facecolor="white")
    plt.close("all")

    xlsx_file = out_path / "qa_defect_metrics.xlsx"
    with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
        _metrics_frame(report).to_excel(writer, sheet_name="Metrics", index=False)
        _per_type_frame(report).to_excel(writer, sheet_name="PerDefectType", index=False)
        _assumptions_frame(report).to_excel(writer, sheet_name="Assumptions", index=False)
        _per_image_frame(report).to_excel(writer, sheet_name="PerImageScores", index=False)

    tracked = [pdf_file, xlsx_file,
               fig_path / "gallery.png", fig_path / "roc_pr.png", fig_path / "per_type_auc.png"]
    return {str(p): p.stat().st_size for p in tracked}
