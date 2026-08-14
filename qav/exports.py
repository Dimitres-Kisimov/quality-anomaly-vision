"""Executive deliverables: a PDF report, an Excel workbook, and README figures.

Everything is rendered with matplotlib (Agg backend, no display needed) and
openpyxl via pandas. Every color comes from :mod:`qav.palette` -- one
production line seen under inspection lights: machined-metal neutrals for the
page, material tones (steel / patina / copper) for identity, inspection-lamp
amber for caution, signal red for the thing you act on. One fixed hue per
method, never cycled; anomaly heatmaps use the palette's single sequential
ramp so magnitude reads as light -> dark.
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
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

from qav.data import DEFECT_KINDS  # noqa: E402
from qav.economics import (  # noqa: E402
    CostSweep,
    economics_for_report,
    plain_language_read,
    write_cost_curve_csv,
    write_cost_curve_svg,
)
from qav.evaluate import (  # noqa: E402
    LOCALIZATION_IOU_HIT,
    RECOMMENDATION_RULE,
    EvalReport,
    localization_ious,
)
from qav.palette import (  # noqa: E402
    COPPER,
    GRADE_COLORS,
    GRID,
    HEAT_RAMP,
    INK,
    INK_MUTED,
    LAMP,
    MASK_COLORS,
    PANEL,
    PATINA,
    SIGNAL,
    STEEL,
    SURFACE,
)
from qav.recalibration import (  # noqa: E402
    POLICY_LABELS,
    POLICY_ORDER,
    RecalConfig,
    RecalStudy,
    recalibration_for_report,
    write_recalibration_csv,
    write_recalibration_svg,
)
from qav.robustness import (  # noqa: E402
    RobustnessReport,
    robustness_for_report,
    write_robustness_csv,
)
from qav.severity import (  # noqa: E402
    GRADES,
    SeverityStudy,
    severity_for_report,
    write_severity_csv,
    write_severity_svg,
)
from qav.severity import (  # noqa: E402
    plain_language_read as severity_read,
)
from qav.spc import (  # noqa: E402
    RULE_LABELS,
    SPCConfig,
    SPCStudy,
    chart_rows,
    spc_for_report,
    write_spc_csv,
    write_spc_svg,
)
from qav.spc import (  # noqa: E402
    plain_language_read as spc_read,
)

# Fixed method -> hue assignment: the three material tones of the palette,
# never cycled (validated as a categorical trio -- every check PASS, worst
# all-pairs CVD delta-E 8.2).
METHOD_COLORS = {
    "Local statistics": STEEL,
    "PCA reconstruction": PATINA,
    "Conv autoencoder": COPPER,
}
FALLBACK_COLOR = LAMP
# Single sequential ramp for anomaly heatmaps: lamp-lit metal -> hot mark.
HEAT_CMAP = LinearSegmentedColormap.from_list("qav_inspection", list(HEAT_RAMP))
# Ground truth: material where the part is clean, signal red where it is not.
MASK_CMAP = ListedColormap(list(MASK_COLORS))
TEXT_PRIMARY = INK
TEXT_SECONDARY = INK_MUTED
GRID_COLOR = GRID

DISCLAIMER = (
    "Synthetic-data study: every image in this report is procedurally generated "
    "(64x64 grayscale textures with injected defects). This is a deliberately "
    "small-scale method demonstration for a QA screening decision -- it is not a "
    "production vision system and reports no real production data."
)


def _method_color(name: str) -> str:
    return METHOD_COLORS.get(name, FALLBACK_COLOR)


def _page(figsize=(8.27, 11.69)) -> plt.Figure:
    """A plate on the palette's bench surface (A4 portrait by default)."""
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    return fig


def _panels(nrows: int, ncols: int, figsize):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def _legend(ax, **kwargs):
    """Legends sit on the bench surface, hairline-ringed, never pure white."""
    kwargs.setdefault("framealpha", 0.92)
    return ax.legend(facecolor=SURFACE, edgecolor=GRID_COLOR, **kwargs)


def _caption_box(facecolor: str = PANEL, edgecolor: str = INK_MUTED, pad: float = 0.9) -> dict:
    return {"boxstyle": f"round,pad={pad}", "facecolor": facecolor, "edgecolor": edgecolor}


def _style_table(table) -> None:
    """Tables wear the plate too: header row on the panel tone, body on the
    bench surface, hairline rules -- never matplotlib's default white."""
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        cell.set_linewidth(0.8)
        cell.set_facecolor(PANEL if row == 0 else SURFACE)
        cell.get_text().set_color(TEXT_PRIMARY)


def _cover_figure(report: EvalReport) -> plt.Figure:
    fig = _page()  # A4 portrait
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
             bbox=_caption_box(pad=1.0))
    fig.text(0.5, 0.30, textwrap.fill(DISCLAIMER, width=76), ha="center",
             va="center", fontsize=9, color=TEXT_SECONDARY, linespacing=1.6,
             bbox=_caption_box(facecolor=SURFACE, edgecolor=SIGNAL, pad=0.8))
    return fig


def _gallery_figure(report: EvalReport) -> plt.Figure:
    """One example per class: image, ground truth, then one heatmap per method."""
    ds = report.dataset
    row_kinds = ["clean", *DEFECT_KINDS]
    example_idx = [ds.test_types.index(kind) for kind in row_kinds]
    n_cols = 2 + len(report.methods)
    fig, axes = _panels(len(row_kinds), n_cols,
                        figsize=(2.1 * n_cols, 2.2 * len(row_kinds)))
    for r, (kind, idx) in enumerate(zip(row_kinds, example_idx, strict=True)):
        axes[r, 0].imshow(ds.test_images[idx], cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_ylabel(kind.replace("_", "-"), fontsize=10, color=TEXT_PRIMARY)
        axes[r, 1].imshow(ds.test_masks[idx], cmap=MASK_CMAP, vmin=0, vmax=1)
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
    fig, (ax_roc, ax_pr) = _panels(1, 2, figsize=(11, 4.6))
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
        _legend(ax, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def _per_type_bar_figure(report: EvalReport) -> plt.Figure:
    fig, ax = _panels(1, 1, figsize=(8, 4.4))
    kinds = list(DEFECT_KINDS)
    x = np.arange(len(kinds), dtype=float)
    width = 0.8 / len(report.methods)
    for i, method in enumerate(report.methods):
        vals = [method.per_type_auc[k] for k in kinds]
        pos = x + (i - (len(report.methods) - 1) / 2) * width
        ax.bar(pos, vals, width=width * 0.94, color=_method_color(method.name),
               edgecolor=SURFACE, linewidth=1, label=method.name)
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
    _legend(ax, fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def _table_figure(report: EvalReport) -> plt.Figure:
    fig = _page()
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
    _style_table(table)
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
    _style_table(table2)
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
             bbox=_caption_box())
    return fig


def _economics_figure(sweep: CostSweep) -> plt.Figure:
    """Cost curve for the PDF: total / escape / scrap vs reject rate, optimum marked."""
    fig = _page()
    ax = fig.add_axes((0.12, 0.55, 0.80, 0.34))
    pts = sweep.points
    x = [p.reject_rate for p in pts]
    ax.plot(x, [p.escape_cost for p in pts], color=LAMP,
            linewidth=1.6, linestyle="--", label="escaped-defect cost")
    ax.plot(x, [p.false_reject_cost for p in pts], color=STEEL,
            linewidth=1.6, linestyle=":", label="false-reject / scrap cost")
    ax.plot(x, [p.expected_cost for p in pts], color=SIGNAL, linewidth=2.4,
            label="total expected cost")
    b = sweep.best
    ax.axvline(b.reject_rate, color=SIGNAL, linewidth=1, linestyle="--")
    ax.plot([b.reject_rate], [b.expected_cost], "o", color=SIGNAL, markersize=7)
    ax.annotate(f"recommended\n{b.reject_rate:.1%} rejected\n{b.expected_cost:,.0f} EUR",
                xy=(b.reject_rate, b.expected_cost), xytext=(12, 14),
                textcoords="offset points", fontsize=8.5, color=SIGNAL, weight="bold")
    m = sweep.model
    ax.set_xlabel("Reject rate (share of parts pulled for manual inspection)",
                  fontsize=9, color=TEXT_PRIMARY)
    ax.set_ylabel(f"Expected cost (EUR per {m.units_basis:,} parts)", fontsize=9,
                  color=TEXT_PRIMARY)
    ax.set_title(f"Inspection-threshold economics: scrap vs escape ({sweep.method})",
                 fontsize=12, color=TEXT_PRIMARY)
    _style_axes(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    _legend(ax, fontsize=8, loc="upper center")

    caption = "\n\n".join(
        [
            plain_language_read(sweep),
            "Cost rates and defect prevalence above are ILLUSTRATIVE, labelled constants "
            "(see docs/BUSINESS_CASE.md), not guarantees; the true-/false-positive rates at "
            "every threshold are measured from the synthetic test set.",
        ]
    )
    caption = "\n".join(textwrap.fill(line, width=92) for line in caption.splitlines())
    fig.text(0.5, 0.27, caption, ha="center", va="center", fontsize=8.4,
             color=TEXT_PRIMARY, linespacing=1.5,
             bbox=_caption_box())
    return fig


def _severity_figure(sev: SeverityStudy) -> plt.Figure:
    """Severity page: what graded escapes cost, and which grades escape.

    Mirrors ``figures/severity.svg`` -- panel A puts the graded cost curve
    against the flat-rate one on the same threshold grid (so the optima are
    comparable at a glance), panel B shows the share of each grade the screen
    catches at every reject rate.
    """
    fig = _page()
    ax_cost = fig.add_axes((0.12, 0.66, 0.80, 0.24))
    ax_catch = fig.add_axes((0.12, 0.42, 0.80, 0.16))
    m = sev.model
    x = [p.reject_rate for p in sev.points]
    ax_cost.plot([p.reject_rate for p in sev.flat.points],
                 [p.expected_cost for p in sev.flat.points], color=STEEL,
                 linewidth=1.6, linestyle="--",
                 label=f"one flat escape rate [A4]: {m.cost_escaped:.0f} EUR each")
    ax_cost.plot(x, [p.expected_cost for p in sev.points], color=GRADE_COLORS["critical"],
                 linewidth=2.4,
                 label="graded escapes [A11]: "
                       + " / ".join(f"{p.escape_cost:.0f}" for p in sev.profiles) + " EUR")
    ax_cost.axvline(sev.best.reject_rate, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    ax_cost.plot([sev.best.reject_rate], [sev.best.expected_cost], "o",
                 color=GRADE_COLORS["critical"], markersize=7)
    ax_cost.plot([sev.flat.best.reject_rate], [sev.flat.best.expected_cost], "o",
                 color=SURFACE, markeredgecolor=STEEL, markeredgewidth=1.8, markersize=6)
    same = "the same point" if not sev.moved else "different points"
    note = (f"both models recommend {same}: {sev.best.reject_rate:.2%} of parts pulled\n"
            f"graded bill {sev.at_flat_best.expected_cost:,.0f} EUR vs flat "
            f"{sev.flat.best.expected_cost:,.0f} EUR per {m.units_basis:,} parts")
    if sev.breakeven is not None:
        note += (f"\nthe point moves only once a critical escape is worth "
                 f"{sev.breakeven.critical_cost:,.0f} EUR ({sev.breakeven.multiple:.1f}x [A4])")
    ax_cost.text(0.03, 0.96, note, transform=ax_cost.transAxes, ha="left", va="top",
                 fontsize=8, color=TEXT_PRIMARY, linespacing=1.4)
    ax_cost.set_ylabel(f"expected cost (EUR / {m.units_basis:,} parts)", fontsize=8.5,
                       color=TEXT_PRIMARY)
    ax_cost.set_title("A - the bill: graded escapes vs one flat escape rate", fontsize=10,
                      color=TEXT_PRIMARY, loc="left")
    _legend(ax_cost, fontsize=7.5, loc="lower right")

    dashes = {"minor": ":", "major": "--", "critical": "-"}
    for p in sev.profiles:
        ax_catch.plot(x, [q.tpr_by_grade[p.grade] for q in sev.points],
                      color=GRADE_COLORS[p.grade], linewidth=1.8, linestyle=dashes[p.grade],
                      label=f"{p.grade} ({p.n} parts, AUC {p.roc_auc_vs_clean:.3f})")
    ax_catch.axvline(sev.best.reject_rate, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    ax_catch.set_ylabel("share of that grade caught", fontsize=8.5, color=TEXT_PRIMARY)
    ax_catch.set_xlabel("reject rate (share of parts pulled for manual inspection)",
                        fontsize=9, color=TEXT_PRIMARY)
    ax_catch.set_title("B - who actually escapes", fontsize=10, color=TEXT_PRIMARY, loc="left")
    ax_catch.yaxis.set_major_formatter(lambda v, _pos: f"{v:.0%}")
    _legend(ax_catch, fontsize=7.5, loc="lower right")
    for ax in (ax_cost, ax_catch):
        _style_axes(ax)
        ax.set_xlim(0, 1)
    ax_cost.set_ylim(bottom=0)
    ax_catch.set_ylim(0, 1.04)
    ax_cost.xaxis.set_major_formatter(lambda v, _pos: f"{v:.0%}")
    ax_catch.xaxis.set_major_formatter(lambda v, _pos: f"{v:.0%}")

    fig.suptitle("Not every escape costs the same: severity-graded economics",
                 fontsize=13, color=TEXT_PRIMARY, weight="bold", y=0.965)
    fig.text(0.5, 0.928, "Parts graded by displaced intensity (a measured property of the "
             "mark), each grade priced separately [A11].\nThe severity index, the per-grade "
             "rates and the AUCs are measured; cut points and prices are labelled "
             "illustrative constants.",
             ha="center", fontsize=8, color=TEXT_SECONDARY)

    caption = "\n".join(textwrap.fill(line, width=104)
                        for line in severity_read(sev).splitlines())
    fig.text(0.5, 0.21, caption, ha="center", va="center", fontsize=7.6,
             color=TEXT_PRIMARY, linespacing=1.5,
             bbox=_caption_box(pad=0.8))
    return fig


_PERTURBATION_TITLES = {
    "gaussian_noise": "Sensor noise",
    "brightness": "Illumination shift",
    "blur": "Defocus blur",
    "contrast": "Contrast loss",
}
_PERTURBATION_XLABELS = {
    "gaussian_noise": "additive noise std (higher = worse)",
    "brightness": "brightness added (higher = worse)",
    "blur": "Gaussian sigma, px (higher = worse)",
    "contrast": "contrast gain (lower = more washout)",
}


def _robustness_figure(robust: RobustnessReport) -> plt.Figure:
    """PR-AUC vs corruption severity, one small-multiple per perturbation.

    Each method is a colored line; its clean-test PR-AUC is a dashed horizontal
    reference in the same hue, so the vertical gap reads directly as the drop.
    """
    kinds = list(robust.perturbations)
    fig, axes = _panels(2, 2, figsize=(8.27, 9.2))
    axes = axes.ravel()
    methods = list(robust.baselines)
    for ax, kind in zip(axes, kinds, strict=False):
        severities = list(robust.perturbations[kind])
        for name in methods:
            color = _method_color(name)
            ys = [
                next(
                    p.pr_auc
                    for p in robust.points
                    if p.method == name and p.perturbation == kind and p.severity == sev
                )
                for sev in severities
            ]
            ax.plot(severities, ys, marker="o", markersize=4, linewidth=1.8,
                    color=color, label=name)
            ax.axhline(robust.baselines[name].pr_auc, color=color, linewidth=1,
                       linestyle="--", alpha=0.55)
        ax.set_title(_PERTURBATION_TITLES.get(kind, kind), fontsize=10.5, color=TEXT_PRIMARY)
        ax.set_xlabel(_PERTURBATION_XLABELS.get(kind, "severity"), fontsize=8,
                      color=TEXT_SECONDARY)
        ax.set_ylabel("PR-AUC", fontsize=8, color=TEXT_SECONDARY)
        _style_axes(ax)
        ax.set_ylim(0.45, 0.9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), fontsize=8.5,
               framealpha=0.92, facecolor=SURFACE, edgecolor=GRID_COLOR,
               bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Robustness: PR-AUC under camera-realistic corruptions",
                 fontsize=12.5, color=TEXT_PRIMARY, weight="bold", y=0.95)
    fig.text(0.5, 0.915, "Detectors fit once on clean images, then re-scored on corrupted "
             "test images (no re-fitting). Dashed = clean-test PR-AUC baseline.",
             ha="center", fontsize=8.5, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))

    note = (
        "Reading: the vertical gap from each method's dashed baseline is its PR-AUC drop under "
        "that corruption. Full per-severity deltas (ROC-AUC / PR-AUC / TPR@5%FPR) are in the "
        "Robustness sheet and robustness.csv. Synthetic corruptions -- relative fragility "
        "signals, not production guarantees."
    )
    note = "\n".join(textwrap.fill(line, width=118) for line in note.splitlines())
    fig.text(0.5, 0.04, note, ha="center", va="center", fontsize=8,
             color=TEXT_SECONDARY, linespacing=1.45)
    return fig


def _spc_axis(ax, sc, limits, ymax: float) -> None:
    """One p-chart panel: series, frozen limits, zones, phases, alarms."""
    idx = [p.index for p in sc.points]
    props = [p.proportion for p in sc.points]
    # Zones and limits first (recessive), the series and alarms on top. The
    # series is a single stream, so it wears a neutral blue (CVD-safe against
    # the red status hue -- see qav/spc.py's SVG for the same choices), not a
    # method-identity color; the method is named in the axes title.
    for k in (1.0, 2.0):
        for bound in limits.zone(k):
            if 0.0 < bound < ymax:
                ax.axhline(bound, color=TEXT_SECONDARY, linewidth=0.7,
                           linestyle=":", alpha=0.5)
    ax.axhline(limits.center, color=TEXT_SECONDARY, linewidth=1.4)
    for bound in (limits.ucl, limits.lcl):
        if 0.0 < bound < ymax:
            ax.axhline(bound, color=SIGNAL, linewidth=1.3, linestyle="--")
    phase1_len = sum(1 for p in sc.points if p.phase == 1)
    ax.axvline(phase1_len - 0.5, color=GRID_COLOR, linewidth=1.2)
    ax.axvline(sc.change_index - 0.5, color=LAMP, linewidth=1.5, linestyle="--")
    ax.plot(idx, props, color=STEEL, linewidth=1.6, marker="o", markersize=3,
            markerfacecolor=SURFACE, markeredgecolor=STEEL, zorder=3)
    alarmed = [i for i, r in enumerate(sc.rules) if r.alarm]
    if alarmed:
        ax.plot([sc.points[i].index for i in alarmed],
                [sc.points[i].proportion for i in alarmed], "o", color=SIGNAL,
                markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=4)
    if sc.first_alarm is not None:
        pt = sc.points[sc.first_alarm]
        firing = [RULE_LABELS[r] for r, i in sc.first_by_rule.items() if i == sc.first_alarm]
        ax.annotate(f"first alarm: subgroup {pt.index}\n({', '.join(firing)})",
                    xy=(pt.index, pt.proportion), xytext=(-10, 30),
                    textcoords="offset points", ha="right", fontsize=7.5,
                    color=SIGNAL, weight="bold")
    ax.text(phase1_len / 2 - 0.5, ymax * 0.94, "Phase I (limits frozen)", ha="center",
            fontsize=7, color=TEXT_SECONDARY)
    ax.text(phase1_len + (len(sc.points) - phase1_len) / 2 - 0.5, ymax * 0.94,
            "Phase II (monitoring)", ha="center", fontsize=7, color=TEXT_SECONDARY)
    ax.set_xlim(-0.8, len(sc.points) - 0.2)
    ax.set_ylim(0.0, ymax)
    ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:.1%}")
    _style_axes(ax)


def _spc_figure(spc: SPCStudy) -> plt.Figure:
    """p-chart page for the PDF: both scenarios against the frozen limits."""
    fig = _page()
    ax_a = fig.add_axes((0.11, 0.70, 0.82, 0.20))
    ax_b = fig.add_axes((0.11, 0.44, 0.82, 0.20))
    ymax = max(
        max(p.proportion for sc in spc.scenarios for p in sc.points), spc.limits.ucl
    ) * 1.18
    titles = {
        "process_shift": "A - process shift: ",
        "camera_drift": "B - camera drift: ",
    }
    for ax, sc in zip((ax_a, ax_b), spc.scenarios, strict=True):
        _spc_axis(ax, sc, spc.limits, ymax)
        ax.set_title(titles.get(sc.name, "") + sc.label, fontsize=9, color=TEXT_PRIMARY,
                     loc="left")
        ax.set_ylabel("share flagged", fontsize=8, color=TEXT_SECONDARY)
    ax_b.set_xlabel(f"subgroup (one shift = {spc.config.subgroup_size:,} parts)",
                    fontsize=8.5, color=TEXT_PRIMARY)
    fig.suptitle("Is the line still in control? p-chart on the screening output",
                 fontsize=13, color=TEXT_PRIMARY, weight="bold", y=0.965)
    fig.text(0.5, 0.930, "Solid gray = center line, dashed red = 3-sigma limits, dotted = "
             "1/2-sigma zones, dashed amber = change point,\nfilled red = Western Electric "
             "rule violation. Modelled stream from measured flag rates.",
             ha="center", fontsize=8, color=TEXT_SECONDARY)

    caption = "\n\n".join(
        [
            spc_read(spc),
            "The per-class flag rates are MEASURED (fitted detector, fresh synthetic "
            "calibration parts, no re-fitting); the stream itself is MODELLED (seeded "
            "i.i.d. draws) -- see the SPC sheet / spc_chart.csv for every subgroup.",
        ]
    )
    caption = "\n".join(textwrap.fill(line, width=104) for line in caption.splitlines())
    fig.text(0.5, 0.21, caption, ha="center", va="center", fontsize=7.6,
             color=TEXT_PRIMARY, linespacing=1.45,
             bbox=_caption_box(pad=0.8))
    return fig


# One fixed (color, linestyle) per OCAP policy -- linestyle carries the
# identity redundantly to hue, mirroring the hand-drawn SVG's dash patterns.
_POLICY_PLOT_STYLE = {
    "no_action": (SIGNAL, "solid"),
    "rate_recenter": (LAMP, "dashed"),
    "refit_recent": (PATINA, "dashdot"),
    "fix_camera": (STEEL, "dotted"),
}


def _recal_figure(recal: RecalStudy) -> plt.Figure:
    """OCAP page for the PDF: expected cost of each drift response, plus the
    catch-rate panel that exposes the quiet-chart trap of rate re-centering."""
    fig = _page()
    ax_cost = fig.add_axes((0.12, 0.66, 0.80, 0.24))
    ax_catch = fig.add_axes((0.12, 0.40, 0.80, 0.18))
    deltas = [0.0, *recal.config.drift_deltas]
    base = recal.in_control
    m = recal.model
    for policy in POLICY_ORDER:
        color, style = _POLICY_PLOT_STYLE[policy]
        outs = [recal.outcome(policy, d) for d in recal.config.drift_deltas]
        costs = [base.expected_cost] + [o.expected_cost for o in outs]
        caught = [base.caught_defects] + [o.caught_defects for o in outs]
        ax_cost.plot(deltas, costs, color=color, linestyle=style, linewidth=2,
                     marker="o", markersize=4, label=POLICY_LABELS[policy])
        ax_catch.plot(deltas, caught, color=color, linestyle=style, linewidth=2,
                      marker="o", markersize=4)
    ax_cost.axhline(base.expected_cost, color=TEXT_SECONDARY, linewidth=0.9,
                    linestyle=":", alpha=0.7)
    ax_cost.set_ylabel(f"expected cost (EUR / {m.units_basis:,} parts)",
                       fontsize=8.5, color=TEXT_PRIMARY)
    ax_cost.set_title("What each response costs as the drift grows", fontsize=10,
                      color=TEXT_PRIMARY, loc="left")
    _legend(ax_cost, fontsize=7.5, loc="upper left")
    ax_catch.axhline(m.units_basis * m.prevalence, color=TEXT_SECONDARY,
                     linewidth=0.9, linestyle=":", alpha=0.7)
    ax_catch.text(deltas[-1], m.units_basis * m.prevalence * 1.03,
                  f"all {m.units_basis * m.prevalence:.0f} defects", ha="right",
                  fontsize=7, color=TEXT_SECONDARY)
    ax_catch.set_ylabel(f"defects caught / {m.units_basis:,} parts", fontsize=8.5,
                        color=TEXT_PRIMARY)
    ax_catch.set_xlabel("brightness drift the camera is running at", fontsize=9,
                        color=TEXT_PRIMARY)
    ax_catch.set_title("...and how many defects it still catches", fontsize=10,
                       color=TEXT_PRIMARY, loc="left")
    for ax in (ax_cost, ax_catch):
        _style_axes(ax)
        ax.set_xlim(0, deltas[-1] * 1.02)
        ax.set_ylim(bottom=0)
    fig.suptitle("The alarm fired -- now what? A measured out-of-control action plan",
                 fontsize=13, color=TEXT_PRIMARY, weight="bold", y=0.965)
    fig.text(0.5, 0.928, "Four responses to the camera-drift alarm, each measured on the "
             "same fresh calibration parts at every drift level.\nNote the amber trap: "
             "re-centering the threshold flattens the cost AND the catch rate -- the chart "
             "goes quiet while the screen goes blind.",
             ha="center", fontsize=8, color=TEXT_SECONDARY)

    # Condensed caption -- the full per-policy grid lives in the Recalibration
    # sheet / recalibration.csv (and recal_read() on the console).
    d0 = recal.config.drift_deltas[0]
    d_max = recal.config.drift_deltas[-1]
    na0 = recal.outcome("no_action", d0)
    na_m = recal.outcome("no_action", d_max)
    rc_m = recal.outcome("rate_recenter", d_max)
    rf_all = [recal.outcome("refit_recent", d) for d in recal.config.drift_deltas]
    n_def = m.units_basis * m.prevalence
    recovery = (
        f", recovering {recal.cost_recovered('refit_recent', d_max):.0%} of the "
        f"drift-induced cost at +{d_max:g}" if na_m.d_cost > 0 else ""
    )
    caption_lines = [
        f"In control: {base.expected_cost:,.0f} EUR per {m.units_basis:,} parts, flagging "
        f"{base.flag_rate:.2%} and catching {base.caught_defects:.1f} of {n_def:.1f} defects "
        f"(ROC-AUC {base.roc_auc:.3f}; {recal.method}, threshold {recal.base_threshold:.4f}).",
        f"Doing nothing costs {na0.expected_cost:,.0f} EUR ({na0.d_cost:+,.0f}) at the drift "
        f"level the chart alarms at (+{d0:g}) and {na_m.expected_cost:,.0f} EUR at +{d_max:g} "
        f"({na_m.flag_rate:.0%} of all parts flagged).",
        f"The trap: re-centering the threshold at +{d_max:g} returns the flag rate to "
        f"{rc_m.flag_rate:.2%} -- the chart goes quiet -- while catching {rc_m.caught_defects:.1f} "
        f"of {n_def:.1f} defects (ROC-AUC {rc_m.roc_auc:.3f}, identical to no action: a "
        "threshold cannot restore lost separability).",
        f"Re-fitting on {recal.config.refit_clean_frames} recent verified-clean frames [A10] "
        f"holds {min(o.expected_cost for o in rf_all):,.0f}-"
        f"{max(o.expected_cost for o in rf_all):,.0f} EUR at every drift level"
        f"{recovery}; repairing the camera restores the in-control point exactly.",
        "Flag rates and ROC-AUC are MEASURED (fitted or re-fitted detectors on fresh "
        "synthetic calibration parts); stream composition and cost rates are the labelled "
        "illustrative model; refit frames are assumed clean and free [A10]. Full grid: "
        "Recalibration sheet / recalibration.csv.",
    ]
    caption = "\n".join(textwrap.fill(line, width=104) for line in caption_lines)
    fig.text(0.5, 0.20, caption, ha="center", va="center", fontsize=7.6,
             color=TEXT_PRIMARY, linespacing=1.5,
             bbox=_caption_box(pad=0.8))
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


def _economics_frame(sweep: CostSweep) -> pd.DataFrame:
    """The threshold sweep as a sheet: one row per operating point (columns dict)."""
    pts = sweep.points
    return pd.DataFrame(
        {
            "threshold": [round(p.threshold, 6) for p in pts],
            "reject_rate": [round(p.reject_rate, 6) for p in pts],
            "precision_at_prevalence": [round(p.precision, 6) for p in pts],
            "recall": [round(p.tpr, 6) for p in pts],
            "fpr": [round(p.fpr, 6) for p in pts],
            "flagged_per_basis": [round(p.flagged, 3) for p in pts],
            "caught_defects_per_basis": [round(p.caught_defects, 3) for p in pts],
            "missed_defects_per_basis": [round(p.missed_defects, 3) for p in pts],
            "false_rejects_per_basis": [round(p.false_rejects, 3) for p in pts],
            "escape_cost_eur": [round(p.escape_cost, 2) for p in pts],
            "false_reject_cost_eur": [round(p.false_reject_cost, 2) for p in pts],
            "expected_cost_eur": [round(p.expected_cost, 2) for p in pts],
            "is_recommended": [1 if p is sweep.best else 0 for p in pts],
        }
    )


def _severity_frame(sev: SeverityStudy) -> pd.DataFrame:
    """The severity-graded sweep as a sheet: one row per operating point, with
    the escape ledger split by grade and the flat model's cost alongside."""
    pts = sev.points
    frame = {
        "threshold": [round(p.threshold, 6) for p in pts],
        "reject_rate": [round(p.reject_rate, 6) for p in pts],
        "fpr": [round(p.fpr, 6) for p in pts],
        "recall": [round(p.tpr, 6) for p in pts],
    }
    for g in GRADES:
        frame[f"recall_{g}"] = [round(p.tpr_by_grade[g], 6) for p in pts]
    frame["flagged_per_basis"] = [round(p.flagged, 3) for p in pts]
    frame["caught_defects_per_basis"] = [round(p.caught_defects, 3) for p in pts]
    for g in GRADES:
        frame[f"missed_{g}_per_basis"] = [round(p.missed_by_grade[g], 3) for p in pts]
    frame["missed_defects_per_basis"] = [round(p.missed_defects, 3) for p in pts]
    frame["false_rejects_per_basis"] = [round(p.false_rejects, 3) for p in pts]
    for g in GRADES:
        frame[f"escape_cost_{g}_eur"] = [round(p.escape_cost_by_grade[g], 2) for p in pts]
    frame["escape_cost_eur"] = [round(p.escape_cost, 2) for p in pts]
    frame["false_reject_cost_eur"] = [round(p.false_reject_cost, 2) for p in pts]
    frame["expected_cost_eur"] = [round(p.expected_cost, 2) for p in pts]
    frame["flat_expected_cost_eur"] = [round(p.expected_cost, 2) for p in sev.flat.points]
    frame["is_recommended_weighted"] = [1 if p is sev.best else 0 for p in pts]
    frame["is_recommended_flat"] = [1 if p is sev.flat.best else 0 for p in sev.flat.points]
    return pd.DataFrame(frame)


def _severity_grades_frame(sev: SeverityStudy) -> pd.DataFrame:
    """One row per severity grade: what it is, how much of it there is, how
    well the screen sees it, and what it costs at the recommended point."""
    ps = sev.profiles
    best = sev.best
    flat_best_pt = sev.at_flat_best
    frame = {
        "grade": [p.grade for p in ps],
        "index_low": [round(p.low, 3) for p in ps],
        "index_high": [None if p.high == float("inf") else round(p.high, 3) for p in ps],
        "escape_cost_eur": [round(p.escape_cost, 2) for p in ps],
        "n_parts": [p.n for p in ps],
        "share_of_defects": [round(p.share, 6) for p in ps],
        "mean_index": [round(p.mean_index, 4) for p in ps],
        "min_index": [round(p.min_index, 4) for p in ps],
        "max_index": [round(p.max_index, 4) for p in ps],
    }
    for kind in DEFECT_KINDS:
        frame[f"n_{kind}"] = [p.kind_counts[kind] for p in ps]
    frame["roc_auc_vs_clean"] = [round(p.roc_auc_vs_clean, 6) for p in ps]
    frame["recall_at_weighted_optimum"] = [round(best.tpr_by_grade[p.grade], 6) for p in ps]
    frame["recall_at_flat_optimum"] = [round(flat_best_pt.tpr_by_grade[p.grade], 6) for p in ps]
    frame["missed_per_basis_at_optimum"] = [round(best.missed_by_grade[p.grade], 3) for p in ps]
    frame["escape_cost_eur_at_optimum"] = [
        round(best.escape_cost_by_grade[p.grade], 2) for p in ps
    ]
    frame["share_of_escape_bill"] = [
        round(best.escape_cost_by_grade[p.grade] / best.escape_cost, 6)
        if best.escape_cost > 0 else 0.0
        for p in ps
    ]
    frame["cost_neutral_escape_cost_eur"] = [round(sev.neutral_costs[p.grade], 4) for p in ps]
    return pd.DataFrame(frame)


def _robustness_frame(robust: RobustnessReport) -> pd.DataFrame:
    """The robustness sweep as a sheet: baseline rows (perturbation ``none``)
    first, then one row per (method, perturbation, severity) with deltas."""
    # (method, perturbation, severity, roc, pr, tpr, d_roc, d_pr, d_tpr)
    records: list[tuple] = [
        (name, "none", 0.0, base.roc_auc, base.pr_auc, base.tpr_at_5pct_fpr, 0.0, 0.0, 0.0)
        for name, base in robust.baselines.items()
    ]
    records += [
        (p.method, p.perturbation, p.severity, p.roc_auc, p.pr_auc, p.tpr_at_5pct_fpr,
         p.d_roc_auc, p.d_pr_auc, p.d_tpr_at_5pct_fpr)
        for p in robust.points
    ]
    columns = ["method", "perturbation", "severity", "roc_auc", "pr_auc",
               "tpr_at_5pct_fpr", "d_roc_auc", "d_pr_auc", "d_tpr_at_5pct_fpr"]
    numeric = {c: i for i, c in enumerate(columns) if i >= 2}
    return pd.DataFrame(
        {
            c: [r[i] for r in records] if c not in numeric
            else [round(float(r[i]), 4) for r in records]
            for i, c in enumerate(columns)
        }
    )


def _spc_frame(spc: SPCStudy) -> pd.DataFrame:
    """The control-chart study as a sheet: the shared Phase I block once
    (scenario ``baseline``), then each scenario's monitored subgroups."""
    rows = chart_rows(spc)
    lim = spc.limits
    return pd.DataFrame(
        {
            "scenario": [scenario for scenario, _, _, _ in rows],
            "subgroup": [pt.index for _, pt, _, _ in rows],
            "phase": [pt.phase for _, pt, _, _ in rows],
            "n": [pt.n for _, pt, _, _ in rows],
            "true_prevalence": [round(pt.true_prevalence, 6) for _, pt, _, _ in rows],
            "brightness_delta": [round(pt.brightness_delta, 6) for _, pt, _, _ in rows],
            "p_flag_clean": [round(pt.p_flag_clean, 6) for _, pt, _, _ in rows],
            "p_flag_defective": [round(pt.p_flag_defective, 6) for _, pt, _, _ in rows],
            "defective_drawn": [pt.defective_drawn for _, pt, _, _ in rows],
            "flags": [pt.flags for _, pt, _, _ in rows],
            "proportion": [round(pt.proportion, 6) for _, pt, _, _ in rows],
            "center": [round(lim.center, 6)] * len(rows),
            "ucl": [round(lim.ucl, 6)] * len(rows),
            "lcl": [round(lim.lcl, 6)] * len(rows),
            "rule_beyond_3sigma": [int(h.beyond_3sigma) for _, _, h, _ in rows],
            "rule_2of3_beyond_2sigma": [
                int(h.two_of_three_beyond_2sigma) for _, _, h, _ in rows
            ],
            "rule_4of5_beyond_1sigma": [
                int(h.four_of_five_beyond_1sigma) for _, _, h, _ in rows
            ],
            "rule_8_same_side": [int(h.eight_same_side) for _, _, h, _ in rows],
            "any_alarm": [int(h.alarm) for _, _, h, _ in rows],
            "is_change_point": [int(change) for _, _, _, change in rows],
        }
    )


def _recalibration_frame(recal: RecalStudy) -> pd.DataFrame:
    """The action-plan study as a sheet: the in-control anchor row first, then
    one row per (drift level, policy) outcome."""
    outs = [recal.in_control, *recal.outcomes]
    return pd.DataFrame(
        {
            "policy": [o.policy for o in outs],
            "delta": [round(o.delta, 4) for o in outs],
            "threshold": [round(o.threshold, 6) for o in outs],
            "p_flag_clean": [round(o.p_clean, 6) for o in outs],
            "p_flag_defective": [round(o.p_defective, 6) for o in outs],
            "roc_auc": [round(o.roc_auc, 6) for o in outs],
            "flag_rate": [round(o.flag_rate, 6) for o in outs],
            "caught_defects_per_basis": [round(o.caught_defects, 3) for o in outs],
            "missed_defects_per_basis": [round(o.missed_defects, 3) for o in outs],
            "false_rejects_per_basis": [round(o.false_rejects, 3) for o in outs],
            "expected_cost_eur": [round(o.expected_cost, 2) for o in outs],
            "d_cost_eur": [round(o.d_cost, 2) for o in outs],
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
        ("Cost economics (illustrative)", "Threshold sweep on the recommended method's "
         "anomaly scores costs the scrap-vs-escape tradeoff. ILLUSTRATIVE, labelled rates "
         "(not guarantees): escaped defect 35 EUR [A4], false reject 3 EUR, defect prevalence "
         "1.5% [A3]; reported per 1,000 parts. TPR/FPR at each threshold are measured; the "
         "cost-minimising threshold is in the Economics sheet / cost_curve.csv."),
        ("Defect severity grading [A11]", "Every defective part carries a MEASURED severity "
         "index: the total absolute intensity its injection displaced (area and contrast "
         "together, no detector involved). Parts are graded minor / major / critical at the "
         "fixed cut points 8 and 15 on that index -- chosen once from the index distribution "
         "alone, before any cost was computed; a real line takes them from its customer defect "
         "catalogue. ILLUSTRATIVE per-grade escape costs [A11]: minor 10 EUR, major 35 EUR "
         "(anchored to [A4], so the flat model is a special case), critical 140 EUR. The "
         "Severity sheet re-costs every operating point of the SAME threshold grid with that "
         "graded ledger; the SeverityGrades sheet reports each grade's size, defect-kind "
         "composition, ROC-AUC vs clean and share of the escape bill. The grade MIX is this "
         "generator's mix, not a real line's."),
        ("Robustness stress-test", "Each detector is fit ONCE on clean training images, then "
         "re-scored (never re-fit) on test images corrupted by four camera-realistic "
         "perturbations -- gaussian_noise, brightness, blur, contrast -- each swept over three "
         "severities. The Robustness sheet reports ROC-AUC / PR-AUC / TPR@5%FPR and their delta "
         "vs the clean baseline. Corruptions are synthetic stand-ins for real camera faults; the "
         "deltas are relative fragility signals, not production guarantees."),
        ("SPC monitoring (modelled stream)", "p-chart on the screen's flag rate per subgroup "
         "(2,000 parts = one shift [A1]) with Western Electric rules on limits frozen from an "
         "in-control Phase I. The per-class flag rates are MEASURED by re-scoring fresh "
         "synthetic calibration parts (1,000 clean + 600 defective, own seed) with the fitted "
         "recommended detector at the cost-optimal threshold; the stream is MODELLED as seeded "
         "i.i.d. draws at the labelled prevalence [A3] -- real lines autocorrelate, so "
         "detection delays are illustrative chart mechanics, not performance guarantees. The "
         "camera-drift scenario changes only the imaging (brightness ramp), never the true "
         "defect rate: its alarms demonstrate that the chart confounds process and measurement "
         "system."),
        ("Out-of-control action plan (OCAP)", "When the p-chart alarms on camera drift, four "
         "responses are measured at every drift level on the SPC layer's calibration parts: "
         "no_action (frozen detector + threshold), rate_recenter (threshold re-set so the "
         "modelled unlabelled stream's flag rate returns to its in-control value -- no labels "
         "at decision time), refit_recent (fresh same-family detector fit on recent "
         "verified-clean frames captured under the drifted camera [A10], threshold "
         "rate-re-centered), fix_camera (root cause repaired -- recovers the in-control point "
         "by construction). [A10]: the refit window is assumed available and free; "
         "verification effort and recalibration downtime are NOT costed. Costs use the same "
         "illustrative rates as the economics layer; the drift is a synthetic brightness "
         "stand-in. Full grid: Recalibration sheet / recalibration.csv."),
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
    spc_config: SPCConfig | None = None,
    recal_config: RecalConfig | None = None,
) -> dict[str, int]:
    """Write PDF + Excel + README figures; return {path: size_bytes}."""
    out_path = Path(out_dir)
    fig_path = Path(fig_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    sweep = economics_for_report(report)
    sev = severity_for_report(report, sweep=sweep)
    robust = robustness_for_report(report)
    spc = spc_for_report(report, sweep=sweep, config=spc_config)
    recal = recalibration_for_report(
        report, sweep=sweep, config=recal_config, spc_config=spc_config
    )

    pdf_file = out_path / "qa_defect_report.pdf"
    gallery = _gallery_figure(report)
    curves = _curves_figure(report)
    bars = _per_type_bar_figure(report)
    robustness_fig = _robustness_figure(robust)
    with PdfPages(pdf_file) as pdf:
        for fig in (_cover_figure(report), gallery, curves, bars,
                    _table_figure(report), _economics_figure(sweep),
                    _severity_figure(sev), robustness_fig,
                    _spc_figure(spc), _recal_figure(recal)):
            pdf.savefig(fig)
    gallery.savefig(fig_path / "gallery.png", dpi=150, facecolor=SURFACE)
    curves.savefig(fig_path / "roc_pr.png", dpi=150, facecolor=SURFACE)
    bars.savefig(fig_path / "per_type_auc.png", dpi=150, facecolor=SURFACE)
    robustness_fig.savefig(fig_path / "robustness.png", dpi=150, facecolor=SURFACE)
    plt.close("all")

    # Byte-identical, dependency-light cost-curve + severity + robustness + SPC
    # + OCAP outputs.
    csv_file = out_path / "cost_curve.csv"
    svg_file = fig_path / "cost_curve.svg"
    severity_csv_file = out_path / "severity.csv"
    severity_svg_file = fig_path / "severity.svg"
    robustness_csv_file = out_path / "robustness.csv"
    spc_csv_file = out_path / "spc_chart.csv"
    spc_svg_file = fig_path / "spc_chart.svg"
    recal_csv_file = out_path / "recalibration.csv"
    recal_svg_file = fig_path / "recalibration.svg"
    write_cost_curve_csv(sweep, csv_file)
    write_cost_curve_svg(sweep, svg_file)
    write_severity_csv(sev, severity_csv_file)
    write_severity_svg(sev, severity_svg_file)
    write_robustness_csv(robust, robustness_csv_file)
    write_spc_csv(spc, spc_csv_file)
    write_spc_svg(spc, spc_svg_file)
    write_recalibration_csv(recal, recal_csv_file)
    write_recalibration_svg(recal, recal_svg_file)

    xlsx_file = out_path / "qa_defect_metrics.xlsx"
    with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
        _metrics_frame(report).to_excel(writer, sheet_name="Metrics", index=False)
        _per_type_frame(report).to_excel(writer, sheet_name="PerDefectType", index=False)
        _economics_frame(sweep).to_excel(writer, sheet_name="Economics", index=False)
        _severity_frame(sev).to_excel(writer, sheet_name="Severity", index=False)
        _severity_grades_frame(sev).to_excel(writer, sheet_name="SeverityGrades", index=False)
        _robustness_frame(robust).to_excel(writer, sheet_name="Robustness", index=False)
        _spc_frame(spc).to_excel(writer, sheet_name="SPC", index=False)
        _recalibration_frame(recal).to_excel(writer, sheet_name="Recalibration", index=False)
        _assumptions_frame(report).to_excel(writer, sheet_name="Assumptions", index=False)
        _per_image_frame(report).to_excel(writer, sheet_name="PerImageScores", index=False)

    tracked = [pdf_file, xlsx_file, csv_file, svg_file,
               severity_csv_file, severity_svg_file, robustness_csv_file,
               spc_csv_file, spc_svg_file, recal_csv_file, recal_svg_file,
               fig_path / "gallery.png", fig_path / "roc_pr.png", fig_path / "per_type_auc.png",
               fig_path / "robustness.png"]
    return {str(p): p.stat().st_size for p in tracked}
