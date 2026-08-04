"""Defect-cost / inspection-threshold economics: the classic QA scrap-vs-escape tradeoff.

The detectors in this project turn each part into one anomaly score. A screening
station has to pick a *threshold*: flag every part scoring at or above it for
manual inspection, let the rest ship. This module sweeps that threshold across
the recommended method's scores and, at every operating point, reports

- **precision / recall** of the flag,
- **reject rate** -- the share of parts pulled for manual inspection, i.e. the
  inspector's workload,
- **escaped defects** -- defective parts that scored below the threshold and
  shipped,
- **expected cost** = ``cost_escaped_defect * (missed defects)`` +
  ``cost_false_reject * (good units flagged)``.

It then returns the cost-minimising operating threshold with a plain-language
read: pull the threshold down and you catch more defects but scrap/re-inspect
more good parts; push it up and you save inspection effort but let more defects
escape. The minimum is where those two costs balance.

Honesty scope, stated plainly:

- The true-positive and false-positive **rates at every threshold are MEASURED**
  from the synthetic test set (procedural textures with injected defects, one
  seed) -- they are not invented.
- The three money/rate inputs in :class:`CostModel` are **ILLUSTRATIVE
  constants** for a plausible finishing line, not guarantees. They mirror the
  labelled assumptions [A3] (defect prevalence) and [A4] (escaped-defect cost)
  in ``docs/BUSINESS_CASE.md`` so the two documents stay consistent; the
  false-reject cost is a fourth labelled assumption. Change any of them and the
  recommended threshold moves -- surfacing that sensitivity is the whole point.
- Because a real finishing line runs a low defect prevalence (here 1.5%), all
  counts are reported per :data:`UNITS_BASIS` parts passing the station, with
  the measured within-class rates reweighted to that prevalence. Precision is
  therefore the honest low-prevalence precision, not the balanced-test-set
  figure the PR-AUC reflects.

Everything here is pure numpy plus the standard library: no wall-clock, no RNG,
no plotting dependency. The CSV and hand-drawn SVG it emits are byte-identical
across re-runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- Illustrative constants (NOT guarantees); see module docstring and BUSINESS_CASE.md ---
DEFECT_PREVALENCE = 0.015  # [A3] share of parts that are truly defective
COST_ESCAPED_DEFECT_EUR = 35.0  # [A4] downstream cost of one defect that ships (scrap/rework/complaint)
COST_FALSE_REJECT_EUR = 3.0  # [A-FR] a good part pulled for manual re-inspection, with a fraction re-scrapped
UNITS_BASIS = 1000  # costs and counts are reported per this many parts passing the station


@dataclass(frozen=True)
class CostModel:
    """The labelled illustrative economics. Defaults mirror BUSINESS_CASE.md."""

    prevalence: float = DEFECT_PREVALENCE
    cost_escaped: float = COST_ESCAPED_DEFECT_EUR
    cost_false_reject: float = COST_FALSE_REJECT_EUR
    units_basis: int = UNITS_BASIS


@dataclass(frozen=True)
class ThresholdPoint:
    """One operating point: flag every part scoring >= ``threshold``.

    Counts are per ``CostModel.units_basis`` parts, reweighted to the model's
    defect prevalence from the test-set within-class rates (``tpr``, ``fpr``).
    """

    threshold: float
    tpr: float  # recall: share of true defects flagged
    fpr: float  # share of good parts flagged
    precision: float  # at the model's prevalence (nan when nothing is flagged)
    reject_rate: float  # share of ALL parts flagged = inspector workload
    flagged: float  # parts pulled for manual inspection, per basis
    caught_defects: float  # defects flagged (correctly), per basis
    missed_defects: float  # defects that escaped, per basis
    false_rejects: float  # good parts flagged (wrongly), per basis
    escape_cost: float  # cost_escaped * missed_defects
    false_reject_cost: float  # cost_false_reject * false_rejects
    expected_cost: float  # escape_cost + false_reject_cost, per basis


@dataclass(frozen=True)
class CostSweep:
    """The full threshold sweep plus the cost-minimising point and endpoints."""

    method: str
    model: CostModel
    points: list[ThresholdPoint]  # ordered by ascending reject rate (flag-none -> flag-all)
    best: ThresholdPoint  # cost-minimising operating point
    flag_none: ThresholdPoint  # flag nothing: every defect escapes
    flag_all: ThresholdPoint  # flag everything: no escapes, maximum scrap


def _point(threshold: float, tpr: float, fpr: float, model: CostModel) -> ThresholdPoint:
    n = model.units_basis
    p = model.prevalence
    defective_units = n * p
    clean_units = n * (1.0 - p)
    caught = defective_units * tpr
    missed = defective_units * (1.0 - tpr)
    false_rej = clean_units * fpr
    flagged = caught + false_rej
    escape_cost = model.cost_escaped * missed
    false_reject_cost = model.cost_false_reject * false_rej
    precision = caught / flagged if flagged > 0 else math.nan
    return ThresholdPoint(
        threshold=float(threshold),
        tpr=float(tpr),
        fpr=float(fpr),
        precision=float(precision),
        reject_rate=float(flagged / n),
        flagged=float(flagged),
        caught_defects=float(caught),
        missed_defects=float(missed),
        false_rejects=float(false_rej),
        escape_cost=float(escape_cost),
        false_reject_cost=float(false_reject_cost),
        expected_cost=float(escape_cost + false_reject_cost),
    )


def sweep_thresholds(
    labels: np.ndarray,
    scores: np.ndarray,
    model: CostModel | None = None,
    method_name: str = "detector",
) -> CostSweep:
    """Sweep the score threshold and cost out every operating point.

    Thresholds are placed at the midpoints between consecutive distinct scores,
    plus one padded point below the minimum (flag everything) and one above the
    maximum (flag nothing) -- so the sweep spans the whole reject-rate range and
    every threshold sits strictly between two observed scores. A part is flagged
    when ``score >= threshold``.
    """
    model = model or CostModel()
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("sweep_thresholds needs at least one defective and one clean unit")

    uniq = np.unique(scores)
    gap = max((float(uniq[-1]) - float(uniq[0])) * 1e-3, 1e-9)
    midpoints = (uniq[:-1] + uniq[1:]) / 2.0
    thresholds = np.concatenate([[uniq[-1] + gap], midpoints[::-1], [uniq[0] - gap]])

    points: list[ThresholdPoint] = []
    for t in thresholds:
        flagged = scores >= t
        tpr = float((flagged & labels).sum()) / n_pos
        fpr = float((flagged & ~labels).sum()) / n_neg
        points.append(_point(t, tpr, fpr, model))

    # Cost-minimising point. Deterministic tie-break: fewer parts pulled for
    # inspection first, then the higher (more conservative) threshold.
    best = min(points, key=lambda pt: (round(pt.expected_cost, 6), pt.reject_rate, -pt.threshold))
    return CostSweep(
        method=method_name,
        model=model,
        points=points,
        best=best,
        flag_none=points[0],
        flag_all=points[-1],
    )


def economics_for_report(report, model: CostModel | None = None) -> CostSweep:
    """Run the sweep on the method the study recommends (its scores + the test labels)."""
    name = report.recommendation.method
    method = next(m for m in report.methods if m.name == name)
    return sweep_thresholds(report.dataset.test_labels, method.scores, model, method_name=name)


def plain_language_read(sweep: CostSweep) -> str:
    """A few sentences a QA lead can act on -- all numbers straight from the sweep."""
    m = sweep.model
    b = sweep.best
    basis = m.units_basis
    lines = [
        f"Recommended operating threshold for {sweep.method}: score >= {b.threshold:.4f}.",
        f"At that threshold, out of every {basis:,} parts the station screens "
        f"(defect prevalence {m.prevalence:.1%}, illustrative):",
        f"  - flag {b.flagged:.0f} parts for manual inspection "
        f"(reject rate {b.reject_rate:.1%} = inspector workload),",
        f"  - catch {b.caught_defects:.1f} of the {basis * m.prevalence:.1f} true defects "
        f"(recall {b.tpr:.1%}); {b.missed_defects:.1f} escape,",
        f"  - wrongly pull {b.false_rejects:.1f} good parts (precision {b.precision:.1%} "
        "at this prevalence -- low precision is normal when defects are rare),",
        f"  - expected cost {b.expected_cost:,.0f} EUR per {basis:,} parts "
        f"(= {m.cost_escaped:.0f} EUR x {b.missed_defects:.1f} escaped "
        f"+ {m.cost_false_reject:.0f} EUR x {b.false_rejects:.1f} false rejects).",
        f"For comparison: flag nothing and every defect escapes at "
        f"{sweep.flag_none.expected_cost:,.0f} EUR/{basis:,}; flag everything and scrap/re-inspect "
        f"floods the line at {sweep.flag_all.expected_cost:,.0f} EUR/{basis:,}.",
        "The tradeoff: lower the threshold to catch more defects (fewer escapes) but pull more "
        "good parts (more scrap/re-inspection); raise it to save inspection effort but let more "
        "defects ship. The recommended point is where those two costs balance for the labelled, "
        "illustrative cost rates -- re-run with your own rates to move it.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Byte-identical CSV
# --------------------------------------------------------------------------

_CSV_HEADER = (
    "threshold,reject_rate,precision_at_prevalence,recall,fpr,"
    "flagged_per_basis,caught_defects_per_basis,missed_defects_per_basis,"
    "false_rejects_per_basis,escape_cost_eur,false_reject_cost_eur,expected_cost_eur,is_recommended"
)


def _num(x: float, decimals: int) -> str:
    if math.isnan(x):
        return ""
    return f"{x:.{decimals}f}"


def cost_curve_csv(sweep: CostSweep) -> str:
    """The sweep as CSV text (LF line endings, no trailing newline drift)."""
    rows = [_CSV_HEADER]
    for pt in sweep.points:
        recommended = "1" if pt is sweep.best else "0"
        rows.append(
            ",".join(
                [
                    _num(pt.threshold, 6),
                    _num(pt.reject_rate, 6),
                    _num(pt.precision, 6),
                    _num(pt.tpr, 6),
                    _num(pt.fpr, 6),
                    _num(pt.flagged, 3),
                    _num(pt.caught_defects, 3),
                    _num(pt.missed_defects, 3),
                    _num(pt.false_rejects, 3),
                    _num(pt.escape_cost, 2),
                    _num(pt.false_reject_cost, 2),
                    _num(pt.expected_cost, 2),
                    recommended,
                ]
            )
        )
    return "\n".join(rows) + "\n"


def write_cost_curve_csv(sweep: CostSweep, path: str | Path) -> int:
    text = cost_curve_csv(sweep)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return len(text.encode("utf-8"))


# --------------------------------------------------------------------------
# Hand-drawn (byte-identical) SVG cost curve
# --------------------------------------------------------------------------

_SVG_W, _SVG_H = 780, 470
_PLOT = (86.0, 55.0, 560.0, 372.0)  # x0, y0, x1, y1 of the plot rectangle
_COL_TOTAL = "#b3261e"
_COL_ESCAPE = "#2a78d6"
_COL_SCRAP = "#008300"
_COL_AXIS = "#52514e"
_COL_GRID = "#d9d8d4"
_COL_TEXT = "#0b0b0b"


def _nice_ceiling(value: float) -> float:
    """Round a positive value up to a clean 1/2/5 x 10^k grid maximum."""
    if value <= 0:
        return 1.0
    exp = math.floor(math.log10(value))
    base = 10.0**exp
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        if value <= mult * base:
            return mult * base
    return 10.0 * base


def _fmt_eur(value: float) -> str:
    return f"{value:,.0f}"


def cost_curve_svg(sweep: CostSweep) -> str:
    """A self-contained SVG of the scrap-vs-escape cost curve, drawn by hand.

    Three polylines over the reject rate (x): total expected cost (bold),
    escaped-defect cost (falling) and false-reject/scrap cost (rising). The
    cost-minimising operating point is marked. Deterministic: no timestamps,
    no random ids, coordinates rounded to two decimals.
    """
    x0, y0, x1, y1 = _PLOT
    pts = sweep.points
    ymax = _nice_ceiling(max(p.expected_cost for p in pts) * 1.02)

    def px(reject_rate: float) -> float:
        return x0 + reject_rate * (x1 - x0)

    def py(cost: float) -> float:
        return y1 - (cost / ymax) * (y1 - y0)

    def poly(getter) -> str:
        return " ".join(f"{px(p.reject_rate):.2f},{py(getter(p)):.2f}" for p in pts)

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_SVG_W}" height="{_SVG_H}" '
        f'viewBox="0 0 {_SVG_W} {_SVG_H}" font-family="Segoe UI, Helvetica, Arial, sans-serif">'
    )
    parts.append(f'<rect x="0" y="0" width="{_SVG_W}" height="{_SVG_H}" fill="white"/>')
    parts.append(
        f'<text x="{_SVG_W / 2:.0f}" y="26" text-anchor="middle" font-size="16" '
        f'font-weight="bold" fill="{_COL_TEXT}">Inspection-threshold economics: scrap vs escape'
        "</text>"
    )
    m = sweep.model
    parts.append(
        f'<text x="{_SVG_W / 2:.0f}" y="44" text-anchor="middle" font-size="10.5" '
        f'fill="{_COL_AXIS}">{sweep.method} on the synthetic test set - illustrative rates: '
        f"escaped defect {m.cost_escaped:.0f} EUR, false reject {m.cost_false_reject:.0f} EUR, "
        f"prevalence {m.prevalence:.1%}, per {m.units_basis:,} parts</text>"
    )

    # y grid + ticks
    for i in range(6):
        val = ymax * i / 5.0
        yy = py(val)
        parts.append(
            f'<line x1="{x0:.2f}" y1="{yy:.2f}" x2="{x1:.2f}" y2="{yy:.2f}" '
            f'stroke="{_COL_GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x0 - 8:.2f}" y="{yy + 3.5:.2f}" text-anchor="end" font-size="9" '
            f'fill="{_COL_AXIS}">{_fmt_eur(val)}</text>'
        )
    # x ticks (reject rate)
    for i in range(6):
        r = i / 5.0
        xx = px(r)
        parts.append(
            f'<line x1="{xx:.2f}" y1="{y1:.2f}" x2="{xx:.2f}" y2="{y1 + 5:.2f}" '
            f'stroke="{_COL_AXIS}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xx:.2f}" y="{y1 + 18:.2f}" text-anchor="middle" font-size="9" '
            f'fill="{_COL_AXIS}">{r:.0%}</text>'
        )
    # axes
    parts.append(
        f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x0:.2f}" y2="{y1:.2f}" '
        f'stroke="{_COL_AXIS}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{x0:.2f}" y1="{y1:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
        f'stroke="{_COL_AXIS}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{(x0 + x1) / 2:.0f}" y="{y1 + 40:.0f}" text-anchor="middle" font-size="11" '
        f'fill="{_COL_TEXT}">Reject rate (share of parts pulled for manual inspection)</text>'
    )
    parts.append(
        f'<text x="22" y="{(y0 + y1) / 2:.0f}" text-anchor="middle" font-size="11" '
        f'fill="{_COL_TEXT}" transform="rotate(-90 22 {(y0 + y1) / 2:.0f})">'
        f"Expected cost (EUR per {m.units_basis:,} parts)</text>"
    )

    # curves: escaped-defect cost, false-reject cost, then total on top
    parts.append(
        f'<polyline fill="none" stroke="{_COL_ESCAPE}" stroke-width="1.8" '
        f'stroke-dasharray="6 3" points="{poly(lambda p: p.escape_cost)}"/>'
    )
    parts.append(
        f'<polyline fill="none" stroke="{_COL_SCRAP}" stroke-width="1.8" '
        f'stroke-dasharray="2 3" points="{poly(lambda p: p.false_reject_cost)}"/>'
    )
    parts.append(
        f'<polyline fill="none" stroke="{_COL_TOTAL}" stroke-width="2.6" '
        f'points="{poly(lambda p: p.expected_cost)}"/>'
    )

    # optimum marker + annotation
    b = sweep.best
    bx, by = px(b.reject_rate), py(b.expected_cost)
    parts.append(
        f'<line x1="{bx:.2f}" y1="{by:.2f}" x2="{bx:.2f}" y2="{y1:.2f}" '
        f'stroke="{_COL_TOTAL}" stroke-width="1" stroke-dasharray="3 3"/>'
    )
    parts.append(f'<circle cx="{bx:.2f}" cy="{by:.2f}" r="5" fill="{_COL_TOTAL}"/>')
    label_x = min(bx + 12, x1 - 150)
    parts.append(
        f'<text x="{label_x:.2f}" y="{by - 10:.2f}" font-size="10.5" font-weight="bold" '
        f'fill="{_COL_TOTAL}">recommended: {b.reject_rate:.1%} rejected</text>'
    )
    parts.append(
        f'<text x="{label_x:.2f}" y="{by + 4:.2f}" font-size="10.5" fill="{_COL_TOTAL}">'
        f"{_fmt_eur(b.expected_cost)} EUR/{m.units_basis:,} (score >= {b.threshold:.3f})</text>"
    )

    # legend (top-right inside the plot)
    lx, ly = x1 - 172, y0 + 8
    legend = [
        (_COL_TOTAL, "total expected cost"),
        (_COL_ESCAPE, "escaped-defect cost"),
        (_COL_SCRAP, "false-reject / scrap cost"),
    ]
    parts.append(
        f'<rect x="{lx - 8:.2f}" y="{ly - 4:.2f}" width="168" height="56" rx="4" '
        f'fill="white" fill-opacity="0.85" stroke="{_COL_GRID}" stroke-width="1"/>'
    )
    for i, (color, text) in enumerate(legend):
        yy = ly + 12 + i * 16
        parts.append(
            f'<line x1="{lx:.2f}" y1="{yy:.2f}" x2="{lx + 22:.2f}" y2="{yy:.2f}" '
            f'stroke="{color}" stroke-width="2.6"/>'
        )
        parts.append(
            f'<text x="{lx + 28:.2f}" y="{yy + 3.5:.2f}" font-size="9.5" '
            f'fill="{_COL_TEXT}">{text}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_cost_curve_svg(sweep: CostSweep, path: str | Path) -> int:
    text = cost_curve_svg(sweep)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return len(text.encode("utf-8"))
