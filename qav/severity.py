"""Defect severity grading: the escape ledger the flat cost model cannot keep.

``qav.economics`` prices every escaped defect at one rate ([A4], 35 EUR). Real
QA does not: a faint 20-pixel blob and a hand-sized texture break are not the
same liability, and every defect catalogue on a real line grades them
(minor / major / critical, the classification AQL sampling plans are written
around). This module puts that ledger under the same threshold sweep.

What is MEASURED here
---------------------

- The **severity index** of every defective part: the total absolute intensity
  the injection displaced (``sum |defective - clean|``, intensity x pixels),
  recorded by :func:`qav.data.severity_index` at generation time. It is a
  property of the mark -- area and contrast together -- and involves no
  detector, no threshold and no model.
- The **per-grade detection rates** at every threshold on the recommended
  method's scores: one true-positive rate per grade, plus the shared
  false-positive rate on clean parts. Same grid as the flat sweep
  (:func:`qav.economics.sweep_grid`), so the two optima are directly
  comparable.
- The **per-grade ROC-AUC vs clean**, i.e. how well the screen ranks each grade
  against clean parts.

What is ASSUMED (labelled, illustrative -- not guarantees)
----------------------------------------------------------

- **[A11] the grade cut points** ``GRADE_CUTS`` on the severity index and the
  **per-grade escape costs** ``SeverityModel.escape_costs``. The cut points
  were fixed once, from the index distribution alone, to split this synthetic
  population into three usable grades before any cost was computed; a real line
  takes them from its customer defect catalogue. The middle grade's escape cost
  is anchored to the study's existing [A4] (35 EUR) so the flat model stays a
  special case of this one; minor is priced at 10 EUR and critical at 140 EUR
  (4x [A4]) -- change them and re-run, that sensitivity is the point.
- **the grade mix** used to reweight the sweep to the labelled 1.5% prevalence
  [A3] is the *measured mix of the synthetic defect population*, which is a
  property of the generator, not of any real line.

The honest read the module produces
-----------------------------------

Grading escapes changes the **bill** long before it changes the **decision**:
the cost-minimising threshold is unchanged at the labelled rates (it is pinned
by the zero-false-reject cliff), while the expected escape bill at that same
point rises sharply, and most of the rise comes from the grade the screen is
*worst* at. :meth:`SeverityStudy.breakeven` searches for the critical-escape
cost at which the recommended operating point finally does move.

Pure numpy plus the standard library: no wall-clock, no RNG, no plotting
dependency. The CSV and hand-drawn SVG are byte-identical across re-runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qav.data import DEFECT_KINDS
from qav.economics import CostModel, CostSweep, economics_for_report, sweep_grid
from qav.evaluate import roc_auc
from qav.palette import GRADE_COLORS, GRID, INK, INK_MUTED, PANEL, STEEL, SURFACE

# Ordered least -> most serious. The order is load-bearing: it drives the CSV
# columns, the legend order and the ordinal color ramp.
GRADES = ("minor", "major", "critical")

# [A11] Cut points on the severity index (intensity x pixels), fixed once from
# the index distribution before any cost was computed: minor < 8 <= major
# < 15 <= critical.
GRADE_CUTS = (8.0, 15.0)

# [A11] Illustrative escape cost per grade, EUR. The middle grade is [A4].
GRADE_ESCAPE_COSTS = (10.0, 35.0, 140.0)

# Break-even search: how far above the major-grade cost to look for a critical
# cost that moves the operating point, and how many bisection steps to take.
BREAKEVEN_MAX_MULTIPLE = 100.0
BREAKEVEN_STEPS = 44


@dataclass(frozen=True)
class SeverityModel:
    """The labelled illustrative grading [A11]: cut points + escape cost each."""

    cuts: tuple[float, float] = GRADE_CUTS
    escape_costs: tuple[float, float, float] = GRADE_ESCAPE_COSTS

    def cost(self, grade: str) -> float:
        return float(self.escape_costs[GRADES.index(grade)])

    def costs(self) -> dict[str, float]:
        return {g: self.cost(g) for g in GRADES}

    def with_critical_cost(self, value: float) -> SeverityModel:
        a, b, _ = self.escape_costs
        return SeverityModel(cuts=self.cuts, escape_costs=(a, b, float(value)))

    def bounds(self, grade: str) -> tuple[float, float]:
        """(low, high) severity-index bounds of a grade; ``inf`` for the top."""
        lo, hi = self.cuts
        return {"minor": (0.0, lo), "major": (lo, hi), "critical": (hi, math.inf)}[grade]


def grade_severity(index: np.ndarray, model: SeverityModel | None = None) -> np.ndarray:
    """Grade an array of severity indices into ``GRADES`` (str array).

    Parts with index 0 (clean) grade as ``"clean"`` -- the grading is a property
    of the mark, so a part with no mark has no grade.
    """
    model = model or SeverityModel()
    lo, hi = model.cuts
    index = np.asarray(index, dtype=np.float64)
    out = np.where(index < lo, "minor", np.where(index < hi, "major", "critical"))
    return np.where(index <= 0.0, "clean", out).astype("<U8")


@dataclass(frozen=True)
class GradeProfile:
    """What one grade *is* on this dataset, and how well the screen sees it."""

    grade: str
    escape_cost: float  # [A11] EUR per escaped part of this grade
    low: float  # severity-index bounds of the grade
    high: float
    n: int  # defective test parts in this grade
    share: float  # share of all defective test parts (the modelled grade mix)
    mean_index: float
    min_index: float
    max_index: float
    kind_counts: dict[str, int]  # defect-kind composition of the grade
    roc_auc_vs_clean: float  # measured: this grade's parts vs every clean part


@dataclass(frozen=True)
class SeverityPoint:
    """One operating point, with the escape ledger split by grade.

    Counts are per ``CostModel.units_basis`` parts, reweighted to the labelled
    prevalence from the measured within-class rates, exactly as the flat sweep
    does -- only the escape *price* differs per grade.
    """

    threshold: float
    fpr: float
    tpr: float  # overall recall (mix-weighted)
    tpr_by_grade: dict[str, float]
    reject_rate: float
    flagged: float
    caught_defects: float
    missed_by_grade: dict[str, float]
    missed_defects: float
    false_rejects: float
    escape_cost_by_grade: dict[str, float]
    escape_cost: float
    false_reject_cost: float
    expected_cost: float


@dataclass(frozen=True)
class BreakEven:
    """The critical-escape cost at which the operating point finally moves."""

    critical_cost: float  # EUR per escaped critical part
    multiple: float  # ... as a multiple of the major-grade cost [A4]
    threshold: float
    reject_rate: float
    expected_cost: float
    flat_reject_rate: float  # the operating point it moves away from


@dataclass(frozen=True)
class SeverityStudy:
    """The graded sweep beside the flat one, on one shared threshold grid."""

    method: str
    model: CostModel
    severity: SeverityModel
    profiles: list[GradeProfile]
    points: list[SeverityPoint]  # ordered flag-none -> flag-all, same grid as ``flat``
    best: SeverityPoint  # cost-minimising point under graded escapes
    flat: CostSweep  # the flat-cost sweep, untouched
    at_flat_best: SeverityPoint  # the flat optimum, re-costed with graded escapes
    neutral_costs: dict[str, float]  # grade costs rescaled to the same mean as [A4]
    neutral_cost_at_best: float  # the bill at ``best`` under those rescaled rates
    breakeven: BreakEven | None

    @property
    def mix(self) -> dict[str, float]:
        return {p.grade: p.share for p in self.profiles}

    @property
    def mean_escape_cost(self) -> float:
        """Mix-weighted mean escape cost -- what [A4] would have to be to agree."""
        return float(sum(p.share * p.escape_cost for p in self.profiles))

    @property
    def moved(self) -> bool:
        """Did grading change *where the line should run*, not just the bill?"""
        return self.best.threshold != self.flat.best.threshold

    def profile(self, grade: str) -> GradeProfile:
        return next(p for p in self.profiles if p.grade == grade)


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def _ledger(
    labels: np.ndarray, scores: np.ndarray, grades: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Measured rates on the shared grid: clean FPR, and one TPR per grade."""
    defective = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    clean_scores = scores[~defective]
    fpr = np.array([float((clean_scores >= t).mean()) for t in thresholds])
    tpr: dict[str, np.ndarray] = {}
    for grade in GRADES:
        sel = defective & (grades == grade)
        if not sel.any():
            raise ValueError(f"no defective parts graded {grade!r}; grading is degenerate")
        grade_scores = scores[sel]
        tpr[grade] = np.array([float((grade_scores >= t).mean()) for t in thresholds])
    return fpr, tpr


def _point(
    threshold: float,
    fpr: float,
    tprs: dict[str, float],
    mix: dict[str, float],
    model: CostModel,
    costs: dict[str, float],
) -> SeverityPoint:
    n = model.units_basis
    p = model.prevalence
    clean_units = n * (1.0 - p)
    false_rej = clean_units * fpr
    caught = 0.0
    missed_by: dict[str, float] = {}
    escape_by: dict[str, float] = {}
    for grade in GRADES:
        units = n * p * mix[grade]
        caught += units * tprs[grade]
        missed_by[grade] = units * (1.0 - tprs[grade])
        escape_by[grade] = costs[grade] * missed_by[grade]
    missed = sum(missed_by.values())
    escape_cost = sum(escape_by.values())
    fr_cost = model.cost_false_reject * false_rej
    flagged = caught + false_rej
    return SeverityPoint(
        threshold=float(threshold),
        fpr=float(fpr),
        tpr=float(sum(mix[g] * tprs[g] for g in GRADES)),
        tpr_by_grade={g: float(tprs[g]) for g in GRADES},
        reject_rate=float(flagged / n),
        flagged=float(flagged),
        caught_defects=float(caught),
        missed_by_grade={g: float(v) for g, v in missed_by.items()},
        missed_defects=float(missed),
        false_rejects=float(false_rej),
        escape_cost_by_grade={g: float(v) for g, v in escape_by.items()},
        escape_cost=float(escape_cost),
        false_reject_cost=float(fr_cost),
        expected_cost=float(escape_cost + fr_cost),
    )


def _share(part: float, whole: float) -> float:
    """Share of a total, 0 when there is no total (a screen that misses nothing)."""
    return part / whole if whole > 0 else 0.0


def _best(points: list[SeverityPoint]) -> SeverityPoint:
    """Same tie-break as the flat sweep: cheapest, then fewest parts pulled,
    then the higher (more conservative) threshold."""
    return min(points, key=lambda pt: (round(pt.expected_cost, 6), pt.reject_rate, -pt.threshold))


def _profiles(
    labels: np.ndarray,
    scores: np.ndarray,
    grades: np.ndarray,
    types: np.ndarray,
    severity_index: np.ndarray,
    severity: SeverityModel,
) -> list[GradeProfile]:
    defective = np.asarray(labels).astype(bool)
    n_def = int(defective.sum())
    out: list[GradeProfile] = []
    for grade in GRADES:
        sel = defective & (grades == grade)
        idx = severity_index[sel]
        vs_clean = sel | ~defective
        low, high = severity.bounds(grade)
        out.append(
            GradeProfile(
                grade=grade,
                escape_cost=severity.cost(grade),
                low=float(low),
                high=float(high),
                n=int(sel.sum()),
                share=float(sel.sum() / n_def),
                mean_index=float(idx.mean()),
                min_index=float(idx.min()),
                max_index=float(idx.max()),
                kind_counts={k: int((sel & (types == k)).sum()) for k in DEFECT_KINDS},
                roc_auc_vs_clean=roc_auc(defective[vs_clean], scores[vs_clean]),
            )
        )
    return out


def _breakeven(
    thresholds: np.ndarray,
    fpr: np.ndarray,
    tpr: dict[str, np.ndarray],
    mix: dict[str, float],
    model: CostModel,
    severity: SeverityModel,
    flat_best,
) -> BreakEven | None:
    """Smallest critical-escape cost whose optimum leaves the flat optimum.

    The optimum threshold falls monotonically as the critical cost rises (more
    weight on missed defects can only argue for flagging more), so a fixed
    number of bisection steps between the major-grade cost and
    ``BREAKEVEN_MAX_MULTIPLE`` times it finds the step -- deterministically, no
    search heuristics.
    """

    def optimum(critical_cost: float) -> SeverityPoint:
        costs = severity.with_critical_cost(critical_cost).costs()
        pts = [
            _point(t, fpr[i], {g: tpr[g][i] for g in GRADES}, mix, model, costs)
            for i, t in enumerate(thresholds)
        ]
        return _best(pts)

    lo = severity.cost("major")
    hi = lo * BREAKEVEN_MAX_MULTIPLE
    if optimum(hi).threshold >= flat_best.threshold:
        return None  # no critical price within the cap moves the line
    for _ in range(BREAKEVEN_STEPS):
        mid = (lo + hi) / 2.0
        if optimum(mid).threshold < flat_best.threshold:
            hi = mid
        else:
            lo = mid
    # Round the bracket's upper end UP to whole cents: monotonicity guarantees a
    # value at or above ``hi`` still moves the point, so the reported price is
    # both clean and honest (rounding down could land back below the step).
    critical_cost = math.ceil(hi * 100.0) / 100.0
    point = optimum(critical_cost)
    return BreakEven(
        critical_cost=float(critical_cost),
        multiple=float(critical_cost / severity.cost("major")),
        threshold=point.threshold,
        reject_rate=point.reject_rate,
        expected_cost=point.expected_cost,
        flat_reject_rate=float(flat_best.reject_rate),
    )


def severity_sweep(
    labels: np.ndarray,
    scores: np.ndarray,
    severity_index: np.ndarray,
    types: np.ndarray,
    flat: CostSweep,
    severity: SeverityModel | None = None,
    method_name: str = "detector",
) -> SeverityStudy:
    """Re-cost the flat sweep's grid with one escape price per severity grade."""
    severity = severity or SeverityModel()
    model = flat.model
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    severity_index = np.asarray(severity_index, dtype=np.float64)
    types = np.asarray(types)
    grades = grade_severity(severity_index, severity)

    thresholds = sweep_grid(scores)
    fpr, tpr = _ledger(labels, scores, grades, thresholds)
    profiles = _profiles(labels, scores, grades, types, severity_index, severity)
    mix = {p.grade: p.share for p in profiles}
    costs = severity.costs()

    points = [
        _point(t, fpr[i], {g: tpr[g][i] for g in GRADES}, mix, model, costs)
        for i, t in enumerate(thresholds)
    ]
    best = _best(points)
    flat_index = next(i for i, pt in enumerate(flat.points) if pt is flat.best)
    at_flat_best = points[flat_index]

    # Cost-neutral regrade: the same grade *ratios* rescaled so their mix-weighted
    # mean equals the flat model's single rate. It separates the level effect
    # (graded escapes are dearer on average) from the shape effect (which grades
    # actually escape).
    mean_cost = sum(mix[g] * costs[g] for g in GRADES)
    factor = model.cost_escaped / mean_cost if mean_cost > 0 else 1.0
    neutral_costs = {g: costs[g] * factor for g in GRADES}
    neutral_at_best = (
        sum(neutral_costs[g] * best.missed_by_grade[g] for g in GRADES) + best.false_reject_cost
    )

    breakeven = _breakeven(thresholds, fpr, tpr, mix, model, severity, flat.best)

    return SeverityStudy(
        method=method_name,
        model=model,
        severity=severity,
        profiles=profiles,
        points=points,
        best=best,
        flat=flat,
        at_flat_best=at_flat_best,
        neutral_costs=neutral_costs,
        neutral_cost_at_best=float(neutral_at_best),
        breakeven=breakeven,
    )


def severity_for_report(
    report, sweep: CostSweep | None = None, severity: SeverityModel | None = None
) -> SeverityStudy:
    """Grade the study's own test set and re-cost the recommended method."""
    flat = sweep or economics_for_report(report)
    method = next(m for m in report.methods if m.name == flat.method)
    ds = report.dataset
    return severity_sweep(
        ds.test_labels,
        method.scores,
        ds.test_severity,
        np.array(ds.test_types),
        flat,
        severity=severity,
        method_name=flat.method,
    )


# --------------------------------------------------------------------------
# Plain-language read
# --------------------------------------------------------------------------


def plain_language_read(study: SeverityStudy) -> str:
    """A few sentences a QA lead can act on -- every number from the study."""
    m = study.model
    basis = m.units_basis
    flat_best = study.flat.best
    graded = study.at_flat_best
    worst = min(study.profiles, key=lambda p: p.roc_auc_vs_clean)
    bill = sorted(
        study.best.escape_cost_by_grade.items(), key=lambda kv: kv[1], reverse=True
    )
    top_grade, top_bill = bill[0]
    top_profile = study.profile(top_grade)
    lines = [
        f"Severity grading of the {sum(p.n for p in study.profiles)} defective test parts "
        f"(index = displaced intensity; cut points {study.severity.cuts[0]:g} / "
        f"{study.severity.cuts[1]:g} [A11]): "
        + ", ".join(
            f"{p.n} {p.grade} ({p.share:.0%}, escape {p.escape_cost:.0f} EUR)"
            for p in study.profiles
        )
        + ".",
        "Mix-weighted mean escape cost "
        f"{study.mean_escape_cost:.2f} EUR vs the flat model's {m.cost_escaped:.0f} EUR [A4].",
        "Detectability by grade (ROC-AUC vs clean, "
        f"{study.method}): "
        + ", ".join(f"{p.grade} {p.roc_auc_vs_clean:.3f}" for p in study.profiles)
        + f" -- the screen is WORST on {worst.grade} parts "
        f"({worst.kind_counts['texture_break']} of its {worst.n} are texture-breaks).",
    ]
    if study.moved:
        lines.append(
            f"Graded escapes MOVE the operating point: score >= {study.best.threshold:.4f} "
            f"(reject rate {study.best.reject_rate:.2%}) instead of the flat model's "
            f">= {flat_best.threshold:.4f} ({flat_best.reject_rate:.2%})."
        )
    else:
        lines.append(
            f"Graded escapes do NOT move the operating point: both models pick score >= "
            f"{flat_best.threshold:.4f} (reject rate {flat_best.reject_rate:.2%}). What "
            f"changes is the bill -- {graded.expected_cost:,.0f} EUR per {basis:,} parts "
            f"instead of {flat_best.expected_cost:,.0f} EUR "
            f"({graded.expected_cost / flat_best.expected_cost - 1.0:+.0%})."
        )
    lines.append(
        f"Where that bill comes from: {top_grade} escapes are "
        f"{study.best.missed_by_grade[top_grade]:.1f} of {study.best.missed_defects:.1f} "
        f"missed parts "
        f"({_share(study.best.missed_by_grade[top_grade], study.best.missed_defects):.0%}) "
        f"but {top_bill:,.0f} EUR of the {study.best.escape_cost:,.0f} EUR escape cost "
        f"({_share(top_bill, study.best.escape_cost):.0%}); that grade's parts are "
        f"{top_profile.share:.0%} of all defects."
    )
    lines.append(
        "Level vs shape: re-pricing the same grade ratios so their mean equals "
        f"{m.cost_escaped:.0f} EUR [A4] gives {study.neutral_cost_at_best:,.0f} EUR "
        f"({study.neutral_cost_at_best / flat_best.expected_cost - 1.0:+.0%} vs flat) -- "
        "so the rise is the price of severity, not a mis-shaped escape mix."
    )
    if study.breakeven is not None:
        be = study.breakeven
        lines.append(
            f"Break-even: the recommended point only moves once a critical escape is worth "
            f"{be.critical_cost:,.0f} EUR ({be.multiple:.1f}x the flat {m.cost_escaped:.0f} EUR "
            f"rate [A4]), where it jumps from {be.flat_reject_rate:.2%} to "
            f"{be.reject_rate:.2%} of parts pulled."
        )
    else:
        lines.append(
            "Break-even: no critical-escape price up to "
            f"{BREAKEVEN_MAX_MULTIPLE:.0f}x the major rate moves the recommended point."
        )
    lines.append(
        "Cut points and per-grade costs are ILLUSTRATIVE labelled constants [A11]; the "
        "severity index, the per-grade rates and the AUCs are measured on the synthetic "
        "test set. The grade mix is this generator's mix, not a real line's."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Byte-identical CSV
# --------------------------------------------------------------------------

_CSV_HEADER = (
    "threshold,reject_rate,fpr,recall,"
    + ",".join(f"recall_{g}" for g in GRADES)
    + ",flagged_per_basis,caught_defects_per_basis,"
    + ",".join(f"missed_{g}_per_basis" for g in GRADES)
    + ",missed_defects_per_basis,false_rejects_per_basis,"
    + ",".join(f"escape_cost_{g}_eur" for g in GRADES)
    + ",escape_cost_eur,false_reject_cost_eur,expected_cost_eur,flat_expected_cost_eur,"
    "is_recommended_weighted,is_recommended_flat"
)


def severity_csv(study: SeverityStudy) -> str:
    """The graded sweep as CSV text (LF line endings), one row per point."""
    rows = [_CSV_HEADER]
    for pt, flat_pt in zip(study.points, study.flat.points, strict=True):
        rows.append(
            ",".join(
                [
                    f"{pt.threshold:.6f}",
                    f"{pt.reject_rate:.6f}",
                    f"{pt.fpr:.6f}",
                    f"{pt.tpr:.6f}",
                    *(f"{pt.tpr_by_grade[g]:.6f}" for g in GRADES),
                    f"{pt.flagged:.3f}",
                    f"{pt.caught_defects:.3f}",
                    *(f"{pt.missed_by_grade[g]:.3f}" for g in GRADES),
                    f"{pt.missed_defects:.3f}",
                    f"{pt.false_rejects:.3f}",
                    *(f"{pt.escape_cost_by_grade[g]:.2f}" for g in GRADES),
                    f"{pt.escape_cost:.2f}",
                    f"{pt.false_reject_cost:.2f}",
                    f"{pt.expected_cost:.2f}",
                    f"{flat_pt.expected_cost:.2f}",
                    "1" if pt is study.best else "0",
                    "1" if flat_pt is study.flat.best else "0",
                ]
            )
        )
    return "\n".join(rows) + "\n"


def write_severity_csv(study: SeverityStudy, path: str | Path) -> int:
    text = severity_csv(study)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return len(text.encode("utf-8"))


# --------------------------------------------------------------------------
# Hand-drawn (byte-identical) SVG: the bill, then who escapes
# --------------------------------------------------------------------------

_SVG_W, _SVG_H = 820, 700
_PLOT_A = (88.0, 112.0, 610.0, 356.0)  # cost vs reject rate
_PLOT_B = (88.0, 412.0, 610.0, 560.0)  # recall by grade vs reject rate


def _nice_ceiling(value: float) -> float:
    if value <= 0:
        return 1.0
    exp = math.floor(math.log10(value))
    base = 10.0**exp
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        if value <= mult * base:
            return mult * base
    return 10.0 * base


def severity_svg(study: SeverityStudy) -> str:
    """Two panels: what graded escapes cost, and which grades actually escape.

    Deterministic: no timestamps, no random ids, coordinates rounded to two
    decimals. Every series carries a direct text label as well as its hue.
    """
    m = study.model
    pts = study.points
    flat_pts = study.flat.points
    ymax = _nice_ceiling(max(max(p.expected_cost for p in pts),
                            max(p.expected_cost for p in flat_pts)) * 1.02)

    ax0, ay0, ax1, ay1 = _PLOT_A
    bx0, by0, bx1, by1 = _PLOT_B

    def apx(rate: float) -> float:
        return ax0 + rate * (ax1 - ax0)

    def apy(cost: float) -> float:
        return ay1 - (cost / ymax) * (ay1 - ay0)

    def bpx(rate: float) -> float:
        return bx0 + rate * (bx1 - bx0)

    def bpy(recall: float) -> float:
        return by1 - recall * (by1 - by0)

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_SVG_W}" height="{_SVG_H}" '
        f'viewBox="0 0 {_SVG_W} {_SVG_H}" font-family="Segoe UI, Helvetica, Arial, sans-serif">'
    )
    parts.append(f'<rect x="0" y="0" width="{_SVG_W}" height="{_SVG_H}" fill="{SURFACE}"/>')
    parts.append(
        f'<text x="{_SVG_W / 2:.0f}" y="28" text-anchor="middle" font-size="16" '
        f'font-weight="bold" fill="{INK}">Not every escape costs the same: '
        "severity-graded inspection economics</text>"
    )
    parts.append(
        f'<text x="{_SVG_W / 2:.0f}" y="46" text-anchor="middle" font-size="10.5" '
        f'fill="{INK_MUTED}">{study.method} on the synthetic test set - parts graded by '
        f"displaced intensity, cut points {study.severity.cuts[0]:g} / "
        f"{study.severity.cuts[1]:g} [A11]</text>"
    )
    parts.append(
        f'<text x="{_SVG_W / 2:.0f}" y="61" text-anchor="middle" font-size="10.5" '
        f'fill="{INK_MUTED}">illustrative rates: escaped defect '
        + " / ".join(f"{p.escape_cost:.0f}" for p in study.profiles)
        + " EUR by grade, false reject "
        + f"{m.cost_false_reject:.0f} EUR, prevalence {m.prevalence:.1%}, "
        f"per {m.units_basis:,} parts</text>"
    )
    # Grade legend: swatch + count + price, so hue never carries the grade alone.
    legend = [
        (p.grade, f"{p.grade}: {p.n} parts ({p.share:.0%}), {p.escape_cost:.0f} EUR each")
        for p in study.profiles
    ]
    lx = 96.0
    for grade, text in legend:
        parts.append(
            f'<rect x="{lx:.1f}" y="76" width="11" height="11" rx="2" '
            f'fill="{GRADE_COLORS[grade]}"/>'
        )
        parts.append(
            f'<text x="{lx + 16:.1f}" y="85.5" font-size="9.5" fill="{INK}">{text}</text>'
        )
        lx += 34.0 + 5.6 * len(text)

    # ---- panel A: cost curves -------------------------------------------
    parts.append(
        f'<text x="{ax0:.1f}" y="{ay0 - 10:.1f}" font-size="11.5" font-weight="bold" '
        f'fill="{INK}">A - the bill: graded escapes vs one flat escape rate</text>'
    )
    for i in range(6):
        val = ymax * i / 5.0
        yy = apy(val)
        parts.append(
            f'<line x1="{ax0:.2f}" y1="{yy:.2f}" x2="{ax1:.2f}" y2="{yy:.2f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{ax0 - 8:.2f}" y="{yy + 3.5:.2f}" text-anchor="end" font-size="9" '
            f'fill="{INK_MUTED}">{val:,.0f}</text>'
        )
    for i in range(6):
        r = i / 5.0
        xx = apx(r)
        parts.append(
            f'<line x1="{xx:.2f}" y1="{ay1:.2f}" x2="{xx:.2f}" y2="{ay1 + 5:.2f}" '
            f'stroke="{INK_MUTED}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xx:.2f}" y="{ay1 + 18:.2f}" text-anchor="middle" font-size="9" '
            f'fill="{INK_MUTED}">{r:.0%}</text>'
        )
    for x_line in (ax0,):
        parts.append(
            f'<line x1="{x_line:.2f}" y1="{ay0:.2f}" x2="{x_line:.2f}" y2="{ay1:.2f}" '
            f'stroke="{INK_MUTED}" stroke-width="1.5"/>'
        )
    parts.append(
        f'<line x1="{ax0:.2f}" y1="{ay1:.2f}" x2="{ax1:.2f}" y2="{ay1:.2f}" '
        f'stroke="{INK_MUTED}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="24" y="{(ay0 + ay1) / 2:.0f}" text-anchor="middle" font-size="10.5" '
        f'fill="{INK}" transform="rotate(-90 24 {(ay0 + ay1) / 2:.0f})">'
        f"Expected cost (EUR per {m.units_basis:,} parts)</text>"
    )
    flat_poly = " ".join(
        f"{apx(p.reject_rate):.2f},{apy(p.expected_cost):.2f}" for p in flat_pts
    )
    graded_poly = " ".join(f"{apx(p.reject_rate):.2f},{apy(p.expected_cost):.2f}" for p in pts)
    parts.append(
        f'<polyline fill="none" stroke="{STEEL}" stroke-width="1.8" stroke-dasharray="6 3" '
        f'points="{flat_poly}"/>'
    )
    parts.append(
        f'<polyline fill="none" stroke="{GRADE_COLORS["critical"]}" stroke-width="2.6" '
        f'points="{graded_poly}"/>'
    )
    # Optimum markers: the graded one, and the flat one if it sits elsewhere.
    bpt = study.best
    parts.append(
        f'<line x1="{apx(bpt.reject_rate):.2f}" y1="{apy(bpt.expected_cost):.2f}" '
        f'x2="{apx(bpt.reject_rate):.2f}" y2="{ay1:.2f}" stroke="{INK_MUTED}" '
        f'stroke-width="1" stroke-dasharray="3 3"/>'
    )
    parts.append(
        f'<circle cx="{apx(bpt.reject_rate):.2f}" cy="{apy(bpt.expected_cost):.2f}" r="5" '
        f'fill="{GRADE_COLORS["critical"]}" stroke="{SURFACE}" stroke-width="1.2"/>'
    )
    fbest = study.flat.best
    parts.append(
        f'<circle cx="{apx(fbest.reject_rate):.2f}" cy="{apy(fbest.expected_cost):.2f}" r="4" '
        f'fill="{SURFACE}" stroke="{STEEL}" stroke-width="1.8"/>'
    )
    # Callout in the empty upper-left of the panel, with a leader down to the
    # marked point (the recommended point sits hard against the y axis).
    same = "the same operating point" if not study.moved else "different operating points"
    call_x, call_y = ax0 + 26.0, ay0 + 30.0
    lines = [
        (INK, "bold", f"both models recommend {same}: {bpt.reject_rate:.2%} of parts pulled"),
        (
            GRADE_COLORS["critical"],
            "normal",
            f"solid - graded escapes [A11]: {study.at_flat_best.expected_cost:,.0f} EUR/"
            f"{m.units_basis:,} "
            f"({study.at_flat_best.expected_cost / fbest.expected_cost - 1.0:+.0%})",
        ),
        (
            STEEL,
            "normal",
            f"dashed - one flat escape rate [A4]: {fbest.expected_cost:,.0f} EUR/"
            f"{m.units_basis:,}",
        ),
    ]
    if study.breakeven is not None:
        be = study.breakeven
        lines.append(
            (
                INK_MUTED,
                "normal",
                f"the point moves only once a critical escape is worth "
                f"{be.critical_cost:,.0f} EUR ({be.multiple:.1f}x [A4]) -&gt; "
                f"{be.reject_rate:.2%} pulled",
            )
        )
    parts.append(
        f'<line x1="{call_x - 8:.2f}" y1="{call_y + 4:.2f}" '
        f'x2="{apx(bpt.reject_rate) + 6:.2f}" y2="{apy(bpt.expected_cost) - 6:.2f}" '
        f'stroke="{INK_MUTED}" stroke-width="0.8" stroke-dasharray="2 3" opacity="0.7"/>'
    )
    for i, (color, weight, text) in enumerate(lines):
        weight_attr = ' font-weight="bold"' if weight == "bold" else ""
        parts.append(
            f'<text x="{call_x:.2f}" y="{call_y + i * 14:.2f}" font-size="10"{weight_attr} '
            f'fill="{color}">{text}</text>'
        )

    # ---- panel B: recall by grade ---------------------------------------
    parts.append(
        f'<text x="{bx0:.1f}" y="{by0 - 10:.1f}" font-size="11.5" font-weight="bold" '
        f'fill="{INK}">B - who actually escapes: share of each grade the screen catches</text>'
    )
    for i in range(5):
        val = i / 4.0
        yy = bpy(val)
        parts.append(
            f'<line x1="{bx0:.2f}" y1="{yy:.2f}" x2="{bx1:.2f}" y2="{yy:.2f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{bx0 - 8:.2f}" y="{yy + 3.5:.2f}" text-anchor="end" font-size="9" '
            f'fill="{INK_MUTED}">{val:.0%}</text>'
        )
    for i in range(6):
        r = i / 5.0
        xx = bpx(r)
        parts.append(
            f'<line x1="{xx:.2f}" y1="{by1:.2f}" x2="{xx:.2f}" y2="{by1 + 5:.2f}" '
            f'stroke="{INK_MUTED}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xx:.2f}" y="{by1 + 18:.2f}" text-anchor="middle" font-size="9" '
            f'fill="{INK_MUTED}">{r:.0%}</text>'
        )
    parts.append(
        f'<line x1="{bx0:.2f}" y1="{by0:.2f}" x2="{bx0:.2f}" y2="{by1:.2f}" '
        f'stroke="{INK_MUTED}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{bx0:.2f}" y1="{by1:.2f}" x2="{bx1:.2f}" y2="{by1:.2f}" '
        f'stroke="{INK_MUTED}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="24" y="{(by0 + by1) / 2:.0f}" text-anchor="middle" font-size="10.5" '
        f'fill="{INK}" transform="rotate(-90 24 {(by0 + by1) / 2:.0f})">'
        "Share of that grade caught</text>"
    )
    parts.append(
        f'<text x="{(bx0 + bx1) / 2:.0f}" y="{by1 + 42:.0f}" text-anchor="middle" '
        f'font-size="11" fill="{INK}">Reject rate '
        "(share of parts pulled for manual inspection) - both panels</text>"
    )
    dashes = {"minor": "2 3", "major": "6 3", "critical": "none"}
    for p in study.profiles:
        color = GRADE_COLORS[p.grade]
        poly = " ".join(
            f"{bpx(q.reject_rate):.2f},{bpy(q.tpr_by_grade[p.grade]):.2f}" for q in pts
        )
        dash = dashes[p.grade]
        dash_attr = f' stroke-dasharray="{dash}"' if dash != "none" else ""
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2"{dash_attr} '
            f'points="{poly}"/>'
        )
    # Direct end labels, staggered deterministically so they never collide.
    at_best = [
        (study.best.tpr_by_grade[p.grade], p.grade, p.roc_auc_vs_clean) for p in study.profiles
    ]
    ordered = sorted(at_best, key=lambda t: (-t[0], t[1]))
    ys: list[float] = []
    for recall, _, _ in ordered:
        yy = bpy(recall)
        if ys and yy < ys[-1] + 13.0:
            yy = ys[-1] + 13.0
        ys.append(yy)
    for (recall, grade, auc), yy in zip(ordered, ys, strict=True):
        parts.append(
            f'<text x="{bx1 + 8:.2f}" y="{yy + 3.5:.2f}" font-size="9" font-weight="bold" '
            f'fill="{GRADE_COLORS[grade]}">{grade}: {recall:.0%} caught (AUC {auc:.3f})</text>'
        )
    parts.append(
        f'<line x1="{bpx(study.best.reject_rate):.2f}" y1="{by0:.2f}" '
        f'x2="{bpx(study.best.reject_rate):.2f}" y2="{by1:.2f}" stroke="{INK_MUTED}" '
        f'stroke-width="1" stroke-dasharray="3 3"/>'
    )
    parts.append(
        f'<text x="{bpx(study.best.reject_rate) + 5:.2f}" y="{by0 + 12:.2f}" font-size="8.5" '
        f'fill="{INK_MUTED}">recommended point</text>'
    )

    worst = min(study.profiles, key=lambda p: p.roc_auc_vs_clean)
    parts.append(
        f'<rect x="60" y="{_SVG_H - 84:.0f}" width="{_SVG_W - 120}" height="52" rx="4" '
        f'fill="{PANEL}" stroke="{GRID}" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{_SVG_W / 2:.0f}" y="{_SVG_H - 65:.0f}" text-anchor="middle" font-size="9.5" '
        f'fill="{INK}">The {worst.grade} grade is the one the screen ranks worst '
        f"(AUC {worst.roc_auc_vs_clean:.3f}) and the one priced highest: at the recommended "
        f"point it is {study.best.missed_by_grade[worst.grade]:.1f} of "
        f"{study.best.missed_defects:.1f} escaped parts but "
        f"{_share(study.best.escape_cost_by_grade[worst.grade], study.best.escape_cost):.0%} "
        "of the escape bill.</text>"
    )
    parts.append(
        f'<text x="{_SVG_W / 2:.0f}" y="{_SVG_H - 50:.0f}" text-anchor="middle" font-size="9" '
        f'fill="{INK_MUTED}">Grading changes what the line should EXPECT TO PAY long before it '
        "changes where the line should RUN.</text>"
    )
    parts.append(
        f'<text x="{_SVG_W / 2:.0f}" y="{_SVG_H - 37:.0f}" text-anchor="middle" font-size="8.5" '
        f'fill="{INK_MUTED}">Severity index, per-grade rates and AUCs are MEASURED on the '
        "synthetic test set; cut points and per-grade costs are illustrative labelled "
        "constants [A11]. Full grid: severity.csv / Severity sheet.</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_severity_svg(study: SeverityStudy, path: str | Path) -> int:
    text = severity_svg(study)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return len(text.encode("utf-8"))
