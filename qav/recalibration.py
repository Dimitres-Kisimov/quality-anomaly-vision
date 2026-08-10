"""Out-of-control action plan (OCAP): when the p-chart alarms on camera
drift, which corrective action recovers the screen -- and how much?

The SPC layer ends on an instruction: an out-of-control signal means
*investigate the measurement system before blaming the line*. This module
measures what happens next. The camera has drifted (a global brightness shift,
the corruption the robustness stress-test showed the recommended PCA screen is
most fragile to), the chart has alarmed -- and the process engineer has a menu
of responses. Four are measured here, at every drift severity, on the same
fresh calibration parts the SPC layer already uses:

- ``no_action``       keep running: original detector, original cost-optimal
                      threshold, drifted camera. The do-nothing baseline every
                      other response is judged against.
- ``rate_recenter``   keep the detector, re-set the threshold so the modelled
                      (unlabelled) stream's flag rate returns to its in-control
                      value. Needs no labels and no images beyond the live
                      stream -- the cheapest response, and the most dangerous:
                      it also quiets the control chart, whether or not the
                      screen still works.
- ``refit_recent``    re-fit a fresh detector of the same family on a window of
                      recent *verified-clean* frames captured under the drifted
                      camera [A10], then rate-re-center its threshold. The
                      "recalibrate on recent data" response.
- ``fix_camera``      repair the root cause (brightness restored to zero),
                      original detector and threshold. Recovers the in-control
                      operating point exactly, by construction -- the yardstick.

Every policy is scored the same way: per-class flag rates measured on the
calibration set (:func:`qav.spc.measured_flag_rates`), separability as the
calibration ROC-AUC, and the expected cost per 1,000 parts from the economics
module's labelled cost model ([A3], [A4], false-reject rate) -- so the answer
arrives in the same EUR units the threshold was chosen in.

What is measured and what is modelled, stated plainly:

- **Measured:** every flag rate and ROC-AUC, on the same fresh synthetic
  calibration parts as the SPC layer (fitted detectors, re-fit only where the
  policy says so). The drift is applied with :func:`qav.robustness.perturb`,
  identical to the robustness and SPC layers.
- **Modelled:** the stream composition (the labelled prevalence [A3] weights
  the per-class rates into a stream flag rate and an expected cost), and the
  drift itself (a synthetic brightness stand-in for lamp aging / exposure
  creep). ``rate_recenter`` models threshold re-centering on the *unlabelled*
  production stream by matching the prevalence-weighted mixture of the
  calibration score distributions -- no labels are used at decision time.
- **Assumed [A10]:** ``refit_recent`` presumes a window of recent frames
  verified clean (golden samples / inspector-cleared parts) is available at
  zero modelled cost; verification effort and recalibration downtime are NOT
  costed. The refit window is a fresh seeded draw from the same synthetic
  part family, drifted like the live camera.

Pure numpy plus the standard library on top of the existing pipeline: no
wall-clock, no plotting dependency. The CSV and hand-drawn SVG are
byte-identical across re-runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qav.baselines import LocalStatsDetector, PCAReconstruction
from qav.data import DataConfig, make_dataset
from qav.economics import CostModel, _nice_ceiling, _point
from qav.evaluate import roc_auc
from qav.robustness import perturb
from qav.spc import SPCConfig, calibration_dataset, measured_flag_rates

# --- Defaults (labelled; see docstring and BUSINESS_CASE.md) ---
# Drift severities: +0.02 is the SPC camera-drift scenario's endpoint (the
# level the chart actually alarmed at); 0.05 / 0.10 / 0.20 are the robustness
# sweep's brightness severities, so the two studies share their x-axis.
DRIFT_DELTAS: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)
REFIT_CLEAN_FRAMES = 200  # recent verified-clean window for the refit policy [A10]
REFIT_SEED = 5151  # its own seed: the window is NOT the original training set

POLICY_ORDER = ("no_action", "rate_recenter", "refit_recent", "fix_camera")
POLICY_LABELS = {
    "no_action": "keep running (no action)",
    "rate_recenter": "re-center threshold on the unlabelled stream",
    "refit_recent": "re-fit detector on recent verified-clean frames",
    "fix_camera": "repair the camera (root cause)",
}


@dataclass(frozen=True)
class RecalConfig:
    """Drift grid and refit-window shape. Tests use tiny values."""

    drift_deltas: tuple[float, ...] = DRIFT_DELTAS
    refit_clean_frames: int = REFIT_CLEAN_FRAMES
    refit_seed: int = REFIT_SEED


@dataclass(frozen=True)
class PolicyOutcome:
    """One (policy, drift level) operating point, measured and costed.

    Counts are per ``CostModel.units_basis`` parts at the labelled prevalence,
    exactly like the economics module's :class:`ThresholdPoint`.
    """

    policy: str
    delta: float  # brightness drift the camera is running at (0 = repaired)
    threshold: float  # the threshold this policy operates
    p_clean: float  # measured P(flag | clean part)
    p_defective: float  # measured P(flag | defective part)
    roc_auc: float  # separability on the calibration set under this policy
    flag_rate: float  # modelled stream flag rate at the labelled prevalence
    caught_defects: float
    missed_defects: float
    false_rejects: float
    expected_cost: float  # EUR per units_basis, labelled illustrative rates
    d_cost: float  # vs the in-control operating point (positive = money lost)


@dataclass(frozen=True)
class RecalStudy:
    """The full action-plan study: one in-control anchor, then every
    (drift level, policy) outcome in ``POLICY_ORDER`` within each level."""

    method: str
    base_threshold: float
    target_flag_rate: float  # the in-control stream flag rate policies re-center to
    model: CostModel
    config: RecalConfig
    in_control: PolicyOutcome
    outcomes: list[PolicyOutcome]

    def outcome(self, policy: str, delta: float) -> PolicyOutcome:
        return next(
            o for o in self.outcomes if o.policy == policy and o.delta == float(delta)
        )

    def cost_recovered(self, policy: str, delta: float) -> float:
        """Share of the drift-induced extra cost this policy removes at
        ``delta`` (1 = fully recovered, 0 = no better than doing nothing).
        Only defined where doing nothing actually costs extra."""
        lost = self.outcome("no_action", delta).d_cost
        if lost <= 0.0:
            raise ValueError(f"no drift-induced cost to recover at delta {delta}")
        return 1.0 - self.outcome(policy, delta).d_cost / lost


# --------------------------------------------------------------------------
# Pure, hand-checkable pieces
# --------------------------------------------------------------------------


def rate_matched_threshold(
    clean_scores: np.ndarray,
    defect_scores: np.ndarray,
    prevalence: float,
    target_rate: float,
) -> float:
    """The threshold that returns the modelled stream's flag rate to
    ``target_rate``, using no labels at decision time.

    The unlabelled stream's score distribution is modelled as the
    prevalence-weighted mixture of the two calibration score distributions:
    ``rate(t) = prevalence * P(defect score >= t) + (1 - prevalence) *
    P(clean score >= t)``. Candidate thresholds sit at the midpoints between
    consecutive distinct scores plus one padded point beyond each end (the
    same construction as the economics sweep); the candidate with the mixture
    rate closest to ``target_rate`` wins, ties going to the higher (more
    conservative) threshold.
    """
    clean = np.asarray(clean_scores, dtype=np.float64)
    defect = np.asarray(defect_scores, dtype=np.float64)
    if clean.size == 0 or defect.size == 0:
        raise ValueError("rate_matched_threshold needs scores from both classes")
    if not 0.0 < prevalence < 1.0:
        raise ValueError(f"prevalence must be strictly inside (0, 1), got {prevalence}")
    if not 0.0 <= target_rate <= 1.0:
        raise ValueError(f"target_rate must lie in [0, 1], got {target_rate}")

    uniq = np.unique(np.concatenate([clean, defect]))
    gap = max((float(uniq[-1]) - float(uniq[0])) * 1e-3, 1e-9)
    midpoints = (uniq[:-1] + uniq[1:]) / 2.0
    candidates = np.concatenate([[uniq[-1] + gap], midpoints[::-1], [uniq[0] - gap]])

    def mixture_rate(t: float) -> float:
        return float(
            prevalence * (defect >= t).mean() + (1.0 - prevalence) * (clean >= t).mean()
        )

    best = min(
        (float(t) for t in candidates),
        key=lambda t: (round(abs(mixture_rate(t) - target_rate), 12), -t),
    )
    return best


def outcome_from_rates(
    policy: str,
    delta: float,
    threshold: float,
    rates,
    auc: float,
    model: CostModel,
    base_cost: float | None = None,
) -> PolicyOutcome:
    """Cost one measured operating point with the economics module's arithmetic
    (``rates`` needs ``p_clean`` / ``p_defective``, e.g. :class:`qav.spc.FlagRates`)."""
    pt = _point(threshold, tpr=rates.p_defective, fpr=rates.p_clean, model=model)
    return PolicyOutcome(
        policy=policy,
        delta=float(delta),
        threshold=float(threshold),
        p_clean=float(rates.p_clean),
        p_defective=float(rates.p_defective),
        roc_auc=float(auc),
        flag_rate=pt.reject_rate,
        caught_defects=pt.caught_defects,
        missed_defects=pt.missed_defects,
        false_rejects=pt.false_rejects,
        expected_cost=pt.expected_cost,
        d_cost=pt.expected_cost - base_cost if base_cost is not None else 0.0,
    )


def refit_factory_for(detector):
    """A zero-argument constructor for a fresh, unfitted detector of the same
    family and settings as ``detector`` (the refit policy re-fits from scratch;
    it never mutates the running detector)."""
    if isinstance(detector, PCAReconstruction):
        return lambda: PCAReconstruction(
            n_components=detector.n_components, smooth_sigma=detector.smooth_sigma
        )
    if isinstance(detector, LocalStatsDetector):
        return lambda: LocalStatsDetector(
            window=detector.window, smooth_sigma=detector.smooth_sigma
        )
    try:  # torch-gated: only reachable when the fitted detector came from torch
        from qav.model import ConvAutoencoderDetector
    except ImportError:  # pragma: no cover - torch absent implies classical detector
        ConvAutoencoderDetector = None
    if ConvAutoencoderDetector is not None and isinstance(detector, ConvAutoencoderDetector):
        return lambda: ConvAutoencoderDetector(detector.config)
    raise ValueError(
        f"no refit recipe for detector of type {type(detector).__name__}; "
        "expected PCAReconstruction, LocalStatsDetector or ConvAutoencoderDetector"
    )


# --------------------------------------------------------------------------
# The study
# --------------------------------------------------------------------------


def refit_window(config: RecalConfig, image_size: int = 64) -> np.ndarray:
    """The recent verified-clean frames [A10]: a fresh seeded draw from the
    same synthetic part family (its own seed -- NOT the original training set).
    The caller drifts them like the live camera before fitting."""
    if config.refit_clean_frames < 2:
        raise ValueError("refit_clean_frames must be >= 2 to fit a detector")
    return make_dataset(
        DataConfig(
            n_train=config.refit_clean_frames,
            n_test_clean=1,
            n_test_defective=0,
            image_size=image_size,
            seed=config.refit_seed,
        )
    ).train


def run_recalibration_study(
    detector,
    threshold: float,
    method_name: str,
    refit_factory=None,
    config: RecalConfig | None = None,
    spc_config: SPCConfig | None = None,
    cost_model: CostModel | None = None,
    image_size: int = 64,
) -> RecalStudy:
    """Measure all four policies at every drift level.

    ``detector`` is the already-fitted screen (never mutated); ``threshold``
    is its operating threshold (the economics module's cost-optimal point in
    the deliverables). The calibration set is the SPC layer's (same config,
    same seed), so all monitoring-layer numbers share one measurement basis.
    """
    cfg = config or RecalConfig()
    if not cfg.drift_deltas:
        raise ValueError("drift_deltas must contain at least one level")
    if any(d < 0.0 for d in cfg.drift_deltas):
        raise ValueError("drift_deltas must be non-negative brightness shifts")
    refit_factory = refit_factory or refit_factory_for(detector)
    model = cost_model or CostModel()

    cal = calibration_dataset(spc_config or SPCConfig(), image_size)
    labels = np.asarray(cal.test_labels).astype(bool)

    base_scores = detector.scores(cal.test_images)
    base_rates = measured_flag_rates(base_scores, labels, threshold)
    base_auc = roc_auc(labels, base_scores)
    in_control = outcome_from_rates(
        "in_control", 0.0, threshold, base_rates, base_auc, model
    )
    target = in_control.flag_rate

    window = refit_window(cfg, image_size)

    outcomes: list[PolicyOutcome] = []
    for delta in cfg.drift_deltas:
        drifted = perturb(cal.test_images, "brightness", delta)
        drift_scores = detector.scores(drifted)
        drift_auc = roc_auc(labels, drift_scores)

        # no_action: original detector and threshold meet the drifted camera.
        rates = measured_flag_rates(drift_scores, labels, threshold)
        outcomes.append(
            outcome_from_rates(
                "no_action", delta, threshold, rates, drift_auc, model,
                base_cost=in_control.expected_cost,
            )
        )

        # rate_recenter: same scores, new threshold -- flag rate back to target.
        t_rc = rate_matched_threshold(
            drift_scores[~labels], drift_scores[labels], model.prevalence, target
        )
        rates = measured_flag_rates(drift_scores, labels, t_rc)
        outcomes.append(
            outcome_from_rates(
                "rate_recenter", delta, t_rc, rates, drift_auc, model,
                base_cost=in_control.expected_cost,
            )
        )

        # refit_recent: fresh detector fit on the drifted verified-clean window,
        # then its threshold rate-re-centered the same label-free way.
        refit = refit_factory().fit(perturb(window, "brightness", delta))
        refit_scores = refit.scores(drifted)
        t_rf = rate_matched_threshold(
            refit_scores[~labels], refit_scores[labels], model.prevalence, target
        )
        rates = measured_flag_rates(refit_scores, labels, t_rf)
        outcomes.append(
            outcome_from_rates(
                "refit_recent", delta, t_rf, rates, roc_auc(labels, refit_scores),
                model, base_cost=in_control.expected_cost,
            )
        )

        # fix_camera: root cause repaired -- the in-control point, by construction.
        outcomes.append(
            outcome_from_rates(
                "fix_camera", delta, threshold, base_rates, base_auc, model,
                base_cost=in_control.expected_cost,
            )
        )

    return RecalStudy(
        method=method_name,
        base_threshold=float(threshold),
        target_flag_rate=target,
        model=model,
        config=cfg,
        in_control=in_control,
        outcomes=outcomes,
    )


def recalibration_for_report(
    report,
    sweep=None,
    config: RecalConfig | None = None,
    spc_config: SPCConfig | None = None,
) -> RecalStudy:
    """Run the action-plan study on the method the study recommends, at the
    economics module's cost-optimal threshold, reusing the fitted detector."""
    from qav.economics import economics_for_report

    sweep = sweep or economics_for_report(report)
    name = report.recommendation.method
    detector = report.detectors[name]
    return run_recalibration_study(
        detector,
        sweep.best.threshold,
        name,
        refit_factory=refit_factory_for(detector),
        config=config,
        spc_config=spc_config,
        image_size=report.dataset.config.image_size,
    )


def plain_language_read(study: RecalStudy) -> str:
    """A short read a process engineer can act on -- all numbers from the study."""
    m = study.model
    basis = m.units_basis
    base = study.in_control
    deltas = list(study.config.drift_deltas)
    d_max = deltas[-1]
    lines = [
        f"Out-of-control action plan for the camera-drift alarm ({study.method}, "
        f"base threshold {study.base_threshold:.4f}).",
        f"In control the screen flags {base.flag_rate:.2%} of the stream "
        f"(ROC-AUC {base.roc_auc:.3f}) at {base.expected_cost:,.0f} EUR per "
        f"{basis:,} parts (illustrative rates [A3, A4]).",
    ]
    for delta in deltas:
        lines.append(f"At brightness drift +{delta:g}:")
        for policy in POLICY_ORDER:
            o = study.outcome(policy, delta)
            lines.append(
                f"  - {POLICY_LABELS[policy]}: {o.expected_cost:,.0f} EUR "
                f"({o.d_cost:+,.0f} vs in control), flags {o.flag_rate:.2%}, "
                f"catches {o.caught_defects:.1f} of {basis * m.prevalence:.1f} defects, "
                f"ROC-AUC {o.roc_auc:.3f}."
            )
    rc = study.outcome("rate_recenter", d_max)
    na = study.outcome("no_action", d_max)
    lines.append(
        f"Warning measured at +{d_max:g}: re-centering the threshold returns the flag rate "
        f"to {rc.flag_rate:.2%} -- the control chart goes quiet -- while the screen catches "
        f"{rc.caught_defects:.1f} of {basis * m.prevalence:.1f} defects (ROC-AUC {rc.roc_auc:.3f}, "
        f"identical to doing nothing: a threshold cannot restore lost separability)."
    )
    if na.d_cost > 0:
        rf_share = study.cost_recovered("refit_recent", d_max)
        lines.append(
            f"Re-fitting on {study.config.refit_clean_frames} recent verified-clean frames "
            f"[A10] recovers {rf_share:.0%} of the drift-induced cost at +{d_max:g}; "
            "repairing the camera recovers it all, by construction."
        )
    lines.append(
        "Every rate and AUC is measured on the SPC layer's fresh synthetic calibration "
        "parts; the drift is a synthetic brightness stand-in, the stream composition is "
        "the labelled prevalence [A3], and refit assumes free verified-clean frames "
        "[A10] -- recalibration downtime and verification effort are not costed."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Byte-identical CSV
# --------------------------------------------------------------------------

_CSV_HEADER = (
    "policy,delta,threshold,p_flag_clean,p_flag_defective,roc_auc,flag_rate,"
    "caught_defects_per_basis,missed_defects_per_basis,false_rejects_per_basis,"
    "expected_cost_eur,d_cost_eur"
)


def _csv_row(o: PolicyOutcome) -> str:
    return ",".join(
        [
            o.policy,
            f"{o.delta:.4f}",
            f"{o.threshold:.6f}",
            f"{o.p_clean:.6f}",
            f"{o.p_defective:.6f}",
            f"{o.roc_auc:.6f}",
            f"{o.flag_rate:.6f}",
            f"{o.caught_defects:.3f}",
            f"{o.missed_defects:.3f}",
            f"{o.false_rejects:.3f}",
            f"{o.expected_cost:.2f}",
            f"{o.d_cost:.2f}",
        ]
    )


def recalibration_csv(study: RecalStudy) -> str:
    """The study as CSV text (LF endings): the in-control anchor row first,
    then every (drift level, policy) outcome."""
    rows = [_CSV_HEADER, _csv_row(study.in_control)]
    rows += [_csv_row(o) for o in study.outcomes]
    return "\n".join(rows) + "\n"


def write_recalibration_csv(study: RecalStudy, path: str | Path) -> int:
    text = recalibration_csv(study)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return len(text.encode("utf-8"))


# --------------------------------------------------------------------------
# Hand-drawn (byte-identical) SVG: expected cost vs drift, one line per policy
# --------------------------------------------------------------------------

_SVG_W, _SVG_H = 780, 470
_PLOT = (86.0, 72.0, 610.0, 380.0)  # x0, y0, x1, y1 of the plot rectangle
_COL_AXIS = "#52514e"
_COL_GRID = "#d9d8d4"
_COL_TEXT = "#0b0b0b"
# One fixed hue + dash per policy (dash patterns carry meaning, never hue alone).
_POLICY_STYLE = {
    "no_action": ("#b3261e", "none", 2.6),
    "rate_recenter": ("#eda100", "6 3", 2.0),
    "refit_recent": ("#008300", "8 3 2 3", 2.0),
    "fix_camera": ("#2a78d6", "2 3", 2.0),
}


def recalibration_svg(study: RecalStudy) -> str:
    """Expected cost per basis vs brightness drift, one polyline per policy,
    with the in-control cost as a reference line. Deterministic: no
    timestamps, no random ids, coordinates rounded to two decimals."""
    x0, y0, x1, y1 = _PLOT
    m = study.model
    deltas = list(study.config.drift_deltas)
    d_max = deltas[-1]
    all_costs = [study.in_control.expected_cost] + [o.expected_cost for o in study.outcomes]
    ymax = _nice_ceiling(max(all_costs) * 1.02)

    def px(delta: float) -> float:
        return x0 + (delta / d_max) * (x1 - x0)

    def py(cost: float) -> float:
        return y1 - (cost / ymax) * (y1 - y0)

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_SVG_W}" height="{_SVG_H}" '
        f'viewBox="0 0 {_SVG_W} {_SVG_H}" font-family="Segoe UI, Helvetica, Arial, sans-serif">'
    )
    parts.append(f'<rect x="0" y="0" width="{_SVG_W}" height="{_SVG_H}" fill="white"/>')
    parts.append(
        f'<text x="{_SVG_W / 2:.0f}" y="26" text-anchor="middle" font-size="16" '
        f'font-weight="bold" fill="{_COL_TEXT}">The alarm fired -- now what? '
        "Cost of each response to camera drift</text>"
    )
    parts.append(
        f'<text x="{_SVG_W / 2:.0f}" y="44" text-anchor="middle" font-size="10.5" '
        f'fill="{_COL_AXIS}">{study.method} on fresh calibration parts - illustrative rates: '
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
            f'fill="{_COL_AXIS}">{val:,.0f}</text>'
        )
    # x ticks at every measured drift level (plus zero)
    for delta in [0.0, *deltas]:
        xx = px(delta)
        parts.append(
            f'<line x1="{xx:.2f}" y1="{y1:.2f}" x2="{xx:.2f}" y2="{y1 + 5:.2f}" '
            f'stroke="{_COL_AXIS}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xx:.2f}" y="{y1 + 18:.2f}" text-anchor="middle" font-size="9" '
            f'fill="{_COL_AXIS}">+{delta:g}</text>'
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
        f'fill="{_COL_TEXT}">Brightness drift the camera is running at</text>'
    )
    parts.append(
        f'<text x="22" y="{(y0 + y1) / 2:.0f}" text-anchor="middle" font-size="11" '
        f'fill="{_COL_TEXT}" transform="rotate(-90 22 {(y0 + y1) / 2:.0f})">'
        f"Expected cost (EUR per {m.units_basis:,} parts)</text>"
    )

    # in-control reference line
    base_y = py(study.in_control.expected_cost)
    parts.append(
        f'<line x1="{x0:.2f}" y1="{base_y:.2f}" x2="{x1:.2f}" y2="{base_y:.2f}" '
        f'stroke="{_COL_AXIS}" stroke-width="1" stroke-dasharray="1.5 3.5"/>'
    )
    parts.append(
        f'<text x="{x0 + 4:.2f}" y="{base_y - 5:.2f}" font-size="8.5" '
        f'fill="{_COL_AXIS}">in control: {study.in_control.expected_cost:,.0f} EUR</text>'
    )

    # one polyline + markers per policy; every line starts at the in-control point
    for policy in POLICY_ORDER:
        color, dash, width = _POLICY_STYLE[policy]
        pts = [(0.0, study.in_control.expected_cost)] + [
            (d, study.outcome(policy, d).expected_cost) for d in deltas
        ]
        poly = " ".join(f"{px(d):.2f},{py(c):.2f}" for d, c in pts)
        dash_attr = f' stroke-dasharray="{dash}"' if dash != "none" else ""
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="{width}"'
            f'{dash_attr} points="{poly}"/>'
        )
        for d, c in pts[1:]:
            parts.append(
                f'<circle cx="{px(d):.2f}" cy="{py(c):.2f}" r="3" fill="{color}" '
                f'stroke="white" stroke-width="1"/>'
            )

    # end-of-line labels, staggered deterministically so they never collide
    labels = sorted(
        (
            (study.outcome(policy, d_max).expected_cost, policy)
            for policy in POLICY_ORDER
        ),
        key=lambda t: (py(t[0]), t[1]),
    )
    min_gap = 14.0
    ys: list[float] = []
    for cost, _ in labels:
        yy = py(cost)
        if ys and yy < ys[-1] + min_gap:
            yy = ys[-1] + min_gap
        ys.append(yy)
    for (cost, policy), yy in zip(labels, ys, strict=True):
        color = _POLICY_STYLE[policy][0]
        parts.append(
            f'<text x="{x1 + 8:.2f}" y="{yy + 3.5:.2f}" font-size="9" font-weight="bold" '
            f'fill="{color}">{policy.replace("_", " ")}: {cost:,.0f}</text>'
        )

    parts.append(
        f'<text x="{_SVG_W / 2:.0f}" y="{_SVG_H - 22:.0f}" text-anchor="middle" '
        f'font-size="8.5" fill="{_COL_AXIS}">Measured flag rates and ROC-AUC on fresh '
        "calibration parts; costs from the labelled illustrative model. Full grid: "
        "recalibration.csv / Recalibration sheet.</text>"
    )
    parts.append(
        f'<text x="{_SVG_W / 2:.0f}" y="{_SVG_H - 8:.0f}" text-anchor="middle" '
        f'font-size="8.5" fill="{_COL_AXIS}">Re-centering the threshold quiets the control '
        "chart but cannot restore lost separability -- see the amber line's catch rate in "
        "the CSV.</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_recalibration_svg(study: RecalStudy, path: str | Path) -> int:
    text = recalibration_svg(study)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return len(text.encode("utf-8"))
