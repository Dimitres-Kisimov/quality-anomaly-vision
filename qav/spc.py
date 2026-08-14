"""Statistical process control on the screening output: a p-chart with
Western Electric run rules over a modelled production stream.

The rest of this project answers "which detector?" (evaluation), "at which
threshold?" (economics) and "does it survive a dirty camera?" (robustness) --
all on a *static* test set. A real QA station is a *stream*: parts arrive
shift after shift, and the quantity a process engineer actually watches is the
**share of parts the screen flags per subgroup**. That is a textbook p-chart:
control limits at ``p-bar +/- 3 sqrt(p-bar (1 - p-bar) / n)`` frozen from an
in-control Phase I, then prospective Phase II monitoring with the Western
Electric run rules (one point beyond 3 sigma; 2 of 3 beyond 2 sigma on one
side; 4 of 5 beyond 1 sigma on one side; 8 consecutive on one side of the
center line).

Two Phase II scenarios are simulated against the same frozen limits:

- ``process_shift``  the true defect rate steps up (upstream tooling wear)
  while the camera stays perfect -- the chart's core purpose, and small enough
  that the run rules matter, not just the 3-sigma rule.
- ``camera_drift``   the true defect rate NEVER changes, but the camera's
  brightness drifts slowly upward (lamp aging / exposure creep). The chart
  alarms anyway, because a p-chart on detector output monitors the *process
  and the measurement system together* -- an out-of-control signal says
  "investigate", not "the process shifted". This ties the robustness finding
  (the recommended PCA screen is the method most fragile to illumination
  drift) to the monitoring layer that would actually surface it.

What is measured vs what is modelled, stated plainly:

- **Measured:** the per-class flag probabilities. A fresh synthetic
  *calibration set* (clean and defective parts the study never used) is scored
  by the **already-fitted** recommended detector at the economics module's
  cost-optimal threshold; the drift scenario re-scores the same calibration
  images under each brightness delta (via :func:`qav.robustness.perturb`) --
  no re-fitting anywhere. Calibration at this resolution is honest in a way
  the 150-clean-image test set cannot be: rates below ~0.7% are invisible
  there.
- **Modelled:** the stream itself. Parts are i.i.d. Bernoulli draws (defective
  with the labelled prevalence [A3]; flagged with the measured per-class
  probability), seeded and deterministic. Real lines autocorrelate and drift
  in ways i.i.d. draws do not; the chart mechanics are exact, the stream is a
  modelled stand-in. Subgroup size is one shift's production under [A1].
- The Western Electric rules assume roughly normal subgroup proportions; at
  low p-bar the zones are chunky (counts move in steps of 1/n), and the
  combined rules' in-control false-alarm rate is materially higher than the
  3-sigma rule alone (theoretical ARL ~92 vs ~370 subgroups). Alarms before
  the injected change are reported, never hidden.

Pure numpy plus the standard library on top of already-fitted detectors: no
wall-clock, no plotting dependency. The CSV and hand-drawn SVG are
byte-identical across re-runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qav.data import DataConfig, SurfaceDataset, make_dataset
from qav.economics import DEFECT_PREVALENCE, _nice_ceiling
from qav.palette import GRID, INK, INK_MUTED, LAMP, SIGNAL, STEEL, SURFACE
from qav.robustness import perturb

# --- Stream and chart defaults (labelled; see docstring and BUSINESS_CASE.md) ---
SUBGROUP_SIZE = 2000  # one shift's production under [A1] (4,000 parts/day, two shifts)
PHASE1_SUBGROUPS = 30  # in-control shifts used to freeze the control limits
PHASE2_SUBGROUPS = 25  # prospectively monitored shifts per scenario
CHANGE_AT = 10  # phase-2 index where each scenario's change begins
SHIFTED_PREVALENCE = 0.025  # process-shift scenario: defect rate 1.5% -> 2.5%
DRIFT_MAX_BRIGHTNESS = 0.02  # camera-drift scenario: brightness ramps 0 -> +0.02
# Stream-simulation seed (phase 1; scenarios use seed+1, seed+2). Fixed to a
# value whose modelled Phase I passes its own in-control check (no rule signals
# in the limits-setting window) -- the premise p-chart practice requires before
# freezing limits. Alarms are counted and reported wherever they occur, so a
# different seed changes the narrative, never the bookkeeping.
SPC_SEED = 20920
CALIBRATION_CLEAN = 1000  # fresh clean parts used to measure the flag rates
CALIBRATION_DEFECTIVE = 600  # fresh defective parts (split across defect kinds)
CALIBRATION_SEED = 1723  # distinct from the study seed: genuinely unseen parts

RULE_NAMES = (
    "beyond_3sigma",
    "two_of_three_beyond_2sigma",
    "four_of_five_beyond_1sigma",
    "eight_same_side",
)
RULE_LABELS = {
    "beyond_3sigma": "beyond 3 sigma",
    "two_of_three_beyond_2sigma": "2 of 3 beyond 2 sigma",
    "four_of_five_beyond_1sigma": "4 of 5 beyond 1 sigma",
    "eight_same_side": "8 on one side of CL",
}


@dataclass(frozen=True)
class SPCConfig:
    """Stream shape, scenario magnitudes and seeds. Tests use tiny values."""

    subgroup_size: int = SUBGROUP_SIZE
    phase1_subgroups: int = PHASE1_SUBGROUPS
    phase2_subgroups: int = PHASE2_SUBGROUPS
    change_at: int = CHANGE_AT
    in_control_prevalence: float = DEFECT_PREVALENCE  # [A3]
    shifted_prevalence: float = SHIFTED_PREVALENCE
    drift_max_brightness: float = DRIFT_MAX_BRIGHTNESS
    seed: int = SPC_SEED
    calibration_clean: int = CALIBRATION_CLEAN
    calibration_defective: int = CALIBRATION_DEFECTIVE
    calibration_seed: int = CALIBRATION_SEED


@dataclass(frozen=True)
class FlagRates:
    """Measured per-class flag probabilities at one threshold (with sample sizes)."""

    p_clean: float
    p_defective: float
    n_clean: int
    n_defective: int


@dataclass(frozen=True)
class PChartLimits:
    """Center line and sigma of a p-chart; 3-sigma limits clipped into [0, 1]."""

    center: float
    sigma: float
    subgroup_size: int

    @property
    def ucl(self) -> float:
        return min(self.center + 3.0 * self.sigma, 1.0)

    @property
    def lcl(self) -> float:
        return max(self.center - 3.0 * self.sigma, 0.0)

    def zone(self, k: float) -> tuple[float, float]:
        """(low, high) boundaries at ``center +/- k sigma``, clipped into [0, 1]."""
        return (max(self.center - k * self.sigma, 0.0), min(self.center + k * self.sigma, 1.0))


@dataclass(frozen=True)
class RuleHits:
    """Which Western Electric rules fire at one subgroup."""

    beyond_3sigma: bool = False
    two_of_three_beyond_2sigma: bool = False
    four_of_five_beyond_1sigma: bool = False
    eight_same_side: bool = False

    @property
    def alarm(self) -> bool:
        return (
            self.beyond_3sigma
            or self.two_of_three_beyond_2sigma
            or self.four_of_five_beyond_1sigma
            or self.eight_same_side
        )


@dataclass(frozen=True)
class SubgroupCondition:
    """The stream's (latent) state during one subgroup."""

    prevalence: float
    rates: FlagRates
    brightness_delta: float = 0.0


@dataclass(frozen=True)
class SubgroupPoint:
    """One charted subgroup: the modelled draw plus the condition that produced it."""

    index: int  # global subgroup index; Phase II starts at phase1_subgroups
    phase: int  # 1 = limits-setting, 2 = prospective monitoring
    n: int
    true_prevalence: float
    brightness_delta: float
    p_flag_clean: float
    p_flag_defective: float
    defective_drawn: int  # latent truth of the modelled stream (auditability)
    flags: int
    proportion: float


@dataclass(frozen=True)
class ScenarioResult:
    """One Phase II scenario charted against the frozen Phase I limits."""

    name: str
    label: str
    points: list[SubgroupPoint]  # full sequence: shared phase 1 + this scenario's phase 2
    rules: list[RuleHits]  # aligned with ``points``
    limits: PChartLimits
    change_index: int  # global index of the first changed subgroup
    phase1_alarm_count: int  # rule hits inside phase 1 (should be 0 for usable limits)
    false_alarms: int  # phase-2 alarms strictly before the change
    first_alarm: int | None  # first any-rule alarm at/after the change
    first_by_rule: dict[str, int | None]

    @property
    def detection_delay(self) -> int | None:
        """Subgroups from the change to the first alarm (0 = caught immediately)."""
        return None if self.first_alarm is None else self.first_alarm - self.change_index


@dataclass(frozen=True)
class SPCStudy:
    """The full monitoring study: measured rates, frozen limits, two scenarios."""

    method: str
    threshold: float
    config: SPCConfig
    baseline_rates: FlagRates
    drift_rates: list[tuple[float, FlagRates]]  # (brightness delta, measured rates) per step
    limits: PChartLimits
    phase1: list[SubgroupPoint]
    scenarios: list[ScenarioResult]


# --------------------------------------------------------------------------
# Chart machinery (pure, hand-checkable)
# --------------------------------------------------------------------------


def p_chart_limits(p_bar: float, subgroup_size: int) -> PChartLimits:
    """Standard p-chart limits from an in-control proportion and subgroup size."""
    if not 0.0 < p_bar < 1.0:
        raise ValueError(f"p_bar must be strictly inside (0, 1), got {p_bar}")
    if subgroup_size < 1:
        raise ValueError(f"subgroup_size must be >= 1, got {subgroup_size}")
    sigma = float(np.sqrt(p_bar * (1.0 - p_bar) / subgroup_size))
    return PChartLimits(center=float(p_bar), sigma=sigma, subgroup_size=subgroup_size)


def limits_from_counts(flag_counts: list[int], subgroup_size: int) -> PChartLimits:
    """Freeze limits from Phase I flag counts (``p_bar`` = total flags / total parts)."""
    if not flag_counts:
        raise ValueError("limits_from_counts needs at least one Phase I subgroup")
    total = len(flag_counts) * subgroup_size
    p_bar = sum(flag_counts) / total
    if p_bar <= 0.0:
        raise ValueError(
            "Phase I recorded no flags at all; a p-chart needs a nonzero in-control "
            "flag rate (lower the threshold, raise the subgroup size, or collect more "
            "Phase I subgroups)"
        )
    if p_bar >= 1.0:
        raise ValueError("Phase I flagged every part; the screen is not discriminating")
    return p_chart_limits(p_bar, subgroup_size)


def western_electric_flags(proportions, limits: PChartLimits) -> list[RuleHits]:
    """Evaluate the four Western Electric rules at every point of a sequence.

    Conventions (fixed and tested): "beyond" is strict (a point exactly on a
    boundary is not beyond it); a point exactly on the center line is on
    neither side and so breaks a same-side run; rules 2 and 3 require the
    *signalling point itself* to be beyond the zone line, with the count taken
    over up to the last 3 (respectively 5) points, so an alarm is always
    attributed to the subgroup that completes the pattern.
    """
    props = [float(p) for p in proportions]
    c = limits.center
    u1, u2, u3 = (c + k * limits.sigma for k in (1.0, 2.0, 3.0))
    l1, l2, l3 = (c - k * limits.sigma for k in (1.0, 2.0, 3.0))

    def _kofn(i: int, window: int, need: int, bound: float, high: bool) -> bool:
        recent = props[max(0, i - window + 1) : i + 1]
        beyond = [(q > bound) if high else (q < bound) for q in recent]
        return beyond[-1] and sum(beyond) >= need

    hits: list[RuleHits] = []
    for i, p in enumerate(props):
        rule1 = p > u3 or p < l3
        rule2 = _kofn(i, 3, 2, u2, True) or _kofn(i, 3, 2, l2, False)
        rule3 = _kofn(i, 5, 4, u1, True) or _kofn(i, 5, 4, l1, False)
        rule4 = False
        if i >= 7:
            last8 = props[i - 7 : i + 1]
            rule4 = all(q > c for q in last8) or all(q < c for q in last8)
        hits.append(
            RuleHits(
                beyond_3sigma=rule1,
                two_of_three_beyond_2sigma=rule2,
                four_of_five_beyond_1sigma=rule3,
                eight_same_side=rule4,
            )
        )
    return hits


def simulate_subgroups(
    conditions: list[SubgroupCondition], subgroup_size: int, seed: int
) -> list[tuple[int, int]]:
    """Draw ``(defective_drawn, flags)`` per subgroup, deterministically.

    Two-stage binomial per subgroup: how many parts are truly defective (at
    the condition's prevalence), then how many of each class the screen flags
    (at the measured per-class rates). One ``numpy`` generator, consumed
    sequentially -- the same seed always reproduces the same stream.
    """
    rng = np.random.default_rng(seed)
    draws: list[tuple[int, int]] = []
    for cond in conditions:
        n_def = int(rng.binomial(subgroup_size, cond.prevalence))
        flags = int(
            rng.binomial(n_def, cond.rates.p_defective)
            + rng.binomial(subgroup_size - n_def, cond.rates.p_clean)
        )
        draws.append((n_def, flags))
    return draws


# --------------------------------------------------------------------------
# Measured ingredients
# --------------------------------------------------------------------------


def measured_flag_rates(scores: np.ndarray, labels: np.ndarray, threshold: float) -> FlagRates:
    """Empirical P(flag | clean) and P(flag | defective) at ``score >= threshold``."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_clean = int((~labels).sum())
    n_def = int(labels.sum())
    if n_clean == 0 or n_def == 0:
        raise ValueError("measured_flag_rates needs at least one clean and one defective part")
    flagged = scores >= threshold
    return FlagRates(
        p_clean=float(flagged[~labels].mean()),
        p_defective=float(flagged[labels].mean()),
        n_clean=n_clean,
        n_defective=n_def,
    )


def calibration_dataset(config: SPCConfig, image_size: int = 64) -> SurfaceDataset:
    """Fresh synthetic parts (own seed) for measuring the stream's flag rates.

    Reuses the study's generator via its public API; ``n_train=1`` because the
    generator requires at least one training image, which is simply unused.
    """
    return make_dataset(
        DataConfig(
            n_train=1,
            n_test_clean=config.calibration_clean,
            n_test_defective=config.calibration_defective,
            image_size=image_size,
            seed=config.calibration_seed,
        )
    )


def drift_deltas(config: SPCConfig) -> list[float]:
    """The camera-drift scenario's brightness delta per changed subgroup (linear ramp)."""
    steps = config.phase2_subgroups - config.change_at
    if steps < 1:
        raise ValueError("change_at must leave at least one changed phase-2 subgroup")
    return [config.drift_max_brightness * (j + 1) / steps for j in range(steps)]


# --------------------------------------------------------------------------
# The study
# --------------------------------------------------------------------------


def _run_scenario(
    name: str,
    label: str,
    phase1_points: list[SubgroupPoint],
    phase2_conditions: list[SubgroupCondition],
    limits: PChartLimits,
    config: SPCConfig,
    seed: int,
) -> ScenarioResult:
    n = config.subgroup_size
    offset = len(phase1_points)
    draws = simulate_subgroups(phase2_conditions, n, seed)
    phase2_points = [
        SubgroupPoint(
            index=offset + i,
            phase=2,
            n=n,
            true_prevalence=cond.prevalence,
            brightness_delta=cond.brightness_delta,
            p_flag_clean=cond.rates.p_clean,
            p_flag_defective=cond.rates.p_defective,
            defective_drawn=n_def,
            flags=flags,
            proportion=flags / n,
        )
        for i, (cond, (n_def, flags)) in enumerate(zip(phase2_conditions, draws, strict=True))
    ]
    points = phase1_points + phase2_points
    rules = western_electric_flags([p.proportion for p in points], limits)
    change_index = offset + config.change_at

    phase1_alarm_count = sum(r.alarm for r in rules[:offset])
    false_alarms = sum(r.alarm for r in rules[offset:change_index])
    first_alarm = next((i for i in range(change_index, len(rules)) if rules[i].alarm), None)
    first_by_rule = {
        rule: next(
            (i for i in range(change_index, len(rules)) if getattr(rules[i], rule)), None
        )
        for rule in RULE_NAMES
    }
    return ScenarioResult(
        name=name,
        label=label,
        points=points,
        rules=rules,
        limits=limits,
        change_index=change_index,
        phase1_alarm_count=phase1_alarm_count,
        false_alarms=false_alarms,
        first_alarm=first_alarm,
        first_by_rule=first_by_rule,
    )


def run_spc_study(
    detector,
    threshold: float,
    method_name: str,
    config: SPCConfig | None = None,
    image_size: int = 64,
) -> SPCStudy:
    """Measure the flag rates with ``detector`` (already fitted), then simulate
    and chart the two Phase II scenarios against frozen Phase I limits.

    ``detector`` only needs a ``scores(images)`` method and is never re-fit.
    """
    cfg = config or SPCConfig()
    if not 0 <= cfg.change_at < cfg.phase2_subgroups:
        raise ValueError("change_at must lie inside phase 2")

    cal = calibration_dataset(cfg, image_size)
    labels = cal.test_labels
    baseline = measured_flag_rates(detector.scores(cal.test_images), labels, threshold)

    drift_rates: list[tuple[float, FlagRates]] = []
    for delta in drift_deltas(cfg):
        scores = detector.scores(perturb(cal.test_images, "brightness", delta))
        drift_rates.append((delta, measured_flag_rates(scores, labels, threshold)))

    # Phase 1: in-control stream; limits frozen from its pooled flag rate.
    base_cond = SubgroupCondition(cfg.in_control_prevalence, baseline, 0.0)
    n = cfg.subgroup_size
    p1_draws = simulate_subgroups([base_cond] * cfg.phase1_subgroups, n, cfg.seed)
    limits = limits_from_counts([flags for _, flags in p1_draws], n)
    phase1_points = [
        SubgroupPoint(
            index=i,
            phase=1,
            n=n,
            true_prevalence=base_cond.prevalence,
            brightness_delta=0.0,
            p_flag_clean=baseline.p_clean,
            p_flag_defective=baseline.p_defective,
            defective_drawn=n_def,
            flags=flags,
            proportion=flags / n,
        )
        for i, (n_def, flags) in enumerate(p1_draws)
    ]

    change_global = cfg.phase1_subgroups + cfg.change_at
    shift_cond = SubgroupCondition(cfg.shifted_prevalence, baseline, 0.0)
    shift_conditions = [base_cond] * cfg.change_at + [shift_cond] * (
        cfg.phase2_subgroups - cfg.change_at
    )
    drift_conditions = [base_cond] * cfg.change_at + [
        SubgroupCondition(cfg.in_control_prevalence, rates, delta)
        for delta, rates in drift_rates
    ]

    scenarios = [
        _run_scenario(
            "process_shift",
            f"true defect rate {cfg.in_control_prevalence:.1%} -> "
            f"{cfg.shifted_prevalence:.1%} at subgroup {change_global}",
            phase1_points,
            shift_conditions,
            limits,
            cfg,
            cfg.seed + 1,
        ),
        _run_scenario(
            "camera_drift",
            f"brightness drifts 0 -> +{cfg.drift_max_brightness:g} from subgroup "
            f"{change_global}; true defect rate unchanged",
            phase1_points,
            drift_conditions,
            limits,
            cfg,
            cfg.seed + 2,
        ),
    ]
    return SPCStudy(
        method=method_name,
        threshold=float(threshold),
        config=cfg,
        baseline_rates=baseline,
        drift_rates=drift_rates,
        limits=limits,
        phase1=phase1_points,
        scenarios=scenarios,
    )


def spc_for_report(report, sweep=None, config: SPCConfig | None = None) -> SPCStudy:
    """Run the monitoring study on the method the study recommends, at the
    economics module's cost-optimal threshold, reusing the fitted detector."""
    from qav.economics import economics_for_report

    sweep = sweep or economics_for_report(report)
    name = report.recommendation.method
    detector = report.detectors[name]
    return run_spc_study(
        detector,
        sweep.best.threshold,
        name,
        config,
        image_size=report.dataset.config.image_size,
    )


def plain_language_read(study: SPCStudy) -> str:
    """A short read a process engineer can act on -- all numbers from the study."""
    cfg = study.config
    lim = study.limits
    base = study.baseline_rates
    n = cfg.subgroup_size
    lines = [
        f"p-chart on the screening output ({study.method}, flag when score >= "
        f"{study.threshold:.4f}); one subgroup = {n:,} parts (one shift under [A1]).",
        f"Measured on {base.n_clean:,} fresh clean + {base.n_defective:,} fresh defective "
        f"calibration parts: the screen flags {base.p_clean:.2%} of clean and "
        f"{base.p_defective:.1%} of defective parts at this threshold.",
        f"Phase I ({cfg.phase1_subgroups} in-control subgroups, modelled at "
        f"{cfg.in_control_prevalence:.1%} defect prevalence [A3]): center line "
        f"{lim.center:.2%}, UCL {lim.ucl:.2%} (= {lim.ucl * n:.0f} flags), "
        f"LCL {lim.lcl:.2%}.",
    ]
    for sc in study.scenarios:
        if sc.first_alarm is None:
            outcome = "no alarm within the monitored window"
        else:
            firing = [RULE_LABELS[r] for r, i in sc.first_by_rule.items() if i == sc.first_alarm]
            outcome = (
                f"first alarm at subgroup {sc.first_alarm} ({', '.join(firing)}), "
                f"{sc.detection_delay} subgroup(s) after the change"
            )
        lines.append(
            f"  - {sc.name}: {sc.label} -> {outcome}; "
            f"{sc.false_alarms} false alarm(s) in the {cfg.change_at} monitored "
            f"subgroups before the change."
        )
    shift = next(s for s in study.scenarios if s.name == "process_shift")
    if shift.detection_delay is not None and shift.detection_delay > 0:
        extra_escapes = (
            n * (cfg.shifted_prevalence - cfg.in_control_prevalence) * (1.0 - base.p_defective)
        )
        lines.append(
            f"While the shift ran undetected, each subgroup shipped ~{extra_escapes:.0f} extra "
            f"escaped defects (~{extra_escapes * 35.0:,.0f} EUR at the illustrative 35 EUR [A4])."
        )
    lines.append(
        "Reading: the camera-drift scenario alarms although the true defect rate never moved -- "
        "a p-chart on detector output monitors process and measurement system together, so an "
        "out-of-control signal means 'investigate', not 'the process shifted'. The stream is "
        "modelled (i.i.d. seeded draws from measured per-class rates on synthetic calibration "
        "parts), not measured from a real line."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Byte-identical CSV
# --------------------------------------------------------------------------

_CSV_HEADER = (
    "scenario,subgroup,phase,n,true_prevalence,brightness_delta,p_flag_clean,"
    "p_flag_defective,defective_drawn,flags,proportion,center,ucl,lcl,"
    "rule_beyond_3sigma,rule_2of3_beyond_2sigma,rule_4of5_beyond_1sigma,"
    "rule_8_same_side,any_alarm,is_change_point"
)


def _csv_row(
    scenario: str, pt: SubgroupPoint, hits: RuleHits, lim: PChartLimits, change: bool
) -> str:
    return ",".join(
        [
            scenario,
            str(pt.index),
            str(pt.phase),
            str(pt.n),
            f"{pt.true_prevalence:.6f}",
            f"{pt.brightness_delta:.6f}",
            f"{pt.p_flag_clean:.6f}",
            f"{pt.p_flag_defective:.6f}",
            str(pt.defective_drawn),
            str(pt.flags),
            f"{pt.proportion:.6f}",
            f"{lim.center:.6f}",
            f"{lim.ucl:.6f}",
            f"{lim.lcl:.6f}",
            "1" if hits.beyond_3sigma else "0",
            "1" if hits.two_of_three_beyond_2sigma else "0",
            "1" if hits.four_of_five_beyond_1sigma else "0",
            "1" if hits.eight_same_side else "0",
            "1" if hits.alarm else "0",
            "1" if change else "0",
        ]
    )


def chart_rows(study: SPCStudy) -> list[tuple[str, SubgroupPoint, RuleHits, bool]]:
    """Flatten the study for tabular outputs: the shared Phase I block once
    (scenario ``baseline``), then each scenario's Phase II subgroups. The last
    element marks the scenario's change point."""
    offset = len(study.phase1)
    rows: list[tuple[str, SubgroupPoint, RuleHits, bool]] = []
    phase1_rules = study.scenarios[0].rules[:offset] if study.scenarios else []
    for pt, hits in zip(study.phase1, phase1_rules, strict=True):
        rows.append(("baseline", pt, hits, False))
    for sc in study.scenarios:
        for pt, hits in zip(sc.points[offset:], sc.rules[offset:], strict=True):
            rows.append((sc.name, pt, hits, pt.index == sc.change_index))
    return rows


def spc_csv(study: SPCStudy) -> str:
    """The study as CSV text (LF endings), one row per charted subgroup."""
    lim = study.limits
    rows = [_CSV_HEADER]
    for scenario, pt, hits, change in chart_rows(study):
        rows.append(_csv_row(scenario, pt, hits, lim, change))
    return "\n".join(rows) + "\n"


def write_spc_csv(study: SPCStudy, path: str | Path) -> int:
    text = spc_csv(study)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return len(text.encode("utf-8"))


# --------------------------------------------------------------------------
# Hand-drawn (byte-identical) SVG control chart
# --------------------------------------------------------------------------

_SVG_W, _SVG_H = 820, 700
_PANELS = ((70.0, 108.0, 756.0, 330.0), (70.0, 402.0, 756.0, 624.0))  # x0, y0, x1, y1
# Plate palette (see qav/palette.py): the monitored stream wears the machined
# steel of the part it watches, control limits and rule violations the signal
# red, the change point the inspection-lamp amber. Dash patterns and marker
# shapes repeat every distinction, so hue never carries meaning alone.
_COL_SERIES = STEEL
_COL_STATUS = SIGNAL
_COL_CHANGE = LAMP
_COL_AXIS = INK_MUTED
_COL_GRID = GRID
_COL_TEXT = INK


def _panel(
    parts: list[str], sc: ScenarioResult, rect, ymax: float, y_step: float, title: str
) -> None:
    x0, y0, x1, y1 = rect
    lim = sc.limits
    n_pts = len(sc.points)

    def px(i: float) -> float:
        return x0 + (i / max(n_pts - 1, 1)) * (x1 - x0)

    def py(v: float) -> float:
        return y1 - (v / ymax) * (y1 - y0)

    parts.append(
        f'<text x="{x0:.2f}" y="{y0 - 12:.2f}" font-size="11.5" font-weight="bold" '
        f'fill="{_COL_TEXT}">{title}</text>'
    )
    # y grid + tick labels (one gridline per step; data-tight scale)
    tick_fmt = ".0%" if y_step >= 0.0099 else ".1%"
    for i in range(round(ymax / y_step) + 1):
        val = y_step * i
        yy = py(val)
        parts.append(
            f'<line x1="{x0:.2f}" y1="{yy:.2f}" x2="{x1:.2f}" y2="{yy:.2f}" '
            f'stroke="{_COL_GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x0 - 8:.2f}" y="{yy + 3.5:.2f}" text-anchor="end" font-size="9" '
            f'fill="{_COL_AXIS}">{val:{tick_fmt}}</text>'
        )
    # x ticks every 5 subgroups
    for i in range(0, n_pts, 5):
        xx = px(i)
        parts.append(
            f'<line x1="{xx:.2f}" y1="{y1:.2f}" x2="{xx:.2f}" y2="{y1 + 5:.2f}" '
            f'stroke="{_COL_AXIS}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xx:.2f}" y="{y1 + 17:.2f}" text-anchor="middle" font-size="9" '
            f'fill="{_COL_AXIS}">{i}</text>'
        )
    # zone lines (1 and 2 sigma, both sides, clipped to the panel)
    for k in (1.0, 2.0):
        for bound in sc.limits.zone(k):
            if 0.0 < bound < ymax:
                yy = py(bound)
                parts.append(
                    f'<line x1="{x0:.2f}" y1="{yy:.2f}" x2="{x1:.2f}" y2="{yy:.2f}" '
                    f'stroke="{_COL_AXIS}" stroke-width="0.8" stroke-dasharray="1.5 3.5" '
                    f'opacity="0.55"/>'
                )
    # center line + 3-sigma limits with direct labels
    yy = py(lim.center)
    parts.append(
        f'<line x1="{x0:.2f}" y1="{yy:.2f}" x2="{x1:.2f}" y2="{yy:.2f}" '
        f'stroke="{_COL_AXIS}" stroke-width="1.4"/>'
    )
    parts.append(
        f'<text x="{x1 + 6:.2f}" y="{yy + 3.5:.2f}" font-size="9" '
        f'fill="{_COL_AXIS}">CL {lim.center:.2%}</text>'
    )
    for bound, name in ((lim.ucl, "UCL"), (lim.lcl, "LCL")):
        if 0.0 < bound < ymax:
            yy = py(bound)
            parts.append(
                f'<line x1="{x0:.2f}" y1="{yy:.2f}" x2="{x1:.2f}" y2="{yy:.2f}" '
                f'stroke="{_COL_STATUS}" stroke-width="1.4" stroke-dasharray="6 3"/>'
            )
            parts.append(
                f'<text x="{x1 + 6:.2f}" y="{yy + 3.5:.2f}" font-size="9" '
                f'fill="{_COL_STATUS}">{name} {bound:.2%}</text>'
            )
    # phase divider + phase captions
    phase1_len = sum(1 for p in sc.points if p.phase == 1)
    xx = px(phase1_len - 0.5)
    parts.append(
        f'<line x1="{xx:.2f}" y1="{y0:.2f}" x2="{xx:.2f}" y2="{y1:.2f}" '
        f'stroke="{_COL_GRID}" stroke-width="1.2"/>'
    )
    parts.append(
        f'<text x="{px(phase1_len / 2 - 0.5):.2f}" y="{y0 + 12:.2f}" text-anchor="middle" '
        f'font-size="8.5" fill="{_COL_AXIS}">Phase I (limits frozen here)</text>'
    )
    parts.append(
        f'<text x="{px(phase1_len + (n_pts - phase1_len) / 2 - 0.5):.2f}" y="{y0 + 12:.2f}" '
        f'text-anchor="middle" font-size="8.5" fill="{_COL_AXIS}">Phase II (monitoring)</text>'
    )
    # change point marker with a text label (never color alone)
    xx = px(sc.change_index - 0.5)
    parts.append(
        f'<line x1="{xx:.2f}" y1="{y0:.2f}" x2="{xx:.2f}" y2="{y1:.2f}" '
        f'stroke="{_COL_CHANGE}" stroke-width="1.6" stroke-dasharray="4 3"/>'
    )
    parts.append(
        f'<text x="{xx + 4:.2f}" y="{y1 - 8:.2f}" font-size="8.5" '
        f'fill="{_COL_TEXT}">change begins</text>'
    )
    # the series itself, then alarm markers on top (filled, ringed, not color-alone)
    poly = " ".join(f"{px(p.index):.2f},{py(p.proportion):.2f}" for p in sc.points)
    parts.append(
        f'<polyline fill="none" stroke="{_COL_SERIES}" stroke-width="1.8" points="{poly}"/>'
    )
    for p, hits in zip(sc.points, sc.rules, strict=True):
        cx, cy = px(p.index), py(p.proportion)
        if hits.alarm:
            parts.append(
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="4" fill="{_COL_STATUS}" '
                f'stroke="{SURFACE}" stroke-width="1.4"/>'
            )
        else:
            parts.append(
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="2.4" fill="{SURFACE}" '
                f'stroke="{_COL_SERIES}" stroke-width="1.2"/>'
            )
    # first-alarm annotation, pinned to a clear band with a dotted leader down
    if sc.first_alarm is not None:
        pt = sc.points[sc.first_alarm]
        firing = [
            RULE_LABELS[r] for r, i in sc.first_by_rule.items() if i == sc.first_alarm
        ]
        cx, cy = px(pt.index), py(pt.proportion)
        ann_y = y0 + 34.0
        anchor = "end" if cx > (x0 + x1) / 2 else "start"
        tx = cx - 8 if anchor == "end" else cx + 8
        parts.append(
            f'<line x1="{cx:.2f}" y1="{ann_y + 4:.2f}" x2="{cx:.2f}" y2="{cy - 6:.2f}" '
            f'stroke="{_COL_STATUS}" stroke-width="0.8" stroke-dasharray="2 3" opacity="0.6"/>'
        )
        parts.append(
            f'<text x="{tx:.2f}" y="{ann_y:.2f}" text-anchor="{anchor}" font-size="9" '
            f'font-weight="bold" fill="{_COL_STATUS}">first alarm: subgroup {pt.index} '
            f"({', '.join(firing)})</text>"
        )


def spc_chart_svg(study: SPCStudy) -> str:
    """Both scenarios as p-chart panels on a shared scale, drawn by hand.

    Deterministic: no timestamps, no random ids, coordinates rounded to two
    decimals. Alarm points are larger, filled and surface-ringed (redundant to
    hue); control limits are dashed; the change point carries a text label.
    """
    raw_max = (
        max(
            max(p.proportion for sc in study.scenarios for p in sc.points),
            study.limits.ucl,
        )
        * 1.05
    )
    y_step = _nice_ceiling(raw_max / 6.0)
    ymax = y_step * math.ceil(raw_max / y_step)
    n = study.config.subgroup_size
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_SVG_W}" height="{_SVG_H}" '
        f'viewBox="0 0 {_SVG_W} {_SVG_H}" font-family="Segoe UI, Helvetica, Arial, sans-serif">'
    )
    parts.append(f'<rect x="0" y="0" width="{_SVG_W}" height="{_SVG_H}" fill="{SURFACE}"/>')
    parts.append(
        f'<text x="{_SVG_W / 2:.0f}" y="26" text-anchor="middle" font-size="16" '
        f'font-weight="bold" fill="{_COL_TEXT}">Is the line still in control? '
        "p-chart on the screening output</text>"
    )
    parts.append(
        f'<text x="{_SVG_W / 2:.0f}" y="44" text-anchor="middle" font-size="10.5" '
        f'fill="{_COL_AXIS}">{study.method} flags score >= {study.threshold:.4f}; '
        f"one subgroup = {n:,} parts (one shift); Western Electric rules on frozen "
        "Phase I limits; modelled stream</text>"
    )
    # legend (line styles and marker shapes carry meaning, not hue alone)
    legend = [
        ("line", _COL_SERIES, "none", "share of parts flagged"),
        ("line", _COL_STATUS, "6 3", "3-sigma control limits"),
        ("line", _COL_AXIS, "1.5 3.5", "1- / 2-sigma zones"),
        ("dot", _COL_STATUS, "", "rule violation"),
        ("line", _COL_CHANGE, "4 3", "change point"),
    ]
    lx = 78.0
    for kind, color, dash, text in legend:
        if kind == "line":
            dash_attr = f' stroke-dasharray="{dash}"' if dash != "none" else ""
            parts.append(
                f'<line x1="{lx:.1f}" y1="62" x2="{lx + 20:.1f}" y2="62" '
                f'stroke="{color}" stroke-width="2"{dash_attr}/>'
            )
        else:
            parts.append(
                f'<circle cx="{lx + 10:.1f}" cy="62" r="4" fill="{color}" '
                f'stroke="{SURFACE}" stroke-width="1.2"/>'
            )
        parts.append(
            f'<text x="{lx + 26:.1f}" y="65.5" font-size="9" fill="{_COL_TEXT}">{text}</text>'
        )
        lx += 40.0 + 6.2 * len(text)

    titles = {
        "process_shift": "A - process shift: ",
        "camera_drift": "B - camera drift: ",
    }
    for sc, rect in zip(study.scenarios, _PANELS, strict=True):
        _panel(parts, sc, rect, ymax, y_step, titles.get(sc.name, "") + sc.label)

    parts.append(
        f'<text x="{(_PANELS[1][0] + _PANELS[1][2]) / 2:.0f}" y="{_SVG_H - 28:.0f}" '
        f'text-anchor="middle" font-size="11" fill="{_COL_TEXT}">subgroup '
        f"(one shift = {n:,} parts)</text>"
    )
    parts.append(
        f'<text x="20" y="{_SVG_H / 2:.0f}" text-anchor="middle" font-size="11" '
        f'fill="{_COL_TEXT}" transform="rotate(-90 20 {_SVG_H / 2:.0f})">share of parts '
        "flagged</text>"
    )
    parts.append(
        f'<text x="{_SVG_W / 2:.0f}" y="{_SVG_H - 8:.0f}" text-anchor="middle" font-size="8.5" '
        f'fill="{_COL_AXIS}">Panel B: the true defect rate never changes -- the chart '
        "alarms on camera drift alone. Full per-subgroup data: spc_chart.csv / SPC sheet."
        "</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_spc_svg(study: SPCStudy, path: str | Path) -> int:
    text = spc_chart_svg(study)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return len(text.encode("utf-8"))
