"""Defect-cost / inspection-threshold economics: hand-checked sweep math, the
cost-minimising recommendation, and byte-identical CSV/SVG outputs.

None of these tests need torch: the sweep works on any (labels, scores) pair
and the deliverable-writing test uses the torch-free PCA-only report.
"""

import math

import numpy as np
import pytest

from qav.data import DataConfig
from qav.economics import (
    CostModel,
    cost_curve_csv,
    economics_for_report,
    plain_language_read,
    sweep_thresholds,
    write_cost_curve_csv,
    write_cost_curve_svg,
)
from qav.evaluate import run_full_evaluation

# A basis equal to the sample count with prevalence = positives/total makes the
# per-basis counts equal the raw confusion counts, so the cost math is
# hand-checkable directly against the 2-positive / 2-negative toy example.
TOY_MODEL = CostModel(prevalence=0.5, cost_escaped=10.0, cost_false_reject=1.0, units_basis=4)


def test_sweep_math_matches_hand_computed_costs():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    sweep = sweep_thresholds(labels, scores, TOY_MODEL, method_name="toy")

    # Endpoints. Flag nothing -> both defects (worth 10 each) escape, no scrap.
    assert sweep.flag_none.tpr == 0.0 and sweep.flag_none.fpr == 0.0
    assert sweep.flag_none.expected_cost == pytest.approx(20.0)
    assert math.isnan(sweep.flag_none.precision)
    # Flag everything -> no escapes, both good parts wrongly rejected (1 each).
    assert sweep.flag_all.tpr == 1.0 and sweep.flag_all.fpr == 1.0
    assert sweep.flag_all.expected_cost == pytest.approx(2.0)

    # Reject rate rises monotonically as the threshold falls along the sweep.
    reject = [p.reject_rate for p in sweep.points]
    assert reject == sorted(reject)
    assert reject[0] == pytest.approx(0.0) and reject[-1] == pytest.approx(1.0)

    # The optimum: threshold at the 0.35/0.4 midpoint's neighbour catches both
    # defects with a single false reject -> cost 1, the minimum over the sweep.
    best = sweep.best
    assert best.expected_cost == pytest.approx(1.0)
    assert best.tpr == pytest.approx(1.0) and best.fpr == pytest.approx(0.5)
    assert best.missed_defects == pytest.approx(0.0)
    assert best.false_rejects == pytest.approx(1.0)
    assert best.precision == pytest.approx(2.0 / 3.0)  # 2 caught / 3 flagged
    assert best.threshold == pytest.approx(0.225)  # midpoint of the 0.1 / 0.35 scores
    # The optimum is never worse than either extreme.
    assert best.expected_cost <= sweep.flag_none.expected_cost
    assert best.expected_cost <= sweep.flag_all.expected_cost


def test_cost_rates_move_the_recommended_threshold():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.55, 0.4, 0.6, 0.9])
    cheap_escape = sweep_thresholds(
        labels, scores, CostModel(prevalence=0.5, cost_escaped=1.0, cost_false_reject=5.0)
    )
    dear_escape = sweep_thresholds(
        labels, scores, CostModel(prevalence=0.5, cost_escaped=100.0, cost_false_reject=1.0)
    )
    # When escapes are dear relative to false rejects, the optimum flags more
    # (a lower threshold, a higher reject rate) than when escapes are cheap.
    assert dear_escape.best.reject_rate >= cheap_escape.best.reject_rate
    assert dear_escape.best.threshold <= cheap_escape.best.threshold


def test_sweep_is_deterministic_and_needs_both_classes():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.3, 0.7, 0.2, 0.9])
    a = cost_curve_csv(sweep_thresholds(labels, scores))
    b = cost_curve_csv(sweep_thresholds(labels, scores))
    assert a == b  # byte-identical across calls
    with pytest.raises(ValueError):
        sweep_thresholds(np.zeros(4), scores)


def test_report_economics_and_outputs_are_byte_identical(tmp_path):
    # Torch-free report (PCA + local stats) -> recommended method drives the sweep.
    report = run_full_evaluation(
        DataConfig(n_train=60, n_test_clean=20, n_test_defective=20, seed=13),
        include_autoencoder=False,
    )
    sweep = economics_for_report(report)
    assert sweep.method == report.recommendation.method
    assert 0.0 <= sweep.best.reject_rate <= 1.0
    assert sweep.best.expected_cost <= sweep.flag_none.expected_cost

    read = plain_language_read(sweep)
    assert "Recommended operating threshold" in read
    assert f"{sweep.best.reject_rate:.1%}" in read

    csv1 = tmp_path / "cost_curve.csv"
    svg1 = tmp_path / "cost_curve.svg"
    n_csv = write_cost_curve_csv(sweep, csv1)
    n_svg = write_cost_curve_svg(sweep, svg1)
    assert n_csv > 0 and n_svg > 0

    csv_bytes = csv1.read_bytes()
    svg_bytes = svg1.read_bytes()
    # Header + one row per operating point; recommended row flagged exactly once.
    lines = csv_bytes.decode("utf-8").splitlines()
    assert lines[0].startswith("threshold,reject_rate,")
    assert len(lines) == 1 + len(sweep.points)
    assert sum(line.endswith(",1") for line in lines[1:]) == 1
    assert svg_bytes.startswith(b"<svg") and b"</svg>" in svg_bytes
    assert b"recommended:" in svg_bytes

    # Re-writing produces the exact same bytes (no wall-clock, no RNG, no ids).
    write_cost_curve_csv(sweep, csv1)
    write_cost_curve_svg(sweep, svg1)
    assert csv1.read_bytes() == csv_bytes
    assert svg1.read_bytes() == svg_bytes
