"""SPC layer: hand-checked p-chart limits and Western Electric rules, a
deterministic modelled stream, collapse-to-base-case behavior, and
byte-identical CSV/SVG outputs.

None of these tests need torch: the chart machinery is pure, and the
end-to-end tests reuse the torch-free (PCA + local stats) report.
"""

import dataclasses
import math

import numpy as np
import pytest

from qav.data import DataConfig
from qav.economics import economics_for_report
from qav.evaluate import run_full_evaluation
from qav.spc import (
    FlagRates,
    PChartLimits,
    SPCConfig,
    SubgroupCondition,
    drift_deltas,
    limits_from_counts,
    measured_flag_rates,
    p_chart_limits,
    plain_language_read,
    simulate_subgroups,
    spc_csv,
    spc_for_report,
    western_electric_flags,
    write_spc_csv,
    write_spc_svg,
)

# Round numbers so every zone boundary is hand-checkable: center 0.5,
# sigma 0.1 -> 1/2/3-sigma lines at 0.6/0.7/0.8 (high) and 0.4/0.3/0.2 (low).
TOY = PChartLimits(center=0.5, sigma=0.1, subgroup_size=100)


# --------------------------------------------------------------------------
# Limits
# --------------------------------------------------------------------------


def test_p_chart_limits_hand_computed():
    lim = p_chart_limits(0.2, 25)
    # sigma = sqrt(0.2 * 0.8 / 25) = sqrt(0.0064) = 0.08 exactly
    assert lim.sigma == pytest.approx(0.08)
    assert lim.ucl == pytest.approx(0.44)
    assert lim.lcl == 0.0  # 0.2 - 0.24 < 0 -> clipped

    lim2 = p_chart_limits(0.5, 100)
    assert lim2.sigma == pytest.approx(0.05)
    assert lim2.ucl == pytest.approx(0.65)
    assert lim2.lcl == pytest.approx(0.35)
    assert lim2.zone(1) == (pytest.approx(0.45), pytest.approx(0.55))
    assert lim2.zone(2) == (pytest.approx(0.40), pytest.approx(0.60))

    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            p_chart_limits(bad, 100)
    with pytest.raises(ValueError):
        p_chart_limits(0.5, 0)


def test_limits_from_counts_pools_phase_one():
    lim = limits_from_counts([2, 4, 6], 100)
    assert lim.center == pytest.approx(0.04)  # 12 flags / 300 parts
    assert lim.sigma == pytest.approx(math.sqrt(0.04 * 0.96 / 100))
    with pytest.raises(ValueError):
        limits_from_counts([0, 0, 0], 100)  # a p-chart needs a nonzero rate
    with pytest.raises(ValueError):
        limits_from_counts([50], 50)  # the screen flags everything
    with pytest.raises(ValueError):
        limits_from_counts([], 100)


# --------------------------------------------------------------------------
# Western Electric rules (strict "beyond"; center-line points reset runs)
# --------------------------------------------------------------------------


def test_rule1_beyond_three_sigma_is_strict():
    hits = western_electric_flags([0.5, 0.5, 0.81], TOY)
    assert [h.beyond_3sigma for h in hits] == [False, False, True]
    # Exactly on the limit is NOT beyond it.
    assert not western_electric_flags([0.8], TOY)[0].beyond_3sigma
    # Low side fires symmetrically.
    hits = western_electric_flags([0.5, 0.19], TOY)
    assert hits[1].beyond_3sigma and not hits[0].beyond_3sigma
    assert not western_electric_flags([0.2], TOY)[0].beyond_3sigma


def test_rule2_two_of_three_beyond_two_sigma_same_side():
    hits = western_electric_flags([0.5, 0.72, 0.5, 0.73], TOY)
    assert [h.two_of_three_beyond_2sigma for h in hits] == [False, False, False, True]
    # The signalling point itself must be beyond the zone line.
    hits = western_electric_flags([0.72, 0.73, 0.5], TOY)
    assert not hits[2].two_of_three_beyond_2sigma
    assert hits[1].two_of_three_beyond_2sigma  # two consecutive beyond also signal
    # Opposite sides never combine.
    hits = western_electric_flags([0.75, 0.25, 0.5], TOY)
    assert not any(h.two_of_three_beyond_2sigma for h in hits)


def test_rule3_four_of_five_beyond_one_sigma_same_side():
    seq = [0.61, 0.62, 0.5, 0.63, 0.61]
    hits = western_electric_flags(seq, TOY)
    assert [h.four_of_five_beyond_1sigma for h in hits] == [False, False, False, False, True]
    low = [1.0 - v for v in seq]  # mirror below the center line
    hits = western_electric_flags(low, TOY)
    assert [h.four_of_five_beyond_1sigma for h in hits] == [False, False, False, False, True]


def test_rule4_eight_consecutive_on_one_side():
    hits = western_electric_flags([0.51] * 8, TOY)
    assert [h.eight_same_side for h in hits] == [False] * 7 + [True]
    assert [h.eight_same_side for h in western_electric_flags([0.49] * 8, TOY)][-1] is True
    # Seven is not enough, and a point exactly on the center line resets the run.
    assert not any(h.eight_same_side for h in western_electric_flags([0.51] * 7, TOY))
    assert not any(
        h.eight_same_side for h in western_electric_flags([0.51] * 4 + [0.5] + [0.51] * 4, TOY)
    )


def test_rules_all_quiet_on_the_center_line():
    hits = western_electric_flags([0.5] * 20, TOY)
    assert not any(h.alarm for h in hits)


# --------------------------------------------------------------------------
# The modelled stream
# --------------------------------------------------------------------------


def test_simulate_subgroups_is_deterministic_and_bounded():
    cond = SubgroupCondition(0.5, FlagRates(0.1, 0.9, 10, 10))
    a = simulate_subgroups([cond] * 6, 1000, seed=5)
    b = simulate_subgroups([cond] * 6, 1000, seed=5)
    assert a == b
    assert a != simulate_subgroups([cond] * 6, 1000, seed=6)
    for n_def, flags in a:
        assert 0 <= n_def <= 1000
        assert 0 <= flags <= 1000


def test_simulate_subgroups_collapses_to_exact_counting():
    # With P(flag | defective) = 1 and P(flag | clean) = 0, the flag count IS
    # the defective count -- no randomness left in the flagging stage.
    cond = SubgroupCondition(0.3, FlagRates(0.0, 1.0, 10, 10))
    for n_def, flags in simulate_subgroups([cond] * 8, 500, seed=11):
        assert flags == n_def


def test_measured_flag_rates_hand_computed():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0, 0, 1, 1])
    rates = measured_flag_rates(scores, labels, threshold=0.15)
    assert rates.p_clean == pytest.approx(0.5)  # only 0.2 of the two clean parts
    assert rates.p_defective == pytest.approx(1.0)
    assert (rates.n_clean, rates.n_defective) == (2, 2)
    # >= threshold: a score exactly at the threshold is flagged.
    assert measured_flag_rates(scores, labels, 0.2).p_clean == pytest.approx(0.5)
    with pytest.raises(ValueError):
        measured_flag_rates(scores, np.zeros(4), 0.15)


def test_drift_deltas_ramp_linearly_to_the_maximum():
    cfg = SPCConfig(phase2_subgroups=25, change_at=10, drift_max_brightness=0.02)
    deltas = drift_deltas(cfg)
    assert len(deltas) == 15
    assert deltas[0] == pytest.approx(0.02 / 15)
    assert deltas[-1] == pytest.approx(0.02)
    assert all(b > a for a, b in zip(deltas, deltas[1:], strict=False))
    with pytest.raises(ValueError):
        drift_deltas(SPCConfig(phase2_subgroups=10, change_at=10))


# --------------------------------------------------------------------------
# End to end on the torch-free report (PCA + local stats)
# --------------------------------------------------------------------------

SMALL_SPC = SPCConfig(
    subgroup_size=200,
    phase1_subgroups=10,
    phase2_subgroups=8,
    change_at=3,
    shifted_prevalence=0.10,
    seed=101,
    calibration_clean=80,
    calibration_defective=60,
    calibration_seed=99,
)


@pytest.fixture(scope="module")
def report():
    return run_full_evaluation(
        DataConfig(n_train=60, n_test_clean=20, n_test_defective=20, seed=13),
        include_autoencoder=False,
    )


@pytest.fixture(scope="module")
def study(report):
    return spc_for_report(report, config=SMALL_SPC)


def test_study_reuses_the_recommended_method_and_threshold(report, study):
    assert study.method == report.recommendation.method
    assert study.threshold == economics_for_report(report).best.threshold
    assert len(study.phase1) == SMALL_SPC.phase1_subgroups
    assert [s.name for s in study.scenarios] == ["process_shift", "camera_drift"]
    for sc in study.scenarios:
        assert len(sc.points) == SMALL_SPC.phase1_subgroups + SMALL_SPC.phase2_subgroups
        assert len(sc.rules) == len(sc.points)
        assert sc.change_index == SMALL_SPC.phase1_subgroups + SMALL_SPC.change_at
        assert all(0.0 <= p.proportion <= 1.0 for p in sc.points)
        # Phase 1 is shared, byte for byte, between the scenarios.
        assert sc.points[: SMALL_SPC.phase1_subgroups] == study.phase1
    assert 0.0 < study.limits.center < 1.0
    assert study.limits.ucl <= 1.0 and study.limits.lcl >= 0.0


def test_process_shift_scenario_alarms_after_the_change(study):
    shift = study.scenarios[0]
    # The injected shift (1.5% -> 10% prevalence) is large at n = 200: it must
    # be caught, it must be caught after the change point, and (for this seed)
    # without any false alarm before it.
    assert shift.first_alarm is not None
    assert shift.first_alarm >= shift.change_index
    assert shift.detection_delay is not None and shift.detection_delay <= 3
    assert shift.false_alarms == 0
    assert shift.phase1_alarm_count == 0
    # Before the change the stream really ran in control.
    for pt in shift.points[: shift.change_index]:
        assert pt.true_prevalence == pytest.approx(SMALL_SPC.in_control_prevalence)
    for pt in shift.points[shift.change_index :]:
        assert pt.true_prevalence == pytest.approx(SMALL_SPC.shifted_prevalence)


def test_camera_drift_scenario_never_changes_the_true_defect_rate(study):
    drift = study.scenarios[1]
    # The whole point: prevalence stays at the in-control value everywhere...
    assert all(
        pt.true_prevalence == pytest.approx(SMALL_SPC.in_control_prevalence)
        for pt in drift.points
    )
    # ...while the measured clean flag rate rises with the brightness delta
    # (the PCA screen mistakes a DC shift for anomaly -- the robustness finding).
    assert study.drift_rates[-1][1].p_clean >= study.baseline_rates.p_clean
    assert drift.points[-1].brightness_delta == pytest.approx(
        SMALL_SPC.drift_max_brightness
    )


def test_collapse_to_base_case_no_change_no_alarm(report):
    null_cfg = dataclasses.replace(
        SMALL_SPC,
        shifted_prevalence=SMALL_SPC.in_control_prevalence,
        drift_max_brightness=0.0,
    )
    null = spc_for_report(report, config=null_cfg)
    base = null.baseline_rates
    for sc in null.scenarios:
        for pt in sc.points:
            assert pt.true_prevalence == pytest.approx(SMALL_SPC.in_control_prevalence)
            assert pt.p_flag_clean == pytest.approx(base.p_clean)
            assert pt.p_flag_defective == pytest.approx(base.p_defective)
    # A zero-magnitude drift measures *identical* rates (delta 0 is a no-op).
    for _, rates in null.drift_rates:
        assert rates == base
    # With nothing injected, the monitored stream stays quiet end to end.
    for sc in null.scenarios:
        assert sc.phase1_alarm_count == 0
        assert sc.false_alarms == 0
        assert sc.first_alarm is None
        assert sc.detection_delay is None


def test_outputs_are_structured_and_byte_identical(report, study, tmp_path):
    read = plain_language_read(study)
    assert "p-chart" in read and "modelled" in read
    assert study.method in read

    csv_text = spc_csv(study)
    lines = csv_text.splitlines()
    assert lines[0].startswith("scenario,subgroup,phase,n,")
    # Header + shared phase 1 once + each scenario's phase 2.
    assert len(lines) == 1 + SMALL_SPC.phase1_subgroups + 2 * SMALL_SPC.phase2_subgroups
    assert sum(line.split(",")[0] == "baseline" for line in lines[1:]) == 10
    # Exactly one change-point row per scenario.
    assert sum(line.endswith(",1") for line in lines[1:]) == 2

    csv_file = tmp_path / "spc_chart.csv"
    svg_file = tmp_path / "spc_chart.svg"
    assert write_spc_csv(study, csv_file) > 0
    assert write_spc_svg(study, svg_file) > 0
    csv_bytes = csv_file.read_bytes()
    svg_bytes = svg_file.read_bytes()
    assert svg_bytes.startswith(b"<svg") and svg_bytes.rstrip().endswith(b"</svg>")
    assert b"Phase I" in svg_bytes and b"Phase II" in svg_bytes

    # Re-running the whole study reproduces both files byte for byte.
    again = spc_for_report(report, config=SMALL_SPC)
    write_spc_csv(again, csv_file)
    write_spc_svg(again, svg_file)
    assert csv_file.read_bytes() == csv_bytes
    assert svg_file.read_bytes() == svg_bytes
