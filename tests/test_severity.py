"""Severity grading: the grading rule, the graded sweep's arithmetic, the
reduction to the flat model, the break-even search, and byte-identical outputs.

None of these tests need torch: the layer works on any (labels, scores,
severity index) triple, and the end-to-end test uses the torch-free PCA-only
report.
"""

import numpy as np
import pytest

from qav.data import DataConfig
from qav.economics import CostModel, economics_for_report, sweep_thresholds
from qav.evaluate import run_full_evaluation
from qav.severity import (
    GRADES,
    SeverityModel,
    grade_severity,
    plain_language_read,
    severity_csv,
    severity_for_report,
    severity_svg,
    severity_sweep,
    write_severity_csv,
    write_severity_svg,
)

# Seven parts: four clean, three defective at one severity per grade -- and,
# as on the real study set, the *worst* grade is the hardest to see (the
# critical part scores 0.30, below three clean parts). With the basis equal to
# the sample count and prevalence = positives/total, the per-basis counts equal
# the raw confusion counts, so every EUR below is hand-checkable.
TOY_LABELS = np.array([0, 0, 0, 0, 1, 1, 1])
TOY_SCORES = np.array([0.10, 0.35, 0.40, 0.50, 0.30, 0.60, 0.90])
TOY_INDEX = np.array([0.0, 0.0, 0.0, 0.0, 30.0, 12.0, 4.0])  # critical / major / minor
TOY_TYPES = np.array(["clean"] * 4 + ["texture_break", "blob", "scratch"])
TOY_MODEL = CostModel(prevalence=3 / 7, cost_escaped=10.0, cost_false_reject=6.0, units_basis=7)
TOY_SEVERITY = SeverityModel(cuts=(8.0, 15.0), escape_costs=(1.0, 10.0, 100.0))


def toy_study(severity: SeverityModel = TOY_SEVERITY, model: CostModel = TOY_MODEL):
    flat = sweep_thresholds(TOY_LABELS, TOY_SCORES, model, method_name="toy")
    return severity_sweep(
        TOY_LABELS, TOY_SCORES, TOY_INDEX, TOY_TYPES, flat, severity, method_name="toy"
    )


def test_grading_applies_the_cut_points_and_leaves_clean_parts_ungraded():
    graded = grade_severity(np.array([0.0, 7.99, 8.0, 14.999, 15.0, 900.0]), TOY_SEVERITY)
    assert list(graded) == ["clean", "minor", "major", "major", "critical", "critical"]
    # The cut points are the only knob: widen the middle grade and parts move.
    wide = grade_severity(np.array([9.0]), SeverityModel(cuts=(2.0, 100.0)))
    assert list(wide) == ["major"]


def test_grade_profiles_measure_size_mix_and_detectability():
    study = toy_study()
    assert [p.grade for p in study.profiles] == list(GRADES)
    assert [p.n for p in study.profiles] == [1, 1, 1]
    assert study.mix == {"minor": pytest.approx(1 / 3), "major": pytest.approx(1 / 3),
                         "critical": pytest.approx(1 / 3)}
    # Mix-weighted mean escape cost of 1 / 10 / 100 EUR at an even mix.
    assert study.mean_escape_cost == pytest.approx(37.0)
    # The minor part (score 0.90) outranks every clean part -> perfect AUC; the
    # critical part (0.30) beats only one of the four -> 0.25.
    aucs = {p.grade: p.roc_auc_vs_clean for p in study.profiles}
    assert aucs["minor"] == pytest.approx(1.0)
    assert aucs["critical"] == pytest.approx(0.25)
    assert study.profile("critical").kind_counts["texture_break"] == 1


def test_graded_sweep_costs_match_hand_computed_euros():
    study = toy_study()
    flag_none, flag_all = study.points[0], study.points[-1]
    # Flag nothing: every grade escapes at its own price (1 + 10 + 100).
    assert flag_none.expected_cost == pytest.approx(111.0)
    assert flag_none.escape_cost_by_grade["critical"] == pytest.approx(100.0)
    # Flag everything: no escapes, all four clean parts wrongly pulled at 6 EUR.
    assert flag_all.expected_cost == pytest.approx(24.0)
    assert flag_all.missed_defects == pytest.approx(0.0)
    # Reject rate rises monotonically along the sweep, spanning the full range.
    reject = [p.reject_rate for p in study.points]
    assert reject == sorted(reject)
    assert reject[0] == pytest.approx(0.0) and reject[-1] == pytest.approx(1.0)
    # Every point's total is its own escape ledger plus false rejects.
    for pt in study.points:
        assert pt.escape_cost == pytest.approx(sum(pt.escape_cost_by_grade.values()))
        assert pt.expected_cost == pytest.approx(pt.escape_cost + pt.false_reject_cost)
        assert pt.missed_defects == pytest.approx(sum(pt.missed_by_grade.values()))


def test_equal_grade_costs_reproduce_the_flat_sweep_exactly():
    """The flat model is this model with one price -- the layer is the same
    arithmetic with a finer ledger, not a different one."""
    flat_rate = TOY_MODEL.cost_escaped
    study = toy_study(severity=SeverityModel(cuts=(8.0, 15.0),
                                             escape_costs=(flat_rate, flat_rate, flat_rate)))
    for graded, flat in zip(study.points, study.flat.points, strict=True):
        assert graded.threshold == pytest.approx(flat.threshold)
        assert graded.reject_rate == pytest.approx(flat.reject_rate)
        assert graded.tpr == pytest.approx(flat.tpr)
        assert graded.expected_cost == pytest.approx(flat.expected_cost)
    assert study.best.threshold == pytest.approx(study.flat.best.threshold)
    assert not study.moved


def test_pricing_the_worst_grade_higher_moves_the_operating_point():
    cheap = toy_study(severity=SeverityModel(cuts=(8.0, 15.0), escape_costs=(1.0, 1.0, 1.0)))
    dear = toy_study(severity=SeverityModel(cuts=(8.0, 15.0), escape_costs=(1.0, 1.0, 500.0)))
    # Making critical escapes dear can only argue for flagging more.
    assert dear.best.reject_rate >= cheap.best.reject_rate
    assert dear.best.threshold <= cheap.best.threshold
    # And the graded optimum is never worse than either endpoint under its model.
    assert dear.best.expected_cost <= dear.points[0].expected_cost
    assert dear.best.expected_cost <= dear.points[-1].expected_cost


def test_breakeven_finds_the_critical_price_that_moves_the_point():
    study = toy_study(severity=SeverityModel(cuts=(8.0, 15.0), escape_costs=(1.0, 10.0, 10.0)))
    be = study.breakeven
    assert be is not None
    assert be.critical_cost > study.severity.cost("major")
    assert be.multiple == pytest.approx(be.critical_cost / study.severity.cost("major"))
    # At the reported price the point really has moved...
    moved = toy_study(
        severity=study.severity.with_critical_cost(be.critical_cost)
    )
    assert moved.best.threshold < study.flat.best.threshold
    assert moved.best.threshold == pytest.approx(be.threshold)
    assert moved.best.reject_rate == pytest.approx(be.reject_rate)
    # ... and just below it, it has not (the search brackets the step).
    just_below = toy_study(
        severity=study.severity.with_critical_cost(be.critical_cost * 0.9)
    )
    assert just_below.best.threshold >= study.flat.best.threshold


def test_breakeven_is_none_when_no_price_within_the_cap_moves_the_point():
    # Free scrap: the flat optimum already catches every defect, so no escape
    # price -- at any grade -- can argue for flagging more.
    free_scrap = CostModel(prevalence=3 / 7, cost_escaped=10.0, cost_false_reject=0.0,
                           units_basis=7)
    study = toy_study(model=free_scrap)
    assert study.flat.best.missed_defects == pytest.approx(0.0)
    assert study.breakeven is None


def test_cost_neutral_regrade_isolates_level_from_shape():
    study = toy_study()
    mean = study.mean_escape_cost
    factor = TOY_MODEL.cost_escaped / mean
    for grade in GRADES:
        assert study.neutral_costs[grade] == pytest.approx(TOY_SEVERITY.cost(grade) * factor)
    # The neutral rates average to the flat rate under the measured mix.
    assert sum(study.mix[g] * study.neutral_costs[g] for g in GRADES) == pytest.approx(
        TOY_MODEL.cost_escaped
    )
    rebuilt = sum(
        study.neutral_costs[g] * study.best.missed_by_grade[g] for g in GRADES
    ) + study.best.false_reject_cost
    assert study.neutral_cost_at_best == pytest.approx(rebuilt)


def test_degenerate_grading_is_rejected():
    # Cut points that leave a grade empty would silently drop it from the
    # ledger, so the sweep refuses to run.
    with pytest.raises(ValueError):
        toy_study(severity=SeverityModel(cuts=(0.5, 1.0), escape_costs=(1.0, 10.0, 100.0)))


def test_csv_and_svg_are_byte_identical_and_carry_both_optima(tmp_path):
    study = toy_study()
    assert severity_csv(study) == severity_csv(study)
    assert severity_svg(study) == severity_svg(study)

    csv_file = tmp_path / "severity.csv"
    svg_file = tmp_path / "severity.svg"
    n_csv = write_severity_csv(study, csv_file)
    n_svg = write_severity_svg(study, svg_file)
    assert n_csv > 0 and n_svg > 0
    csv_bytes = csv_file.read_bytes()
    svg_bytes = svg_file.read_bytes()

    lines = csv_bytes.decode("utf-8").splitlines()
    assert lines[0].startswith("threshold,reject_rate,fpr,recall,recall_minor,")
    assert len(lines) == 1 + len(study.points)
    columns = lines[0].split(",")
    weighted = columns.index("is_recommended_weighted")
    flat = columns.index("is_recommended_flat")
    assert sum(int(line.split(",")[weighted]) for line in lines[1:]) == 1
    assert sum(int(line.split(",")[flat]) for line in lines[1:]) == 1

    assert svg_bytes.startswith(b"<svg") and svg_bytes.rstrip().endswith(b"</svg>")
    for grade in GRADES:
        assert grade.encode() in svg_bytes  # every grade is named, never hue alone
    assert b"[A11]" in svg_bytes  # the labelled-assumption marker survives

    write_severity_csv(study, csv_file)
    write_severity_svg(study, svg_file)
    assert csv_file.read_bytes() == csv_bytes
    assert svg_file.read_bytes() == svg_bytes


def test_report_severity_layer_end_to_end():
    report = run_full_evaluation(
        DataConfig(n_train=60, n_test_clean=20, n_test_defective=20, seed=13),
        include_autoencoder=False,
    )
    sweep = economics_for_report(report)
    study = severity_for_report(report, sweep=sweep)
    assert study.method == report.recommendation.method
    assert sum(p.n for p in study.profiles) == int(report.dataset.test_labels.sum())
    assert sum(p.share for p in study.profiles) == pytest.approx(1.0)
    # The graded sweep walks the flat sweep's grid, point for point.
    assert len(study.points) == len(study.flat.points)
    for graded, flat in zip(study.points, study.flat.points, strict=True):
        assert graded.threshold == pytest.approx(flat.threshold)
    assert study.at_flat_best.threshold == pytest.approx(study.flat.best.threshold)

    read = plain_language_read(study)
    assert "[A11]" in read and "ILLUSTRATIVE" in read
    assert f"{study.mean_escape_cost:.2f} EUR" in read
    assert "Break-even" in read  # stated either way -- a price, or "no price moves it"
    assert f"{study.best.reject_rate:.2%}" in read or f"{study.flat.best.reject_rate:.2%}" in read
