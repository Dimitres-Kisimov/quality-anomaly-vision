"""Recalibration/OCAP layer: hand-checked rate-matching and cost arithmetic,
policy semantics (a threshold cannot restore separability; refit can; repair
recovers everything by construction), collapse-to-base-case behavior, and
byte-identical CSV/SVG outputs.

None of these tests need torch: the pure pieces are numpy-only, and the
end-to-end tests reuse the torch-free (PCA + local stats) report.
"""

import pytest

from qav.baselines import LocalStatsDetector, PCAReconstruction
from qav.data import DataConfig
from qav.economics import CostModel, economics_for_report
from qav.evaluate import run_full_evaluation
from qav.recalibration import (
    POLICY_ORDER,
    RecalConfig,
    outcome_from_rates,
    plain_language_read,
    rate_matched_threshold,
    recalibration_csv,
    recalibration_for_report,
    refit_factory_for,
    refit_window,
    write_recalibration_csv,
    write_recalibration_svg,
)
from qav.spc import FlagRates, SPCConfig

# --------------------------------------------------------------------------
# Rate matching (pure, hand-computed)
# --------------------------------------------------------------------------

# clean scores {1, 2}, defect scores {3, 4}, prevalence 0.5. Candidate
# thresholds: 4.003, 3.5, 2.5, 1.5, 0.997 (midpoints + one padded point per
# end, gap = (4 - 1) * 1e-3). Mixture rates there: 0, 0.25, 0.5, 0.75, 1.
CLEAN = [1.0, 2.0]
DEFECT = [3.0, 4.0]


def test_rate_matched_threshold_hand_computed():
    t = rate_matched_threshold(CLEAN, DEFECT, 0.5, target_rate=0.5)
    assert t == pytest.approx(2.5)
    # target 0.6 -> nearest achievable rates are 0.5 (diff 0.1) and 0.75
    # (diff 0.15): the 0.5-rate threshold wins.
    assert rate_matched_threshold(CLEAN, DEFECT, 0.5, 0.6) == pytest.approx(2.5)
    # target 0 -> flag nothing (padded point above the maximum score).
    assert rate_matched_threshold(CLEAN, DEFECT, 0.5, 0.0) == pytest.approx(4.003)
    # target 1 -> flag everything (padded point below the minimum score).
    assert rate_matched_threshold(CLEAN, DEFECT, 0.5, 1.0) == pytest.approx(0.997)


def test_rate_matched_threshold_tie_breaks_to_the_higher_threshold():
    # target 0.625 sits exactly between the achievable rates 0.5 (t = 2.5)
    # and 0.75 (t = 1.5): the tie goes to the more conservative threshold.
    assert rate_matched_threshold(CLEAN, DEFECT, 0.5, 0.625) == pytest.approx(2.5)


def test_rate_matched_threshold_weights_by_prevalence():
    # At 10% prevalence the mixture is dominated by the clean class: rates at
    # the candidates become 0, 0.05, 0.10, 0.55, 1.0 -- so a 0.10 target now
    # needs the threshold that flags both defect scores but no clean one.
    assert rate_matched_threshold(CLEAN, DEFECT, 0.1, 0.10) == pytest.approx(2.5)
    assert rate_matched_threshold(CLEAN, DEFECT, 0.1, 0.05) == pytest.approx(3.5)


def test_rate_matched_threshold_validates_inputs():
    for bad_prev in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError):
            rate_matched_threshold(CLEAN, DEFECT, bad_prev, 0.5)
    for bad_target in (-0.1, 1.1):
        with pytest.raises(ValueError):
            rate_matched_threshold(CLEAN, DEFECT, 0.5, bad_target)
    with pytest.raises(ValueError):
        rate_matched_threshold([], DEFECT, 0.5, 0.5)
    with pytest.raises(ValueError):
        rate_matched_threshold(CLEAN, [], 0.5, 0.5)


# --------------------------------------------------------------------------
# Cost arithmetic (hand-computed, defaults: prevalence 1.5%, 35/3 EUR, basis 1000)
# --------------------------------------------------------------------------


def test_outcome_from_rates_hand_computed():
    o = outcome_from_rates(
        "x", 0.1, 0.5, FlagRates(p_clean=0.01, p_defective=0.4, n_clean=10, n_defective=10),
        auc=0.9, model=CostModel(),
    )
    # 15 defective and 985 clean parts per 1,000: catch 6, miss 9, wrongly pull 9.85.
    assert o.caught_defects == pytest.approx(6.0)
    assert o.missed_defects == pytest.approx(9.0)
    assert o.false_rejects == pytest.approx(9.85)
    assert o.flag_rate == pytest.approx(0.01585)
    # 35 * 9 + 3 * 9.85 = 315 + 29.55.
    assert o.expected_cost == pytest.approx(344.55)
    assert o.d_cost == 0.0  # no baseline given

    o2 = outcome_from_rates(
        "x", 0.1, 0.5, FlagRates(0.01, 0.4, 10, 10), 0.9, CostModel(), base_cost=300.0
    )
    assert o2.d_cost == pytest.approx(44.55)


# --------------------------------------------------------------------------
# Refit ingredients
# --------------------------------------------------------------------------


def test_refit_factory_builds_a_fresh_unfitted_same_family_detector():
    pca = PCAReconstruction(n_components=5, smooth_sigma=0.5)
    fresh = refit_factory_for(pca)()
    assert isinstance(fresh, PCAReconstruction)
    assert fresh is not pca
    assert (fresh.n_components, fresh.smooth_sigma) == (5, 0.5)
    assert fresh.mean_ is None  # unfitted

    local = LocalStatsDetector(window=5, smooth_sigma=0.0)
    fresh = refit_factory_for(local)()
    assert isinstance(fresh, LocalStatsDetector)
    assert (fresh.window, fresh.smooth_sigma) == (5, 0.0)
    assert fresh.stats_ is None

    with pytest.raises(ValueError):
        refit_factory_for(object())


def test_refit_window_is_seeded_and_validated():
    cfg = RecalConfig(refit_clean_frames=6, refit_seed=3)
    a = refit_window(cfg, image_size=32)
    b = refit_window(cfg, image_size=32)
    assert a.shape == (6, 32, 32)
    assert (a == b).all()
    # Its own seed: NOT the study's training frames.
    assert (a != refit_window(RecalConfig(refit_clean_frames=6, refit_seed=4), 32)).any()
    with pytest.raises(ValueError):
        refit_window(RecalConfig(refit_clean_frames=1))


# --------------------------------------------------------------------------
# End to end on the torch-free report (PCA recommended)
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
SMALL_RECAL = RecalConfig(drift_deltas=(0.02, 0.10), refit_clean_frames=40, refit_seed=5)


@pytest.fixture(scope="module")
def report():
    return run_full_evaluation(
        DataConfig(n_train=60, n_test_clean=20, n_test_defective=20, seed=13),
        include_autoencoder=False,
    )


@pytest.fixture(scope="module")
def study(report):
    return recalibration_for_report(report, config=SMALL_RECAL, spc_config=SMALL_SPC)


def test_study_reuses_the_recommended_method_and_threshold(report, study):
    assert study.method == report.recommendation.method
    assert study.base_threshold == economics_for_report(report).best.threshold
    assert study.target_flag_rate == study.in_control.flag_rate
    # One outcome per (drift level, policy), policies in the fixed order.
    assert [o.delta for o in study.outcomes] == [0.02, 0.02, 0.02, 0.02, 0.1, 0.1, 0.1, 0.1]
    assert [o.policy for o in study.outcomes] == list(POLICY_ORDER) * 2
    assert study.in_control.policy == "in_control"
    assert study.in_control.d_cost == 0.0


def test_fix_camera_recovers_the_in_control_point_exactly(study):
    base = study.in_control
    for delta in SMALL_RECAL.drift_deltas:
        o = study.outcome("fix_camera", delta)
        assert o.threshold == base.threshold
        assert o.p_clean == base.p_clean
        assert o.p_defective == base.p_defective
        assert o.roc_auc == base.roc_auc
        assert o.expected_cost == base.expected_cost
        assert o.d_cost == 0.0
        assert study.cost_recovered("fix_camera", delta) == pytest.approx(1.0)


def test_recentring_the_threshold_cannot_restore_separability(study):
    for delta in SMALL_RECAL.drift_deltas:
        na = study.outcome("no_action", delta)
        rc = study.outcome("rate_recenter", delta)
        # Same scores, different threshold: ROC-AUC is bitwise identical...
        assert rc.roc_auc == na.roc_auc
        # ...and both are degraded vs in control (brightness hurts PCA).
        assert na.roc_auc < study.in_control.roc_auc
        # But the flag rate really is re-centered: strictly closer to target.
        assert abs(rc.flag_rate - study.target_flag_rate) < abs(
            na.flag_rate - study.target_flag_rate
        )
        # Upward drift pushes scores up, so the re-centered threshold is higher.
        assert rc.threshold > study.base_threshold


def test_refit_recovers_separability_and_most_of_the_cost(study):
    d_max = SMALL_RECAL.drift_deltas[-1]
    na = study.outcome("no_action", d_max)
    rc = study.outcome("rate_recenter", d_max)
    rf = study.outcome("refit_recent", d_max)
    # Separability: the re-fit detector absorbs the DC shift the frozen one cannot.
    assert rf.roc_auc > na.roc_auc + 0.1
    # Cost ordering at the strongest drift: refit < recenter < do nothing.
    assert rf.expected_cost < rc.expected_cost < na.expected_cost
    assert study.cost_recovered("refit_recent", d_max) > 0.9
    # Refit beats doing nothing at every measured drift level.
    for delta in SMALL_RECAL.drift_deltas:
        assert (
            study.outcome("refit_recent", delta).expected_cost
            < study.outcome("no_action", delta).expected_cost
        )


def test_collapse_to_base_case_zero_drift(report):
    null = recalibration_for_report(
        report,
        config=RecalConfig(drift_deltas=(0.0,), refit_clean_frames=40, refit_seed=5),
        spc_config=SMALL_SPC,
    )
    base = null.in_control
    # Zero drift is a no-op: doing nothing IS the in-control operating point,
    # and so is repairing the camera.
    for policy in ("no_action", "fix_camera"):
        o = null.outcome(policy, 0.0)
        assert o.p_clean == base.p_clean
        assert o.p_defective == base.p_defective
        assert o.roc_auc == base.roc_auc
        assert o.expected_cost == base.expected_cost
        assert o.d_cost == 0.0
    # Rate re-centering on an unchanged stream reproduces the same operating
    # point (its threshold may sit elsewhere in the same score gap).
    rc = null.outcome("rate_recenter", 0.0)
    assert rc.p_clean == base.p_clean
    assert rc.p_defective == base.p_defective
    assert rc.d_cost == 0.0
    # The refit detector is fit on a fresh, smaller window, so it is close but
    # NOT identical -- that honesty is part of the design (documented [A10]).
    rf = null.outcome("refit_recent", 0.0)
    assert abs(rf.roc_auc - base.roc_auc) < 0.1
    # With no drift-induced loss there is no recovery share to quote.
    with pytest.raises(ValueError):
        null.cost_recovered("refit_recent", 0.0)


def test_config_validation(report):
    with pytest.raises(ValueError):
        recalibration_for_report(
            report, config=RecalConfig(drift_deltas=()), spc_config=SMALL_SPC
        )
    with pytest.raises(ValueError):
        recalibration_for_report(
            report, config=RecalConfig(drift_deltas=(-0.02,)), spc_config=SMALL_SPC
        )


def test_outputs_are_structured_and_byte_identical(report, study, tmp_path):
    read = plain_language_read(study)
    assert study.method in read
    assert "action plan" in read
    assert "verified-clean" in read and "[A10]" in read

    csv_text = recalibration_csv(study)
    lines = csv_text.splitlines()
    assert lines[0].startswith("policy,delta,threshold,")
    # Header + in-control anchor + one row per (drift level, policy).
    assert len(lines) == 1 + 1 + len(SMALL_RECAL.drift_deltas) * len(POLICY_ORDER)
    assert lines[1].startswith("in_control,0.0000,")

    csv_file = tmp_path / "recalibration.csv"
    svg_file = tmp_path / "recalibration.svg"
    assert write_recalibration_csv(study, csv_file) > 0
    assert write_recalibration_svg(study, svg_file) > 0
    csv_bytes = csv_file.read_bytes()
    svg_bytes = svg_file.read_bytes()
    assert svg_bytes.startswith(b"<svg") and svg_bytes.rstrip().endswith(b"</svg>")
    for policy in POLICY_ORDER:
        assert policy.replace("_", " ").encode() in svg_bytes

    # Re-running the whole study reproduces both files byte for byte.
    again = recalibration_for_report(report, config=SMALL_RECAL, spc_config=SMALL_SPC)
    write_recalibration_csv(again, csv_file)
    write_recalibration_svg(again, svg_file)
    assert csv_file.read_bytes() == csv_bytes
    assert svg_file.read_bytes() == svg_bytes
