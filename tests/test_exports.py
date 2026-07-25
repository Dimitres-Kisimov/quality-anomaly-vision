"""Deliverables build end-to-end on a tiny config (torch required, else skip)."""

import pytest

torch = pytest.importorskip("torch")

import openpyxl  # noqa: E402

from qav.data import DataConfig  # noqa: E402
from qav.evaluate import run_full_evaluation  # noqa: E402
from qav.exports import build_deliverables  # noqa: E402
from qav.model import TrainConfig  # noqa: E402


def test_deliverables_and_figures_are_written_and_nonempty(tmp_path):
    report = run_full_evaluation(
        DataConfig(n_train=40, n_test_clean=12, n_test_defective=12, seed=13),
        TrainConfig(epochs=2, batch_size=16, seed=3),
    )
    assert len(report.methods) == 3
    assert report.recommendation.method in {m.name for m in report.methods}

    out = tmp_path / "deliverables"
    figs = tmp_path / "figures"
    sizes = build_deliverables(report, out, figs)

    pdf = out / "qa_defect_report.pdf"
    xlsx = out / "qa_defect_metrics.xlsx"
    assert pdf.exists() and sizes[str(pdf)] > 10_240, "PDF should be a real multi-page report"
    assert xlsx.exists() and sizes[str(xlsx)] > 4_096

    wb = openpyxl.load_workbook(xlsx)
    assert set(wb.sheetnames) == {"Metrics", "PerDefectType", "Assumptions", "PerImageScores"}
    metrics = wb["Metrics"]
    assert metrics.max_row == 4, "header + one row per method"
    assert wb["PerImageScores"].max_row == 25, "header + one row per test image"

    for name in ("gallery.png", "roc_pr.png", "per_type_auc.png"):
        f = figs / name
        assert f.exists() and f.stat().st_size > 1_024, f"{name} should be a real image"
