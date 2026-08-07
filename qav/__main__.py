"""Command-line entry point: ``python -m qav --deliverables``.

Windows-console safe: stdout/stderr are reconfigured to UTF-8 where possible
and every status marker is plain ASCII ([OK] / [FAIL] / [INFO]).
"""

from __future__ import annotations

import argparse
import sys

MIN_DELIVERABLE_BYTES = 10 * 1024


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    parser = argparse.ArgumentParser(
        prog="python -m qav",
        description="Surface-defect screening study: synthetic textures, three "
        "anomaly detectors, honest small-scale evaluation.",
    )
    parser.add_argument("--deliverables", action="store_true",
                        help="run the full study and write the PDF report, Excel "
                        "workbook and README figures")
    parser.add_argument("--outdir", default="deliverables",
                        help="directory for PDF + Excel (default: deliverables)")
    parser.add_argument("--figdir", default="figures",
                        help="directory for README PNGs (default: figures)")
    parser.add_argument("--seed", type=int, default=None,
                        help="override the dataset seed (default: 7)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override autoencoder epochs (default: 15)")
    args = parser.parse_args(argv)

    if not args.deliverables:
        parser.print_help()
        return 0

    from qav.data import DataConfig
    from qav.evaluate import run_full_evaluation
    from qav.exports import build_deliverables
    from qav.model import TrainConfig

    data_cfg = DataConfig() if args.seed is None else DataConfig(seed=args.seed)
    train_cfg = TrainConfig() if args.epochs is None else TrainConfig(epochs=args.epochs)

    print("[INFO] generating data and fitting 3 methods (CPU, deterministic seeds)...")
    report = run_full_evaluation(data_cfg, train_cfg)

    print("[INFO] measured results (image level):")
    header = f"  {'method':22s} {'ROC-AUC':>8s} {'PR-AUC':>8s} {'TPR@5%FPR':>10s} {'meanIoU':>8s}"
    print(header)
    for m in report.methods:
        print(f"  {m.name:22s} {m.roc_auc:8.3f} {m.pr_auc:8.3f} "
              f"{m.tpr_at_5pct_fpr:10.3f} {m.mean_iou:8.3f}")
    for m in report.methods:
        per_type = "  ".join(
            f"{k}={m.per_type_auc[k]:.3f}" for k in sorted(m.per_type_auc)
        )
        print(f"  [INFO] {m.name}: per-type AUC {per_type}")
    print(f"[INFO] recommendation: {report.recommendation.method}")
    print(f"[INFO] rationale: {report.recommendation.rationale}")

    from qav.economics import economics_for_report, plain_language_read

    sweep = economics_for_report(report)
    print("[INFO] inspection-threshold economics (illustrative cost rates, labelled):")
    for line in plain_language_read(sweep).splitlines():
        print(f"  {line}")

    from qav.robustness import plain_language_read as robustness_read
    from qav.robustness import robustness_for_report

    robust = robustness_for_report(report)
    print("[INFO] robustness stress-test (perturbed test images, detectors not re-fit):")
    for line in robustness_read(robust).splitlines():
        print(f"  {line}")

    sizes = build_deliverables(report, args.outdir, args.figdir)
    failed = False
    for path, size in sizes.items():
        must_be_large = path.endswith((".pdf", ".xlsx"))
        if must_be_large and size < MIN_DELIVERABLE_BYTES:
            print(f"[FAIL] {path}: only {size} bytes (< {MIN_DELIVERABLE_BYTES})")
            failed = True
        else:
            print(f"[OK] wrote {path} ({size} bytes)")
    if failed:
        return 1
    print("[OK] deliverables complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
