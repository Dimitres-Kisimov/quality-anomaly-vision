# Business case: automated surface-defect screening at a QA station

This document translates the study in this repository into the decision it is meant
to support. Every quantity that is not measured by the code is an **assumption,
labelled [A1]-[A7]** and kept visible in the arithmetic. The measured inputs come
from synthetic textures, so the honest scope of this case is: *is a pilot on real
camera data worth funding, and with which method?* — not *what will production
performance be*.

## Situation

A mid-size plant runs a finishing line for textured metal panels. Today, surface
quality is checked by eye: an inspector pulls a sample of parts off the line and
looks for scratches, pits and pattern irregularities. Everything that is not sampled
ships uninspected.

## Quantified problem (assumptions labelled)

- [A1] The station passes **4,000 parts per day** (two shifts).
- [A2] Manual inspection samples **10%** of parts (400/day) at ~4 s per part
  (~27 min of inspection time per day).
- [A3] True defect rate is **1.5%** -> ~60 defective parts/day.
- [A4] An escaped defect costs **35 EUR** on average (downstream scrap, rework,
  complaint handling). This is the dominant lever in the whole case; see the
  sensitivity note.
- [A5] Fully loaded inspector cost is **30 EUR/h**.
- [A6] One station of commodity hardware (camera, lighting, small PC) costs
  **6,000 EUR** one-off.
- [A7] Sampled defective parts are always caught (generous to the manual baseline).

Manual baseline: 10% sampling catches at most 10% of the ~60 defective parts
(~6/day) [A2, A3, A7]. **~54 defective parts escape per day**, ~1,890 EUR/day of
escape cost [A4].

## Proposed solution

Photograph **every** part and run the screening method recommended by the study —
**PCA reconstruction error** (chosen by a rule fixed before results were seen;
the small conv autoencoder did not beat it by the required margin). Parts whose
anomaly score exceeds the threshold are routed to the existing human inspector, so
the system augments rather than replaces judgment.

Measured operating point (synthetic benchmark, 5% false-alarm rate):
**TPR = 0.407** — PCA flags ~40.7% of defective parts at the threshold that lets
through 5% of clean parts as false alarms.

## Effect and ROI (all arithmetic from the labelled assumptions)

| Quantity | Manual baseline | With 100% PCA screening |
|---|---|---|
| Defective parts caught / day | ~6 | 6 + 0.407 x 54 ~ **28** (screening acts on the 54 currently unsampled) |
| Escapes / day | ~54 | ~32 |
| Escape cost / day [A4] | ~1,890 EUR | ~1,120 EUR |
| False alarms to re-check / day | — | 5% x ~3,940 clean ~ 197 parts, ~13 min ~ **7 EUR/day** [A2, A5] |

Net saving: (54 - 32) x 35 EUR - 7 EUR ~ **760 EUR/day**, so the 6,000 EUR
hardware [A6] pays back in **~8 working days** — *if* the synthetic-benchmark TPR
transferred unchanged, which nobody should assume.

**Sensitivity (the honest part):** the case scales almost linearly with [A4] and
with the real-world TPR. If the escape cost is 5 EUR instead of 35 EUR, the net
saving is ~103 EUR/day and payback is ~3 months — still positive. If the real-world
TPR at 5% FPR came out at half the synthetic value (0.20), payback under [A4]=35 EUR
is ~16 working days. The case survives pessimistic inputs; what it cannot survive is
skipping the pilot.

## Stakeholders

- **QA lead** — owner of the escape rate; defines the defect catalogue for the pilot.
- **Line operators** — will handle the flagged-parts queue; false-alarm rate is their
  daily experience of the system.
- **Process engineer** — texture-break-type defects hint at upstream process drift;
  the per-type breakdown is their early-warning signal.
- **Plant controller** — owns [A4] and [A6]; the sensitivity table is written for
  this conversation.
- **IT/OT** — deployment target is a small CPU-only box; no GPU, no cloud dependency.

## Deliverable and recommended next step

This repository plus its generated outputs: `deliverables/qa_defect_report.pdf`
(executive summary with disclaimer, examples, curves, tables, recommendation) and
`deliverables/qa_defect_metrics.xlsx` (all metrics, per-image scores, and these
assumptions). 

Recommended next step: a **4-week pilot on real camera data** at one station —
collect ~2,000 real clean images, re-fit the PCA screen, measure the true TPR/FPR
curve against inspector labels, and only then revisit this ROI table with measured
inputs. If texture-break-type defects dominate the real defect mix, re-run the
method comparison: the study shows the autoencoder is the only method meaningfully
above chance on that class (AUC 0.609 vs 0.521/0.453), and the recommendation could
legitimately flip.
