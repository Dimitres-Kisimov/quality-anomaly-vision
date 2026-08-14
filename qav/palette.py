"""The plate palette: one production line, seen under inspection lights.

Every figure this project emits -- the hand-drawn SVGs, the PDF pages, the
README PNGs -- draws its colors from here, so the deliverables read as one
physical subject rather than as five charts that happened to share a script.
The subject is a finishing line under an inspection lamp:

- **machined-metal neutrals** carry the page itself (surface, panels, grid),
- **material tones** carry identity (steel blue, patina green, copper) --
  the parts and the methods that look at them,
- **inspection-lamp amber** carries caution (a change point, the tempting
  cheap fix),
- **signal red** carries the thing you act on (an alarm, an escaped defect,
  the cost curve you are trying to minimise),
- a **calm green** carries "in control / pass".

Every value below was checked with the data-viz skill's palette validator
(``scripts/validate_palette.js``) against this page's own surface
(``SURFACE``), light mode, all pairs:

- methods trio ``STEEL, PATINA, COPPER`` -- every check PASS (worst all-pairs
  CVD delta-E 8.2 deutan, normal-vision 15.3, all >= 3:1 contrast);
- status/policy quartet ``SIGNAL, LAMP, PATINA, STEEL`` -- every check PASS
  (worst all-pairs CVD delta-E 7.5 deutan -- in the 6-8 floor band, so hue is
  never the only channel: every policy line also carries a dash pattern and a
  direct text label -- normal-vision 15.3, all >= 3:1 contrast);
- severity ramp ``GRADE_COLORS`` -- validated as an *ordinal* ramp (grades are
  ordered, not categorical): monotone lightness, adjacent delta-L >= 0.06,
  hue spread 29 degrees, light end 2.13:1 on the surface, every check PASS.
  Grade lines carry direct labels and the grades are tabulated in the
  workbook, which is the relief the sub-3:1 light end requires.

These are print/PDF plates on a light bench surface; there is no dark variant
by design (a PDF has one background). The heatmap ramp is the one sequential
scale, light lamp -> hot metal, so magnitude reads as light -> dark.
"""

from __future__ import annotations

# --- chrome: the bench, the part, the ink ---------------------------------
SURFACE = "#f7f6f3"  # plate background: light machined neutral under a lamp
PANEL = "#eceae3"    # callout / caption box fill
INK = "#17140f"      # primary text
INK_MUTED = "#57534a"  # secondary text, axis labels, tick marks
GRID = "#d9d5cb"     # gridlines, dividers (recessive machined grain)

# --- identity: material tones ---------------------------------------------
STEEL = "#1a6ea8"   # machined blue: the part, the monitored series
PATINA = "#0b7d55"  # calm green: in control, restored, pass
COPPER = "#9c5a10"  # third material tone (third method)

# --- signal: what the lamp is for -----------------------------------------
LAMP = "#bb8012"    # inspection-lamp amber: caution, change point, the trap
SIGNAL = "#9e2a1c"  # signal red: alarm, escape, the cost you act on

# --- severity grades: an ordinal warm ramp ending on the signal red --------
GRADE_COLORS = {
    "minor": "#e59a5a",
    "major": "#c05f22",
    "critical": SIGNAL,
}

# Single-hue sequential ramp for anomaly heatmaps: lamp-lit metal -> hot mark.
HEAT_RAMP = ("#fbf3e6", "#efc98d", "#dd9440", "#c05f22", "#7d2b12")
# Ground-truth masks: material where the part is clean, signal red where it is not.
MASK_COLORS = ("#e4e0d6", SIGNAL)
