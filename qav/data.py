"""Seeded synthetic surface-texture generator with defect injection.

Everything here is procedurally generated -- there is no real camera data in
this project. The "surface" is a 64x64 grayscale texture that combines a woven
sinusoidal pattern with horizontally brushed noise (think brushed metal with a
faint weave). Three defect kinds are injected on top of clean textures:

- ``scratch``        thin bright or dark line segment at a random angle
- ``blob``           local Gaussian bump or pit (bright or dark)
- ``texture_break``  a patch where the pattern is rotated and shifted, so the
                     local *structure* changes while brightness statistics stay
                     roughly the same

Every defective image carries a ground-truth boolean mask of the pixels that
actually changed, so localization (not just image-level scoring) can be
evaluated. All randomness flows through one ``numpy.random.Generator`` seeded
from ``DataConfig.seed``: the same config always produces bit-identical arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

DEFECT_KINDS = ("scratch", "blob", "texture_break")

# A pixel belongs to the ground-truth defect mask when injection moved it by
# more than this (images live in [0, 1]).
MASK_DELTA_THRESHOLD = 0.02


@dataclass(frozen=True)
class DataConfig:
    """Dataset shape and seed. Tests use tiny counts; the study uses defaults."""

    n_train: int = 600
    n_test_clean: int = 150
    n_test_defective: int = 150  # split as evenly as possible across DEFECT_KINDS
    image_size: int = 64
    seed: int = 7


@dataclass
class SurfaceDataset:
    """Arrays for one generated study.

    ``test_types[i]`` is ``"clean"`` or one of ``DEFECT_KINDS``; ``test_masks``
    is all-False for clean images.
    """

    train: np.ndarray  # (n_train, S, S) float32 in [0, 1], all clean
    test_images: np.ndarray  # (n_test, S, S) float32
    test_labels: np.ndarray  # (n_test,) int, 0 = clean, 1 = defective
    test_types: list[str]
    test_masks: np.ndarray  # (n_test, S, S) bool
    config: DataConfig


def _base_texture(rng: np.random.Generator, size: int) -> np.ndarray:
    """One clean surface: fixed texture family, per-image random phase/noise."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    # Woven component: two crossed sinusoids. Frequencies are near-fixed for
    # the whole product family (same "machine settings"), phases are random.
    fx = 6.0 * (1.0 + rng.uniform(-0.04, 0.04))
    fy = 9.0 * (1.0 + rng.uniform(-0.04, 0.04))
    p1, p2 = rng.uniform(0.0, 2.0 * np.pi, size=2)
    weave = np.sin(2.0 * np.pi * fx * xx / size + p1) * np.sin(2.0 * np.pi * fy * yy / size + p2)
    # Brushed component: white noise smeared horizontally (machine direction).
    noise = rng.standard_normal((size, size))
    brushed = ndimage.gaussian_filter(noise, sigma=(0.6, 5.0))
    brushed = brushed / (brushed.std() + 1e-9)
    grain = rng.standard_normal((size, size))
    img = 0.5 + 0.10 * weave + 0.06 * brushed + 0.015 * grain
    return np.clip(img, 0.02, 0.98).astype(np.float32)


def _segment_distance(size: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel Euclidean distance to the segment a-b (both are (y, x))."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    p = np.stack([yy, xx], axis=-1)
    ab = b - a
    denom = float(ab @ ab) + 1e-12
    t = np.clip(((p - a) @ ab) / denom, 0.0, 1.0)
    nearest = a + t[..., None] * ab
    return np.sqrt(((p - nearest) ** 2).sum(axis=-1))


def _inject_scratch(
    image: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    size = image.shape[0]
    center = rng.uniform(14.0, size - 14.0, size=2)
    angle = rng.uniform(0.0, np.pi)
    length = rng.uniform(14.0, 34.0)
    direction = np.array([np.sin(angle), np.cos(angle)])
    a = center - 0.5 * length * direction
    b = center + 0.5 * length * direction
    dist = _segment_distance(size, a, b)
    width = rng.uniform(0.6, 1.1)
    sign = -1.0 if rng.uniform() < 0.5 else 1.0
    amp = sign * rng.uniform(0.18, 0.38)
    out = np.clip(image + amp * np.exp(-((dist / width) ** 2)), 0.0, 1.0)
    return out.astype(np.float32), np.abs(out - image) > MASK_DELTA_THRESHOLD


def _inject_blob(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    size = image.shape[0]
    cy, cx = rng.uniform(10.0, size - 10.0, size=2)
    sigma = rng.uniform(1.8, 3.4)
    sign = -1.0 if rng.uniform() < 0.5 else 1.0
    amp = sign * rng.uniform(0.16, 0.32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    bump = np.exp(-(((yy - cy) ** 2) + ((xx - cx) ** 2)) / (2.0 * sigma**2))
    out = np.clip(image + amp * bump, 0.0, 1.0)
    return out.astype(np.float32), np.abs(out - image) > MASK_DELTA_THRESHOLD


def _inject_texture_break(
    image: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate + shift a square patch of the pattern, blended with a soft window.

    Because the base texture is strongly oriented, a rotated patch breaks the
    structure even though its brightness histogram barely changes. Rarely a
    draw lands where the change is tiny, so a few relocations are attempted and
    the attempt with the largest mask is kept (all rng-driven, deterministic).
    """
    size = image.shape[0]
    best: tuple[np.ndarray, np.ndarray] | None = None
    for _ in range(6):
        half = int(rng.integers(9, 14))  # patch side 18..26 px
        cy, cx = (int(v) for v in rng.integers(half + 2, size - half - 2, size=2))
        k = int(rng.integers(1, 4))
        shift = int(rng.integers(1, half))
        patch = image[cy - half : cy + half, cx - half : cx + half]
        replaced = np.roll(np.rot90(patch, k=k), shift=shift, axis=int(rng.integers(0, 2)))
        py, px = np.mgrid[0:2 * half, 0:2 * half].astype(np.float64)
        r = np.sqrt((py - half + 0.5) ** 2 + (px - half + 0.5) ** 2)
        window = np.exp(-((r / (0.68 * half)) ** 4))  # flat core, soft rim
        blended = patch * (1.0 - window) + replaced * window
        out = image.copy()
        out[cy - half : cy + half, cx - half : cx + half] = blended
        out = np.clip(out, 0.0, 1.0).astype(np.float32)
        mask = np.abs(out - image) > MASK_DELTA_THRESHOLD
        if best is None or mask.sum() > best[1].sum():
            best = (out, mask)
        if mask.sum() >= 40:
            break
    assert best is not None
    return best


_INJECTORS = {
    "scratch": _inject_scratch,
    "blob": _inject_blob,
    "texture_break": _inject_texture_break,
}


def inject_defect(
    image: np.ndarray, kind: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return (defective_image, ground_truth_mask) for one defect kind."""
    if kind not in _INJECTORS:
        raise ValueError(f"unknown defect kind {kind!r}; expected one of {DEFECT_KINDS}")
    return _INJECTORS[kind](image, rng)


def make_dataset(config: DataConfig | None = None) -> SurfaceDataset:
    """Generate the full study dataset deterministically from ``config.seed``.

    Test images are ordered: all clean first, then defect kinds in
    ``DEFECT_KINDS`` order. Evaluation is order-independent, so no shuffle is
    needed (and determinism stays easy to audit).
    """
    cfg = config or DataConfig()
    rng = np.random.default_rng(cfg.seed)
    size = cfg.image_size

    train = np.stack([_base_texture(rng, size) for _ in range(cfg.n_train)])
    test_clean = np.stack([_base_texture(rng, size) for _ in range(cfg.n_test_clean)])

    per_kind = cfg.n_test_defective // len(DEFECT_KINDS)
    remainder = cfg.n_test_defective - per_kind * len(DEFECT_KINDS)
    counts = [per_kind + (1 if i < remainder else 0) for i in range(len(DEFECT_KINDS))]

    defective_images: list[np.ndarray] = []
    defective_masks: list[np.ndarray] = []
    defective_types: list[str] = []
    for kind, count in zip(DEFECT_KINDS, counts, strict=True):
        for _ in range(count):
            base = _base_texture(rng, size)
            img, mask = inject_defect(base, kind, rng)
            defective_images.append(img)
            defective_masks.append(mask)
            defective_types.append(kind)

    n_def = len(defective_images)
    test_images = np.concatenate([test_clean, np.stack(defective_images)]) if n_def else test_clean
    test_labels = np.concatenate(
        [np.zeros(cfg.n_test_clean, dtype=np.int64), np.ones(n_def, dtype=np.int64)]
    )
    test_masks = np.concatenate(
        [
            np.zeros((cfg.n_test_clean, size, size), dtype=bool),
            np.stack(defective_masks) if n_def else np.zeros((0, size, size), dtype=bool),
        ]
    )
    test_types = ["clean"] * cfg.n_test_clean + defective_types
    return SurfaceDataset(
        train=train,
        test_images=test_images,
        test_labels=test_labels,
        test_types=test_types,
        test_masks=test_masks,
        config=cfg,
    )
