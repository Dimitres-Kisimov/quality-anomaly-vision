"""Classical anomaly detectors, implemented from scratch (numpy + scipy only).

Both detectors follow the same protocol as the autoencoder in ``qav.model``:

- ``fit(train_images)``  learns from CLEAN images only
- ``heatmaps(images)``   returns a per-pixel anomaly map per image
- ``scores(images)``     returns one anomaly score per image

The image-level score is deliberately the same rule for every method (the mean
of the hottest 2% of heatmap pixels) so the comparison is about the heatmaps,
not about per-method score tuning. Defects are small; a plain mean over 4096
pixels would drown a 60-pixel scratch.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# One shared image-scoring rule for all methods (see module docstring).
SCORE_TOP_QUANTILE = 0.98


def heatmap_image_scores(heatmaps: np.ndarray, top_quantile: float = SCORE_TOP_QUANTILE) -> np.ndarray:
    """Image score = mean of each heatmap's hottest ``1 - top_quantile`` pixels."""
    flat = heatmaps.reshape(heatmaps.shape[0], -1)
    cut = np.quantile(flat, top_quantile, axis=1, keepdims=True)
    hot = flat >= cut
    return (flat * hot).sum(axis=1) / np.maximum(hot.sum(axis=1), 1)


class PCAReconstruction:
    """Pixel-space PCA reconstruction error (numpy SVD, no sklearn).

    Clean textures concentrate in a low-dimensional subspace (the weave phases
    span a couple of sinusoid components each). A defect is whatever the
    subspace cannot reconstruct, so the squared residual is the heatmap.
    """

    def __init__(self, n_components: int = 32, smooth_sigma: float = 1.0):
        self.n_components = n_components
        self.smooth_sigma = smooth_sigma
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None  # (k, n_pixels)
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, train_images: np.ndarray) -> PCAReconstruction:
        x = train_images.reshape(train_images.shape[0], -1).astype(np.float64)
        self.mean_ = x.mean(axis=0)
        centered = x - self.mean_
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        k = min(self.n_components, vt.shape[0])
        self.components_ = vt[:k]
        var = s**2
        self.explained_variance_ratio_ = var[:k] / max(var.sum(), 1e-12)
        return self

    def heatmaps(self, images: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("fit() must be called before heatmaps()")
        shape = images.shape
        x = images.reshape(shape[0], -1).astype(np.float64) - self.mean_
        recon = (x @ self.components_.T) @ self.components_
        residual2 = ((x - recon) ** 2).reshape(shape)
        if self.smooth_sigma > 0:
            residual2 = np.stack(
                [ndimage.gaussian_filter(r, self.smooth_sigma) for r in residual2]
            )
        return residual2

    def scores(self, images: np.ndarray) -> np.ndarray:
        return heatmap_image_scores(self.heatmaps(images))


class LocalStatsDetector:
    """Local mean/std z-scores against the clean-texture distribution.

    For every pixel, a ``window x window`` neighborhood mean and standard
    deviation are computed with ``scipy.ndimage.uniform_filter``. Fitting
    collects the global distribution (mean and spread) of those two statistics
    over all clean training pixels; the heatmap is the per-pixel z-score
    magnitude ``sqrt(z_mean^2 + z_std^2)``.

    This detector sees brightness anomalies (scratches spike the local std,
    blobs shift the local mean) but is close to blind to texture-breaks, whose
    local statistics match the clean texture. That blindness is part of the
    study's findings, not a bug.
    """

    def __init__(self, window: int = 7, smooth_sigma: float = 1.0):
        self.window = window
        self.smooth_sigma = smooth_sigma
        self.stats_: dict[str, float] | None = None

    def _local_maps(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        img = image.astype(np.float64)
        local_mean = ndimage.uniform_filter(img, size=self.window)
        local_sq = ndimage.uniform_filter(img**2, size=self.window)
        local_var = np.maximum(local_sq - local_mean**2, 0.0)
        return local_mean, np.sqrt(local_var)

    def fit(self, train_images: np.ndarray) -> LocalStatsDetector:
        means: list[np.ndarray] = []
        stds: list[np.ndarray] = []
        for img in train_images:
            lm, ls = self._local_maps(img)
            means.append(lm)
            stds.append(ls)
        mean_all = np.concatenate([m.ravel() for m in means])
        std_all = np.concatenate([s.ravel() for s in stds])
        self.stats_ = {
            "mean_mu": float(mean_all.mean()),
            "mean_sigma": float(mean_all.std() + 1e-9),
            "std_mu": float(std_all.mean()),
            "std_sigma": float(std_all.std() + 1e-9),
        }
        return self

    def heatmaps(self, images: np.ndarray) -> np.ndarray:
        if self.stats_ is None:
            raise RuntimeError("fit() must be called before heatmaps()")
        st = self.stats_
        out = np.empty(images.shape, dtype=np.float64)
        for i, img in enumerate(images):
            lm, ls = self._local_maps(img)
            z_mean = (lm - st["mean_mu"]) / st["mean_sigma"]
            z_std = (ls - st["std_mu"]) / st["std_sigma"]
            h = np.sqrt(z_mean**2 + z_std**2)
            out[i] = ndimage.gaussian_filter(h, self.smooth_sigma) if self.smooth_sigma > 0 else h
        return out

    def scores(self, images: np.ndarray) -> np.ndarray:
        return heatmap_image_scores(self.heatmaps(images))
