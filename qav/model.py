"""Small convolutional autoencoder anomaly detector (PyTorch, CPU).

Trained on CLEAN images only with plain MSE reconstruction loss. At test time
the per-pixel squared reconstruction error is the defect heatmap (the model
never saw defects, so it reconstructs them poorly -- ideally), and the image
score is the same hottest-2% rule used by the classical baselines.

Determinism, honestly stated: ``torch.manual_seed`` is set, shuffling uses a
seeded numpy generator, the device is CPU by default, and
``torch.use_deterministic_algorithms(True, warn_only=True)`` is requested
(``warn_only`` so ops without a deterministic CPU kernel warn instead of
crashing). Repeated runs on the same machine and torch build reproduce
bit-identical metrics (verified during development); across torch versions or
hardware, low-order floating-point differences are possible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage
from torch import nn

from qav.baselines import heatmap_image_scores


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 15
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 7
    device: str = "cpu"  # CPU on purpose: small model, reproducible, no GPU needed
    smooth_sigma: float = 1.0


class ConvAutoencoder(nn.Module):
    """3 stride-2 convs down (64 -> 8 px), bottleneck, 3 transposed convs up."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),  # bottleneck: 8x8x32 = 2048 values
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1),  # 32 -> 64
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class ConvAutoencoderDetector:
    """fit / heatmaps / scores protocol matching the classical baselines."""

    def __init__(self, config: TrainConfig | None = None):
        self.config = config or TrainConfig()
        self.model: ConvAutoencoder | None = None
        self.history: dict = {}

    def fit(self, train_images: np.ndarray) -> ConvAutoencoderDetector:
        cfg = self.config
        torch.manual_seed(cfg.seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            deterministic_requested = True
        except (AttributeError, RuntimeError):  # very old torch builds
            deterministic_requested = False

        device = torch.device(cfg.device)
        model = ConvAutoencoder().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        loss_fn = nn.MSELoss()
        data = torch.from_numpy(train_images.astype(np.float32)).unsqueeze(1).to(device)

        shuffle_rng = np.random.default_rng(cfg.seed)
        epoch_losses: list[float] = []
        model.train()
        for _ in range(cfg.epochs):
            order = shuffle_rng.permutation(len(data))
            total, batches = 0.0, 0
            for start in range(0, len(data), cfg.batch_size):
                batch = data[order[start : start + cfg.batch_size]]
                optimizer.zero_grad()
                loss = loss_fn(model(batch), batch)
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                batches += 1
            epoch_losses.append(total / max(batches, 1))

        self.model = model
        self.history = {
            "epoch_losses": epoch_losses,
            "param_count": count_parameters(model),
            "deterministic_algorithms_requested": deterministic_requested,
            "device": cfg.device,
        }
        return self

    @torch.no_grad()
    def heatmaps(self, images: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit() must be called before heatmaps()")
        self.model.eval()
        device = torch.device(self.config.device)
        x = torch.from_numpy(images.astype(np.float32)).unsqueeze(1).to(device)
        errs: list[np.ndarray] = []
        for start in range(0, len(x), 64):
            batch = x[start : start + 64]
            recon = self.model(batch)
            errs.append(((batch - recon) ** 2).squeeze(1).cpu().numpy())
        err = np.concatenate(errs).astype(np.float64)
        if self.config.smooth_sigma > 0:
            err = np.stack([ndimage.gaussian_filter(e, self.config.smooth_sigma) for e in err])
        return err

    def scores(self, images: np.ndarray) -> np.ndarray:
        return heatmap_image_scores(self.heatmaps(images))
