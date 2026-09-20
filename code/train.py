"""Train CompactSR4x on cached SEN2NAIPv2 pairs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from model import CompactSR4x


SCALE_VALUE = 10000.0


class PairDataset(Dataset):
    def __init__(self, root: Path, split: str, patch: int = 32, repeats: int = 4) -> None:
        self.files = sorted((root / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No samples in {root / split}")
        self.patch = patch
        self.repeats = repeats if split == "train" else 1
        self.training = split == "train"

    def __len__(self) -> int:
        return len(self.files) * self.repeats

    def __getitem__(self, index: int):
        path = self.files[index % len(self.files)]
        with np.load(path) as item:
            lr = item["lr"].astype(np.float32) / SCALE_VALUE
            hr = item["hr"].astype(np.float32) / SCALE_VALUE
        lr = np.clip(lr, 0.0, 1.0)
        hr = np.clip(hr, 0.0, 1.0)
        if self.training:
            max_y = lr.shape[1] - self.patch
            max_x = lr.shape[2] - self.patch
            y = random.randint(0, max_y)
            x = random.randint(0, max_x)
            lr = lr[:, y:y + self.patch, x:x + self.patch]
            hr = hr[:, y * 4:(y + self.patch) * 4, x * 4:(x + self.patch) * 4]
            if random.random() < 0.5:
                lr, hr = lr[:, :, ::-1], hr[:, :, ::-1]
            if random.random() < 0.5:
                lr, hr = lr[:, ::-1, :], hr[:, ::-1, :]
        return torch.from_numpy(lr.copy()), torch.from_numpy(hr.copy()), path.name


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    px = prediction[..., :, 1:] - prediction[..., :, :-1]
    tx = target[..., :, 1:] - target[..., :, :-1]
    py = prediction[..., 1:, :] - prediction[..., :-1, :]
    ty = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)


def combined_loss(prediction: torch.Tensor, target: torch.Tensor, source: torch.Tensor):
    reconstruction = F.l1_loss(prediction, target)
    gradients = gradient_loss(prediction, target)
    consistency = F.l1_loss(
        F.interpolate(prediction, size=source.shape[-2:], mode="area"), source
    )
    total = reconstruction + 0.05 * gradients + 0.10 * consistency
    return total, reconstruction, gradients, consistency


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    sums = {"loss": 0.0, "mae": 0.0, "psnr": 0.0}
    count = 0
    for lr, hr, _ in loader:
        lr, hr = lr.to(device), hr.to(device)
        prediction = model(lr)
        loss, *_ = combined_loss(prediction, hr, lr)
        mse = F.mse_loss(prediction, hr).item()
        sums["loss"] += loss.item()
        sums["mae"] += F.l1_loss(prediction, hr).item()
        sums["psnr"] += -10.0 * np.log10(max(mse, 1e-12))
        count += 1
    return {key: value / count for key, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("work/data/crosssensor_subset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sentinel2_sr_model"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = PairDataset(args.data, "train", patch=32, repeats=4)
    val_set = PairDataset(args.data, "val")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    model = CompactSR4x().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for lr, hr, _ in train_loader:
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(lr)
            loss, *_ = combined_loss(prediction, hr, lr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
        scheduler.step()
        validation = validate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": running / len(train_loader),
            **{f"val_{key}": value for key, value in validation.items()},
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "architecture": {"bands": 4, "channels": 32, "blocks": 6, "scale": 4},
                    "normalization": SCALE_VALUE,
                    "epoch": epoch,
                    "validation": validation,
                },
                args.output / "best_model.pt",
            )

    (args.output / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"Best validation loss: {best_loss:.6f}; device={device}")


if __name__ == "__main__":
    main()
