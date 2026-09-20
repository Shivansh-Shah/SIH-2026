"""Evaluate the trained model against bicubic on the geographic test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw
from rasterio.transform import Affine
from skimage.metrics import structural_similarity
from torch.nn import functional as F

from model import CompactSR4x


SCALE_VALUE = 10000.0
BAND_NAMES = ["B04 red", "B03 green", "B02 blue", "B08 NIR"]


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    error = prediction - target
    mse = float(np.mean(error ** 2))
    mae = float(np.mean(np.abs(error)))
    psnr = float(-10.0 * np.log10(max(mse, 1e-12)))
    ssim_values = [
        structural_similarity(target[i], prediction[i], data_range=1.0)
        for i in range(4)
    ]
    # Mean spectral angle, excluding nearly black pixels.
    dot = np.sum(prediction * target, axis=0)
    norm = np.linalg.norm(prediction, axis=0) * np.linalg.norm(target, axis=0)
    valid = norm > 1e-8
    angles = np.arccos(np.clip(dot[valid] / norm[valid], -1.0, 1.0))
    sam_degrees = float(np.degrees(np.mean(angles))) if angles.size else float("nan")
    return {"mae": mae, "rmse": float(np.sqrt(mse)), "psnr": psnr,
            "ssim": float(np.mean(ssim_values)), "sam_degrees": sam_degrees}


def rgb(array: np.ndarray) -> np.ndarray:
    # Dataset order is R, G, B, NIR. Percentile stretch is display-only.
    image = np.moveaxis(array[:3], 0, -1)
    low, high = np.percentile(image, [2, 98])
    image = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
    return (image * 255).astype(np.uint8)


def comparison_png(lr: np.ndarray, bicubic: np.ndarray, prediction: np.ndarray,
                   target: np.ndarray, output: Path) -> None:
    lr_up = np.asarray(Image.fromarray(rgb(lr)).resize((520, 520), Image.Resampling.NEAREST))
    panels = [lr_up, rgb(bicubic), rgb(prediction), rgb(target)]
    labels = ["Sentinel-2 10 m", "Bicubic 2.5 m grid", "Model 2.5 m grid", "NAIP reference 2.5 m"]
    canvas = Image.new("RGB", (520 * 4, 565), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        canvas.paste(Image.fromarray(panel), (index * 520, 45))
        draw.text((index * 520 + 10, 14), label, fill="black")
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("work/data/crosssensor_subset"))
    parser.add_argument("--model", type=Path, default=Path("outputs/sentinel2_sr_model/best_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sentinel2_sr_model"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    model = CompactSR4x().eval()
    model.load_state_dict(checkpoint["model"])
    rows = []
    sample_payload = None

    for path in sorted((args.data / "test").glob("*.npz")):
        with np.load(path) as item:
            lr = np.clip(item["lr"].astype(np.float32) / SCALE_VALUE, 0, 1)
            hr = np.clip(item["hr"].astype(np.float32) / SCALE_VALUE, 0, 1)
            transform = tuple(item["hr_transform"].tolist())
            crs = str(item["hr_crs"])
        lr_tensor = torch.from_numpy(lr).unsqueeze(0)
        with torch.no_grad():
            bicubic = F.interpolate(lr_tensor, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
            prediction = model(lr_tensor)
        bicubic_np = bicubic.squeeze(0).numpy()
        prediction_np = prediction.squeeze(0).numpy()
        for method, array in [("bicubic", bicubic_np), ("model", prediction_np)]:
            row = {"sample": path.stem, "method": method, **metrics(array, hr)}
            for band, name in enumerate(BAND_NAMES):
                row[f"mae_{name.split()[0]}"] = float(np.mean(np.abs(array[band] - hr[band])))
            rows.append(row)
        if sample_payload is None:
            sample_payload = (path.stem, lr, bicubic_np, prediction_np, hr, transform, crs)

    with (args.output / "test_metrics_per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for method in ["bicubic", "model"]:
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in ["mae", "rmse", "psnr", "ssim", "sam_degrees",
                        "mae_B04", "mae_B03", "mae_B02", "mae_B08"]
        }
    summary["improvement"] = {
        "psnr_db": summary["model"]["psnr"] - summary["bicubic"]["psnr"],
        "ssim": summary["model"]["ssim"] - summary["bicubic"]["ssim"],
        "mae_percent": 100.0 * (summary["bicubic"]["mae"] - summary["model"]["mae"])
        / summary["bicubic"]["mae"],
    }
    (args.output / "test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if sample_payload:
        name, lr, bicubic, prediction, hr, transform, crs = sample_payload
        comparison_png(lr, bicubic, prediction, hr, args.output / "comparison.png")
        profile = {
            "driver": "GTiff", "height": 520, "width": 520, "count": 4,
            "dtype": "float32", "crs": crs,
            "transform": Affine(*transform), "compress": "deflate",
        }
        with rasterio.open(args.output / "example_superresolved_2_5m.tif", "w", **profile) as dst:
            dst.write(prediction.astype(np.float32))
            dst.descriptions = tuple(BAND_NAMES)
            dst.update_tags(
                source_sample=name,
                model="CompactSR4x",
                warning="2.5 m output grid; fine detail is model-inferred, not directly observed",
            )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
