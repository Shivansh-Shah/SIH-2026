"""Download and evaluate ten new Georgia pairs, exporting every result."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
import tacoreader.v1 as tacoreader
import torch
from PIL import Image
from rasterio.transform import Affine
from torch.nn import functional as F

from download_data import download_one
from evaluate import BAND_NAMES, comparison_png, metrics, rgb
from model import CompactSR4x


DATA_ROOT = Path("work/data/additional_10")
OUTPUT_ROOT = Path("outputs/sentinel2_sr_model/additional_10_results")
CHECKPOINT = Path("outputs/sentinel2_sr_model/best_model.pt")
SCALE_VALUE = 10000.0


def download_new_pairs() -> list[dict]:
    dataset = tacoreader.load("tacofoundation:sen2naipv2-crosssensor")
    original_manifest = Path("outputs/sentinel2_sr_model/split_manifest.csv")
    existing_ids = set()
    with original_manifest.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            existing_ids.add(row["sample_id"])

    candidates = dataset[
        (dataset["rai:admin1"] == "Georgia")
        & (dataset["correlation"].astype(float) >= 0.75)
        & (~dataset["tortilla:id"].isin(existing_ids))
    ].sample(n=10, random_state=3030)

    records = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(download_one, dataset, int(index), "test", DATA_ROOT): int(index)
            for index in candidates.index
        }
        for position, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            print(f"Downloaded {position}/10: {record['sample_id']}", flush=True)
    records.sort(key=lambda item: item["sample_id"])
    return records


def save_manifest(records: list[dict]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "additional_test_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    records = download_new_pairs()
    save_manifest(records)
    comparisons = OUTPUT_ROOT / "comparisons"
    geotiffs = OUTPUT_ROOT / "geotiffs"
    model_pngs = OUTPUT_ROOT / "model_rgb"
    for folder in (comparisons, geotiffs, model_pngs):
        folder.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = CompactSR4x().eval()
    model.load_state_dict(checkpoint["model"])
    rows = []
    comparison_paths = []

    for position, path in enumerate(sorted((DATA_ROOT / "test").glob("*.npz")), start=1):
        with np.load(path) as item:
            lr = np.clip(item["lr"].astype(np.float32) / SCALE_VALUE, 0, 1)
            hr = np.clip(item["hr"].astype(np.float32) / SCALE_VALUE, 0, 1)
            transform = tuple(item["hr_transform"].tolist())
            crs = str(item["hr_crs"])
        source = torch.from_numpy(lr).unsqueeze(0)
        with torch.no_grad():
            bicubic = F.interpolate(
                source, scale_factor=4, mode="bicubic", align_corners=False
            ).clamp(0, 1).squeeze(0).numpy()
            prediction = model(source).squeeze(0).numpy()

        result = {"sample": path.stem}
        for prefix, array in (("bicubic", bicubic), ("model", prediction)):
            for key, value in metrics(array, hr).items():
                result[f"{prefix}_{key}"] = value
        rows.append(result)

        comparison_path = comparisons / f"{position:02d}_{path.stem}.png"
        comparison_png(lr, bicubic, prediction, hr, comparison_path)
        comparison_paths.append(comparison_path)
        Image.fromarray(rgb(prediction)).save(model_pngs / f"{position:02d}_{path.stem}.png")

        profile = {
            "driver": "GTiff", "height": 520, "width": 520, "count": 4,
            "dtype": "float32", "crs": crs, "transform": Affine(*transform),
            "compress": "deflate",
        }
        with rasterio.open(geotiffs / f"{position:02d}_{path.stem}.tif", "w", **profile) as dst:
            dst.write(prediction.astype(np.float32))
            dst.descriptions = tuple(BAND_NAMES)
            dst.update_tags(
                model="CompactSR4x",
                warning="2.5 m output grid; fine detail is model-inferred, not directly observed",
            )
        print(f"Evaluated {position}/10: {path.stem}", flush=True)

    with (OUTPUT_ROOT / "metrics_per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metric_names = ["mae", "rmse", "psnr", "ssim", "sam_degrees"]
    summary = {
        method: {
            metric: float(np.mean([row[f"{method}_{metric}"] for row in rows]))
            for metric in metric_names
        }
        for method in ("bicubic", "model")
    }
    summary["improvement"] = {
        "mae_percent": 100 * (summary["bicubic"]["mae"] - summary["model"]["mae"])
        / summary["bicubic"]["mae"],
        "psnr_db": summary["model"]["psnr"] - summary["bicubic"]["psnr"],
        "ssim": summary["model"]["ssim"] - summary["bicubic"]["ssim"],
        "sam_degrees": summary["model"]["sam_degrees"] - summary["bicubic"]["sam_degrees"],
    }
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    thumbs = []
    for path in comparison_paths:
        with Image.open(path) as image:
            thumb = image.copy()
            thumb.thumbnail((1040, 283))
            thumbs.append(thumb)
    sheet = Image.new("RGB", (1040, 283 * len(thumbs)), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, (0, index * 283))
    sheet.save(OUTPUT_ROOT / "all_10_contact_sheet.jpg", quality=90)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
