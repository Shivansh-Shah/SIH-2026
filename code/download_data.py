"""Download a small, geographically split SEN2NAIPv2 cross-sensor subset.

The source archive is a 9.7 GB TACO file. This script downloads only the
byte ranges for selected LR/HR GeoTIFF assets, avoiding a full archive copy.
"""

from __future__ import annotations

import argparse
import csv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import tacoreader.v1 as tacoreader
from rasterio.io import MemoryFile


ARCHIVE_URL = (
    "https://huggingface.co/datasets/tacofoundation/SEN2NAIPv2/resolve/main/"
    "sen2naipv2-crosssensor.taco"
)
TRAIN_STATES = [
    "Texas", "North Dakota", "Colorado", "South Dakota",
    "Montana", "Arizona", "Oregon", "Oklahoma",
]
VAL_STATES = ["California"]
TEST_STATES = ["Florida"]


def parse_range(path: str) -> tuple[int, int]:
    match = re.search(r"/vsisubfile/(\d+)_(\d+),", path)
    if not match:
        raise ValueError(f"Cannot parse byte range from {path}")
    return int(match.group(1)), int(match.group(2))


def fetch_asset(path: str) -> tuple[np.ndarray, tuple[float, ...], str]:
    offset, length = parse_range(path)
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
    last_error: Exception | None = None
    for _ in range(4):
        try:
            response = requests.get(ARCHIVE_URL, headers=headers, timeout=90)
            response.raise_for_status()
            if len(response.content) != length:
                raise IOError(f"Expected {length} bytes, received {len(response.content)}")
            with MemoryFile(response.content) as mem:
                with mem.open() as src:
                    return src.read(), tuple(src.transform)[:6], str(src.crs)
        except Exception as exc:  # retry transient range/network failures
            last_error = exc
    raise RuntimeError(f"Failed byte-range download after retries: {last_error}")


def choose_rows(frame, states: list[str], count: int, seed: int):
    candidates = frame[
        frame["rai:admin1"].isin(states)
        & (frame["correlation"].astype(float) >= 0.75)
    ].copy()
    # Stable random choice so a rerun has identical geographic samples.
    return candidates.sample(n=min(count, len(candidates)), random_state=seed)


def download_one(dataset, row_index: int, split: str, output_dir: Path) -> dict:
    sample = dataset.read(row_index)
    lr, lr_transform, lr_crs = fetch_asset(sample.read(0))
    hr, hr_transform, hr_crs = fetch_asset(sample.read(1))
    if lr.shape != (4, 130, 130) or hr.shape != (4, 520, 520):
        raise ValueError(f"Unexpected shapes LR={lr.shape}, HR={hr.shape}")
    row = dataset.loc[row_index]
    sample_id = str(row["tortilla:id"])
    target = output_dir / split / f"{sample_id}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        lr=lr,
        hr=hr,
        lr_transform=np.asarray(lr_transform),
        hr_transform=np.asarray(hr_transform),
        lr_crs=lr_crs,
        hr_crs=hr_crs,
    )
    return {
        "split": split,
        "sample_id": sample_id,
        "state": str(row["rai:admin1"]),
        "county": str(row["rai:admin2"]),
        "centroid": str(row["stac:centroid"]),
        "correlation": float(row["correlation"]),
        "days_between": int(row["days_between"]),
        "path": str(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("work/data/crosssensor_subset"))
    parser.add_argument("--train", type=int, default=120)
    parser.add_argument("--val", type=int, default=20)
    parser.add_argument("--test", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    dataset = tacoreader.load("tacofoundation:sen2naipv2-crosssensor")
    selected = []
    for split, states, count, seed in [
        ("train", TRAIN_STATES, args.train, 2026),
        ("val", VAL_STATES, args.val, 2027),
        ("test", TEST_STATES, args.test, 2028),
    ]:
        rows = choose_rows(dataset, states, count, seed)
        selected.extend((int(index), split) for index in rows.index)

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, dataset, index, split, args.output): (index, split)
            for index, split in selected
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            index, split = futures[future]
            record = future.result()
            records.append(record)
            print(f"[{completed:03d}/{len(futures)}] {split}: {record['sample_id']}", flush=True)

    records.sort(key=lambda item: (item["split"], item["sample_id"]))
    manifest = args.output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} pairs and manifest to {args.output}")


if __name__ == "__main__":
    main()
