# Sentinel-2 RGB+NIR 4× Super-Resolution Prototype

This is a trained, CPU-scale prototype that converts four native Sentinel-2
10 m bands (B04, B03, B02, B08) to a four-band product on a 2.5 m output grid.
Fine detail in the result is model-inferred and is not a direct 2.5 m
observation by Sentinel-2.

## What was trained

- Dataset: SEN2NAIPv2 real cross-sensor subset (Sentinel-2 LR + NAIP HR)
- Source archive: `tacofoundation/SEN2NAIPv2`, cross-sensor variant
- Quality filter: pair correlation >= 0.75
- Geographic split:
  - Train: 120 pairs from Texas, North Dakota, Colorado, South Dakota,
    Montana, Arizona, Oregon and Oklahoma
  - Validation: 20 pairs from California
  - Test: 20 untouched pairs from Florida
- Model: compact EDSR-style residual CNN
- Input: 4 × 130 × 130 pixels at 10 m for full-scene evaluation
- Output: 4 × 520 × 520 pixels on a 2.5 m grid
- Train crops: 4 × 32 × 32 to 4 × 128 × 128
- Architecture: 32 feature channels, 6 residual blocks, two PixelShuffle stages
- Training: 8 epochs on CPU; best checkpoint selected by California validation loss
- Loss: L1 reconstruction + gradient loss + downsample-to-input consistency loss

## Untouched Florida test results

| Metric | Bicubic | Trained model | Change |
|---|---:|---:|---:|
| MAE | 0.013511 | 0.013323 | 1.395% lower |
| RMSE | 0.021398 | 0.021076 | lower |
| PSNR | 34.1039 dB | 34.2203 dB | +0.1164 dB |
| SSIM | 0.830020 | 0.830636 | +0.000616 |
| Spectral angle | 3.0679° | 3.0153° | 0.0526° lower |

The model beat bicubic on every aggregate metric, but the gain is modest. The
comparison image also shows that this small reconstruction model mostly
preserves and slightly corrects the bicubic result; it does not recover all of
the fine structure visible in the independent NAIP reference.

## Files

- `best_model.pt`: trained PyTorch checkpoint
- `comparison.png`: Sentinel-2, bicubic, model and reference side-by-side
- `example_superresolved_2_5m.tif`: four-band georeferenced example output
- `test_summary.json`: aggregate test metrics
- `test_metrics_per_image.csv`: metrics for each test image and method
- `training_history.json`: per-epoch training/validation history
- `split_manifest.csv`: selected sample IDs, states, coordinates and quality values
- `code/`: download, model, training and evaluation source
- `additional_10_results/`: ten additional unseen Georgia tests, including
  comparison PNGs, model RGB previews and georeferenced GeoTIFF outputs

## Reproduction

Install the dependencies in `requirements.txt`, then run from this project root:

```powershell
python code/download_data.py
python code/train.py
python code/evaluate.py
```

The scripts default to `work/data/crosssensor_subset` for cached data and
`outputs/sentinel2_sr_model` for artifacts. The range downloader retrieves
about 160 selected pairs rather than the complete 9.7 GB archive.

To reproduce the separate ten-image Georgia evaluation, run:

```powershell
python code/test_additional_10.py
```

## Scientific limitations

This is a small proof of concept, not a production model. It was trained on
United States imagery and tested on only 20 Florida locations. NAIP and
Sentinel-2 are different sensors, and residual spatial/radiometric mismatch is
possible even after dataset harmonization. The output has a 2.5 m pixel grid,
but that does not establish a universal 2.5 m effective resolving power.
Before operational use, train on substantially more geographic and seasonal
coverage, add uncertainty calibration, evaluate downstream tasks, and validate
on independent high-resolution imagery from the intended deployment region.
