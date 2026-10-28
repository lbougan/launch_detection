# Model Card: Launch Site Segmentation

## Model Details

- **Architecture**: UNet (default) or DeepLabV3+, from `segmentation-models-pytorch`
- **Encoder**: ResNet-34 with ImageNet weights (default); any smp encoder via `--encoder`
- **Input**: 10 channels, 256 x 256 px at 10 m: Sentinel-2 L2A reflectance (blue, green, red,
  NIR, SWIR 1.6, SWIR 2.2) plus NDVI, NDWI, NDBI, BSI computed from reflectance
- **Output**: one logit per pixel; sigmoid gives the launch-site probability
- **Framework**: PyTorch Lightning; ONNX export for NVIDIA Triton

## Input Normalisation

One function, `libs.features.indices.normalize(array, stats)`, is used in training, evaluation,
local inference and (through the stats file) Triton inference. Per channel: clip to the
channel's p2/p98 range scaled to [0, 1], then subtract the mean and divide by the standard
deviation. The statistics live in `data/manifests/channel_stats.json`, computed from a pixel
sample of the training-AOI composites only (cloud and nodata pixels excluded). The ONNX export
embeds the stats and the band spec in its metadata and copies the stats file next to the model.

## Training Data

- **Labels**: weak labels from 18 known launch-site coordinates: 1 inside the site's buffer
  radius, 0.5 (ignore, loss weight 0) in the ring out to twice the radius, 0 beyond; drawn in
  metres. See the [dataset card](dataset-card.md).
- **Composites**: per-pixel NaN-median of SCL-cloud-masked reflectance over the date range
  (default 2024-01-01 to 2024-06-30).
- **Split**: by site, with a 20 km exclusion zone; the model never sees the validation or test
  sites during training.

## Loss Function

Focal loss (alpha 0.25, gamma 2.0) plus 0.5 x soft Dice, both weighted per pixel so the ignore
ring contributes nothing. No label smoothing. AdamW (lr 1e-3, weight decay 1e-4),
ReduceLROnPlateau (patience 5, factor 0.5), random flips and 90° rotations.

## Model Selection

The checkpoint monitor is `val/site_recall@10` on the validation AOIs' composites (Xichang,
Plesetsk, Sohae), with early stopping (patience 10) on the same metric. Without validation
composites it falls back to `val/loss`.

## Evaluation Metrics

Evaluated on held-out sites only. Validation: Xichang, Plesetsk, Sohae. Test: Satish Dhawan,
Tanegashima, Guiana. A detection matches a site when its centroid is within 5 km.

- **Site recall@K** (K = 5, 10, 20): fraction of the split's sites matched by one of the K
  highest-ranked detections (by `rank_score`)
- **FP per 1000 km²**: unmatched detections per 1000 km² of scanned composite
- **Reliability**: detections binned by `prob` (not `rank_score`), fraction matching a site per bin

```bash
uv run python -m services.training.evaluate --checkpoint checkpoints/<best>.ckpt \
  --split val --split test --update-readme README.md
```

No real-data results yet; they are produced on the first GPU run (`scripts/gpu_day.sh`). The
synthetic end-to-end test (`tests/integration/test_synthetic_pipeline.py`) recovers a planted site
at recall@5 = 1.0, which checks the plumbing, not skill on real imagery.

## Limitations

- 10 m resolution cannot resolve small pads; the model responds to compound-scale structure
- Circular weak labels are a coarse proxy for irregular footprints
- Six held-out sites give coarse recall estimates (each site is 1/3 of a split)
- Temporal composites may blur active construction
- Sentinel-1 is not used

## Intended Use

Research / OSINT analysis of satellite imagery for launch infrastructure identification.
Not intended for operational military intelligence.
