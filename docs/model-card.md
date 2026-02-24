# Model Card: Launch Site Segmentation

## Model Details

- **Architecture**: UNet / DeepLabV3+ (configurable)
- **Encoder**: ResNet-34 (baseline), ResNet-50 (strong)
- **Input**: 10-band Sentinel-2 composite (RGB + NIR + SWIR16 + SWIR22 + NDVI + NDWI + NDBI + BSI)
- **Output**: Single-channel probability mask (launch-site likelihood per pixel)
- **Resolution**: 256x256 tiles at 10m GSD (~2.56 km x 2.56 km)

## Training Data

- **Positives**: Weak labels from buffered known launch site coordinates (18 sites)
- **Negatives**: Tiles far from known sites + confuser regions (quarries, industrial)
- **Composites**: Median over 30-90 day windows from Sentinel-2 L2A
- **Preprocessing**: Cloud masking (SCL), percentile normalization, spectral indices

## Loss Function

Combined focal loss + soft dice loss with label smoothing (0.05).
Focal loss (alpha=0.25, gamma=2.0) handles extreme class imbalance.

## Evaluation Metrics

- **Site-level recall@K**: Fraction of known sites in top-K predictions
- **FP rate per 1000 km²**: False positive density control
- **Calibration**: Confidence vs precision across score bins

## Limitations

- 10m resolution cannot detect small pads; detects compound-level structures
- Weak labels introduce noise; model may miss unusual site layouts
- Temporal composites may blur active construction
- SAR fusion (v2) not included in baseline

## Intended Use

Research / OSINT analysis of satellite imagery for launch infrastructure identification.
Not intended for operational military intelligence.
