# Dataset Card: Launch Site Detection Tiles

## Overview

Tiled satellite imagery chips for training and evaluating launch-site segmentation models.

## Sources

- **Sentinel-2 L2A** (optical, 10m GSD): Bands B02, B03, B04, B08, B11, B12
- **Sentinel-1 RTC** (SAR, 10m GSD): VV and VH polarizations (v2 fusion)
- **Known sites**: 18 publicly documented launch sites / spaceports worldwide

## Data Format

- **Tiles**: 256x256 px GeoTIFF chips (10-band: 6 S2 bands + 4 spectral indices)
- **Labels**: Weak binary masks from buffered known site coordinates
- **Splits**: 70/15/15 train/val/test with stratified positive/negative balance

## Coverage

AOIs include Cape Canaveral, Vandenberg, Baikonur, Boca Chica, and more.
Composites built from 30-90 day median stacks to reduce cloud cover.

## Preprocessing

1. Cloud masking via Sentinel-2 SCL band
2. Percentile normalization (2nd-98th) per band
3. Spectral index computation (NDVI, NDWI, NDBI, BSI)
4. Temporal compositing (median)

## Labeling Strategy

- **Positives**: Circular buffers (1-5 km radius) around known launch site coordinates
- **Negatives**: Random tiles >50 km from any known site
- **Confusers**: Tiles over industrial/quarry/mining areas (optional hard negatives)
- **Label smoothing**: Applied at loss level (not in masks) to handle boundary noise

## Known Issues

- Buffer-based labels are coarse; actual site footprints may be smaller or offset
- Some known sites may have limited Sentinel-2 coverage (high latitudes, tropical cloud cover)
- Negative sampling may accidentally include undocumented sites
