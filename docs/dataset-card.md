# Dataset Card: Launch Site Detection Chips

## Overview

Sentinel-2 chips and label rasters for training and evaluating launch-site segmentation. Nothing
is redistributed: the dataset is rebuilt from public imagery by
`uv run python -m services.pipeline.flows ingest`.

## Sources

- **Sentinel-2 L2A** (Microsoft Planetary Computer STAC): blue, green, red, NIR (10 m), SWIR 1.6
  and SWIR 2.2 (20 m, resampled to 10 m), plus the SCL band for cloud masking
- **Known sites**: 18 publicly documented launch sites in `data/manifests/known_sites.json`, each
  with a `buffer_km` radius (1 to 5 km)
- **AOIs**: 17 areas in `data/manifests/aoi_config.json`: four large v0.1 regions (Florida coast,
  Vandenberg, Baikonur, South Texas) and 13 compact ones, so that every site has imagery
- Sentinel-1 is not included

## Preprocessing

1. Cloud mask at ingest: pixels outside SCL classes 4 to 7 become NaN
2. Reflectance from digital numbers: DN x 1e-4, offset -0.1 for processing baseline >= 04.00
3. Temporal composite: per-pixel NaN-median; the band order is stored as a `BANDSPEC` tag
4. Spectral indices from reflectance at load time: NDVI, NDWI, NDBI, BSI (10 model channels)
5. Channel statistics (p2, p98, mean, std) from training-AOI composites only, saved to
   `data/manifests/channel_stats.json`; indices and normalisation are applied at load time, chips stay in reflectance

## Data Format

- **Chips**: 256 x 256 px GeoTIFFs (2.56 km at 10 m) with the 6 reflectance bands and their
  `BANDSPEC` tag, 32 px overlap, chips with less than half
  valid pixels dropped
- **Labels**: one GeoTIFF per chip on the same grid, values `{0, 0.5, 1}`
- **Manifest**: `data/manifests/chip_manifest.json` with tile id, AOI, bbox, split, chip and
  label paths

## Labeling Strategy

- **Positive (1)**: inside a circle of the site's `buffer_km` radius around its coordinate
- **Ignore (0.5)**: the ring from `buffer_km` to 2 x `buffer_km`; loss weight 0
- **Negative (0)**: everything else in the AOI
- Circles are drawn in metres in a local azimuthal equidistant projection, then rasterised on
  the chip's grid, so their size does not depend on latitude

## Splits

By site, defined in `data/manifests/splits.json`:

- **Test**: Satish Dhawan Space Centre, Tanegashima Space Center, Guiana Space Centre
- **Validation**: Xichang Satellite Launch Center, Plesetsk Cosmodrome, Sohae Satellite Launching
  Station
- **Train**: the other 12 sites

A chip within 20 km of a site takes that site's split; each AOI belongs to one split (the split
of the site nearest its centre).

## Known Issues

- Buffer-based labels are coarse; actual footprints may be smaller, larger or offset
- Some sites have limited clear-sky coverage (persistent cloud, high latitudes)
- Negatives may include undocumented sites
- Only six held-out sites, so per-split recall moves in steps of 1/3
