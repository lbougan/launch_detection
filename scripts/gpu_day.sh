#!/usr/bin/env bash
# Remote GPU day: ingest -> train -> evaluate -> export. Run from the repo root on a machine
# with an NVIDIA GPU, ~200 GB disk and the compose stack (postgres, minio, mlflow) up.
set -euo pipefail
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://localhost:5000}
STATS=data/manifests/channel_stats.json

# 0. Environment. pyproject pins torch to the CPU wheel index (CI and laptops), so swap in the
#    CUDA build of the same versions, then use `uv run --no-sync` so uv does not swap it back.
#    Set TORCH_INDEX to the wheel index that matches the machine's CUDA driver.
uv sync --all-extras
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
RUN=(uv run --no-sync)
TORCH_V=$("${RUN[@]}" python -c "import torch; print(torch.__version__.split('+')[0])")
TV_V=$("${RUN[@]}" python -c "import torchvision; print(torchvision.__version__.split('+')[0])")
uv pip install --reinstall "torch==$TORCH_V" "torchvision==$TV_V" --index-url "$TORCH_INDEX"
"${RUN[@]}" python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device visible'"

# 1. Ingest all 17 AOIs: STAC search, SCL cloud mask, reflectance crops, NaN-median composites,
#    chips, metric labels, site-held-out split, and channel stats from training composites only.
#    --max-scenes 20 keeps the compact AOIs (and the disk footprint) small.
"${RUN[@]}" python -m services.pipeline.flows ingest --date-range 2024-01-01/2024-06-30 --max-scenes 20

# 2. Train. The checkpoint monitor val/site_recall@10 needs the validation AOIs' composites;
#    without them fit() falls back to val/loss.
VAL_ARGS=()
while IFS= read -r comp; do
  VAL_ARGS+=(--val-composite "$comp")
done < <("${RUN[@]}" python - <<'PY'
import json
from pathlib import Path

from libs.config import settings
from libs.geo.chips import load_known_sites
from libs.geo.splits import aoi_split

aois = json.loads(Path("data/manifests/aoi_config.json").read_text())["aois"]
sites = load_known_sites(Path("data/manifests/known_sites.json"))
splits = json.loads(Path("data/manifests/splits.json").read_text())
for aoi in aois:
    comp = Path(settings.data_dir) / "composites" / f"{aoi['name']}_composite.tif"
    if aoi_split(aoi, sites, splits) == "val" and comp.exists():
        print(comp)
PY
)
TRAIN_LOG=$(mktemp)
"${RUN[@]}" python -m services.training.train_cli --manifest data/manifests/chip_manifest.json \
  --stats "$STATS" --epochs 40 --gpus 1 --precision 16-mixed "${VAL_ARGS[@]}" | tee "$TRAIN_LOG"

# The best checkpoint by the monitor (not simply the newest file in checkpoints/).
BEST=$(sed -n 's/^Best checkpoint: //p' "$TRAIN_LOG" | tail -1)
if [[ -z "$BEST" ]]; then
  BEST=$(ls -t checkpoints/*.ckpt | head -1)
fi
echo "Using checkpoint $BEST"

# 3. Evaluate on the held-out validation and test sites; writes out/eval/{val,test}.json and
#    fills the <!-- EVAL:START --> ... <!-- EVAL:END --> block in README.md.
"${RUN[@]}" python -m services.training.evaluate --checkpoint "$BEST" --stats "$STATS" \
  --split val --split test --update-readme README.md

# 4. Export to ONNX for Triton; the normalisation stats and band spec are embedded as metadata.
"${RUN[@]}" python -m services.serving.export_model --checkpoint "$BEST" --stats "$STATS"

echo "Now commit README.md and $STATS; copy model_repository/unet_seg/1/ to the Triton host."
