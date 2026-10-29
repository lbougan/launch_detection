"""Export a trained Lightning checkpoint to ONNX for Triton Inference Server."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_IN_CHANNELS = 10
DEFAULT_TILE_SIZE = 256


def export_to_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    stats_path: str | Path | None = None,
    in_channels: int = DEFAULT_IN_CHANNELS,
    tile_size: int = DEFAULT_TILE_SIZE,
    opset_version: int = 17,
    validate: bool = True,
) -> Path:
    """Export a LaunchSiteSegModule checkpoint to ONNX with dynamic batch axis.

    Args:
        checkpoint_path: Path to the Lightning ``.ckpt`` file.
        output_path: Where to write ``model.onnx``.
        stats_path: Normalisation stats file. The checkpoint's own stats are what gets embedded
            as metadata and written next to the model; a file given here must match them.
            Checkpoints from before stats were embedded need the file (default: the
            project's ``channel_stats.json``).
        in_channels: Number of input channels the model expects.
        tile_size: Spatial dimension (H=W) of input tiles.
        opset_version: ONNX opset version.
        validate: If True, run an ORT session and compare outputs to PyTorch.

    Returns:
        Resolved path to the written ONNX file.
    """
    from libs.config import default_stats_path
    from libs.features.stats import resolve_stats, save_stats
    from services.training.lightning_module import LaunchSiteSegModule

    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Rebuild from hparams + state_dict rather than load_from_checkpoint, which requires
    # Lightning's version bookkeeping keys that plain ``torch.save`` checkpoints lack.
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    embedded = ckpt["hyper_parameters"].get("stats")
    if embedded is None and stats_path is None:
        stats_path = default_stats_path()  # checkpoints from before stats were embedded
    stats = resolve_stats(embedded, stats_path)
    module = LaunchSiteSegModule(**ckpt["hyper_parameters"])
    module.load_state_dict(ckpt["state_dict"])
    module.eval()
    model = module.model

    dummy_input = torch.randn(1, in_channels, tile_size, tile_size)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )
    logger.info("Exported ONNX model → %s", output_path)

    _embed_metadata(output_path, stats, in_channels, tile_size)
    save_stats(stats, output_path.parent / "stats.json")

    if validate:
        _validate_onnx(model, output_path, dummy_input)

    return output_path.resolve()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _embed_metadata(onnx_path: Path, stats: dict, in_channels: int, tile_size: int) -> None:
    """Append stats, band spec and provenance to the ONNX ``metadata_props``."""
    import onnx

    from libs.features.indices import INDEX_NAMES
    from libs.geo.bandspec import S2_BAND_NAMES, BandSpec

    entries = {
        "stats_json": json.dumps(stats),
        "bandspec_json": BandSpec(names=S2_BAND_NAMES, indices=INDEX_NAMES).to_json(),
        "in_channels": str(in_channels),
        "tile_size": str(tile_size),
        "git_commit": _git_commit(),
    }
    model = onnx.load(str(onnx_path))
    keep = [p for p in model.metadata_props if p.key not in entries]
    del model.metadata_props[:]
    model.metadata_props.extend(keep)
    for key, value in entries.items():
        model.metadata_props.add(key=key, value=value)
    onnx.save(model, str(onnx_path))


def _validate_onnx(
    pytorch_model: torch.nn.Module,
    onnx_path: Path,
    dummy_input: torch.Tensor,
    atol: float = 1e-5,
) -> None:
    """Compare ONNX Runtime output against PyTorch to ensure export fidelity."""
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model passed checker validation")

    with torch.no_grad():
        pt_out = pytorch_model(dummy_input).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": dummy_input.numpy()})[0]

    if not np.allclose(pt_out, ort_out, atol=atol):
        max_diff = float(np.max(np.abs(pt_out - ort_out)))
        raise ValueError(f"ONNX validation failed: max abs diff {max_diff:.6e} exceeds atol {atol}")
    logger.info("ONNX validation passed (atol=%s)", atol)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export LaunchSiteSegModule checkpoint to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt Lightning checkpoint")
    parser.add_argument(
        "--output",
        default="model_repository/unet_seg/1/model.onnx",
        help="Output ONNX path",
    )
    parser.add_argument(
        "--stats",
        default=None,
        help="stats JSON; default: the checkpoint's own (checked against this file if given)",
    )
    parser.add_argument("--in-channels", type=int, default=DEFAULT_IN_CHANNELS)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-validation", action="store_true", help="Skip ORT validation")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        stats_path=args.stats,
        in_channels=args.in_channels,
        tile_size=args.tile_size,
        opset_version=args.opset,
        validate=not args.skip_validation,
    )


if __name__ == "__main__":
    main()
