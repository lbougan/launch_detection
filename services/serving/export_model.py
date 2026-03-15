"""Export a trained Lightning checkpoint to ONNX for Triton Inference Server."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_IN_CHANNELS = 10
DEFAULT_TILE_SIZE = 256


def export_to_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    in_channels: int = DEFAULT_IN_CHANNELS,
    tile_size: int = DEFAULT_TILE_SIZE,
    opset_version: int = 17,
    validate: bool = True,
) -> Path:
    """Export a LaunchSiteSegModule checkpoint to ONNX with dynamic batch axis.

    Args:
        checkpoint_path: Path to the Lightning ``.ckpt`` file.
        output_path: Where to write ``model.onnx``.
        in_channels: Number of input channels the model expects.
        tile_size: Spatial dimension (H=W) of input tiles.
        opset_version: ONNX opset version.
        validate: If True, run an ORT session and compare outputs to PyTorch.

    Returns:
        Resolved path to the written ONNX file.
    """
    from services.training.lightning_module import LaunchSiteSegModule

    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    module = LaunchSiteSegModule.load_from_checkpoint(
        str(checkpoint_path), map_location="cpu"
    )
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
    )
    logger.info("Exported ONNX model → %s", output_path)

    if validate:
        _validate_onnx(model, output_path, dummy_input)

    return output_path.resolve()


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
        raise ValueError(
            f"ONNX validation failed: max abs diff {max_diff:.6e} exceeds atol {atol}"
        )
    logger.info("ONNX validation passed (atol=%s)", atol)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export LaunchSiteSegModule checkpoint to ONNX"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to .ckpt Lightning checkpoint"
    )
    parser.add_argument(
        "--output",
        default="model_repository/unet_seg/1/model.onnx",
        help="Output ONNX path",
    )
    parser.add_argument("--in-channels", type=int, default=DEFAULT_IN_CHANNELS)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip ORT validation"
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        in_channels=args.in_channels,
        tile_size=args.tile_size,
        opset_version=args.opset,
        validate=not args.skip_validation,
    )


if __name__ == "__main__":
    main()
