"""Training CLI: a thin click wrapper around `services.training.fit.fit`."""

from __future__ import annotations

import logging

import click

from libs.config import default_stats_path, settings
from services.training.fit import FitConfig, fit


def _resolve_precision(precision: str | None, gpus: int) -> int | str:
    """Explicit ``--precision`` wins; otherwise mixed precision on GPU, full precision on CPU."""
    if precision is None:
        return "16-mixed" if gpus > 0 else 32
    return int(precision) if precision.isdigit() else precision


@click.command()
@click.option("--manifest", type=click.Path(exists=True), required=True, help="Chip manifest JSON.")
@click.option(
    "--stats",
    type=click.Path(exists=True),
    default=str(default_stats_path()),
    help="Channel statistics JSON used to standardise inputs.",
)
@click.option("--sites", "known_sites_path", default="data/manifests/known_sites.json")
@click.option("--splits", "splits_path", default="data/manifests/splits.json")
@click.option("--model-type", type=click.Choice(["unet", "deeplabv3+"]), default="unet")
@click.option("--encoder", default="resnet34")
@click.option("--encoder-weights", default="imagenet", help="Use 'none' for random init.")
@click.option("--lr", default=1e-3, type=float)
@click.option("--epochs", default=50, type=int)
@click.option("--batch-size", default=16, type=int)
@click.option("--gpus", default=0, type=int)
@click.option(
    "--precision",
    default=None,
    help="Lightning precision, e.g. 32, 16-mixed, bf16-mixed. Default: 16-mixed on GPU, else 32.",
)
@click.option("--num-workers", default=2, type=int)
@click.option("--monitor", default="val/site_recall@10")
@click.option("--experiment-name", default="launchsite-seg")
@click.option("--checkpoint-dir", default="checkpoints")
@click.option("--val-composite", "val_composites", multiple=True, help="Composites of val AOIs.")
def train(encoder_weights: str, stats: str, manifest: str, precision: str | None, **kwargs) -> None:
    """Train the launch-site segmentation model."""
    cfg = FitConfig(
        manifest_path=manifest,
        stats_path=stats,
        encoder_weights=None if encoder_weights.lower() == "none" else encoder_weights,
        precision=_resolve_precision(precision, kwargs["gpus"]),
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
        **kwargs,
    )
    best = fit(cfg)
    click.echo(f"Best checkpoint: {best}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
