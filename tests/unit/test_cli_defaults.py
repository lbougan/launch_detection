"""CLI defaults: one stats path everywhere, ingest scene limit and precision pass-through."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from libs.config import default_stats_path
from services.pipeline import flows
from services.serving.export_model import build_parser as export_parser
from services.training.evaluate import build_parser as evaluate_parser
from services.training.train_cli import _resolve_precision, train


def test_all_stats_defaults_equal_default_stats_path():
    expected = str(default_stats_path())
    stats_opt = next(p for p in train.params if p.name == "stats")
    assert stats_opt.default == expected
    assert evaluate_parser().parse_args(["--checkpoint", "c"]).stats == expected
    assert export_parser().parse_args(["--checkpoint", "c"]).stats == expected
    assert flows.build_parser().parse_args(["train"]).stats == expected
    assert flows.build_parser().parse_args(["detect"]).stats == expected


def test_ingest_max_scenes_is_passed_through_to_the_flow():
    with patch.object(flows, "ingest_and_preprocess_flow") as flow:
        flows.main(["ingest", "--max-scenes", "20"])
    assert flow.call_args.kwargs["max_scenes"] == 20
    assert flows.build_parser().parse_args(["ingest"]).max_scenes == 50


def test_ingest_task_forwards_limit_to_stac_client():
    with patch("libs.stac.client.ingest_aoi", return_value=[]) as ingest:
        flows.ingest_aoi_task.fn("a", [0, 0, 1, 1], "2024-01-01/2024-01-31", "out", limit=7)
    assert ingest.call_args.kwargs["limit"] == 7


@pytest.mark.parametrize(
    ("precision", "gpus", "expected"),
    [(None, 0, 32), (None, 1, "16-mixed"), ("16-mixed", 1, "16-mixed"), ("32", 1, 32)],
)
def test_precision_option(precision, gpus, expected):
    assert _resolve_precision(precision, gpus) == expected
    assert any(p.name == "precision" for p in train.params)
