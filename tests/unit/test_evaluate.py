import pytest

from services.training.evaluate import (
    END_MARK,
    START_MARK,
    EvalResult,
    markdown_table,
    scanned_area_km2,
    update_readme,
)


def _result(split):
    return EvalResult(
        split=split,
        n_sites=3,
        scanned_km2=100.0,
        recall_at_5=0.5,
        recall_at_10=0.66,
        recall_at_20=1.0,
        fp_per_1000km2=2.5,
        n_detections=7,
        reliability={},
        checkpoint="c.ckpt",
    )


def test_markdown_table_has_header_and_one_row_per_result():
    lines = markdown_table([_result("val"), _result("test")]).strip().splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("| split")
    assert lines[2].startswith("| val") and lines[3].startswith("| test")


def test_update_readme_replaces_only_the_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"before\n{START_MARK}\nold\n{END_MARK}\nafter\n")
    update_readme(readme, "| t |\n", "c.ckpt", "abc123")
    text = readme.read_text()
    assert text.startswith("before\n") and text.endswith(f"{END_MARK}\nafter\n")
    assert "old" not in text and "| t |" in text and "abc123" in text


def test_update_readme_raises_on_missing_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n")
    with pytest.raises(ValueError):
        update_readme(readme, "t", "c", "x")


def test_scanned_area_of_small_composite(small_composite):
    path, _ = small_composite
    assert scanned_area_km2([path]) == pytest.approx(10.24)
