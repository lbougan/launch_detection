import subprocess
import sys

HEAVY = ("torch", "pytorch_lightning", "sqlalchemy")


def test_pipeline_module_does_not_import_heavy_dependencies():
    code = (
        "import sys; import services.pipeline.flows; "
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"imported at flow import time: {out.stdout.strip()}"
