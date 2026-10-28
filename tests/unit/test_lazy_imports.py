import subprocess
import sys


def test_pipeline_module_does_not_import_torch():
    code = "import sys; import services.pipeline.flows; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
