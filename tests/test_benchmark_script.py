import subprocess
import sys

def test_benchmark_script_execution():
    """Verify that scripts/run_full_benchmark.py runs smoothly on synthetic data."""
    cmd = [sys.executable, "scripts/run_full_benchmark.py", "--size", "1000", "--min_acc", "0.60"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Benchmark failed with output:\n{result.stderr}\n{result.stdout}"
    assert "SUCCESS: Pipeline passed quality criteria" in result.stdout
