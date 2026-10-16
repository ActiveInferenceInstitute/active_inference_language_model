"""Fresh-interpreter enforcement of the oracle's dependency boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"


def _probe(*, plant_leak: bool = False) -> subprocess.CompletedProcess[str]:
    setup = "import types; sys.modules['jax'] = types.ModuleType('jax');" if plant_leak else ""
    code = (
        "import sys;"
        f"sys.path.insert(0, {str(SRC)!r});"
        f"{setup}"
        "import ailm.verification.oracle;"
        "leaked=sorted({m.split('.')[0] for m in sys.modules}"
        " & {'pymdp','jax','jaxlib'});"
        "print(','.join(leaked));"
        "raise SystemExit(3 if leaked else 0)"
    )
    return subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_oracle_imports_no_pymdp_or_jax_in_a_fresh_interpreter():
    result = _probe()
    assert result.returncode == 0, (
        f"oracle dependency boundary leaked: stdout={result.stdout!r}, stderr={result.stderr!r}"
    )
    assert result.stdout.strip() == ""


def test_independence_probe_detects_a_planted_forbidden_module():
    result = _probe(plant_leak=True)
    assert result.returncode == 3
    assert result.stdout.strip() == "jax"
