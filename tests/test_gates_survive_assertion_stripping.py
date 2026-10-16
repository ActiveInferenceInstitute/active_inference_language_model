"""Load-bearing invariants must not be `assert` statements.

`python -O` deletes every `assert` from the bytecode. Any check written as a
bare `assert` therefore stops running under that flag, silently and with no
error — the module imports fine, the function returns, and whatever the check
existed to prevent proceeds.

That distinction is not mechanically reliable: a function-scope ``assert`` may
look like type narrowing while also being the only thing preventing malformed
runtime state. Production and orchestration code therefore use explicit raises
for every invariant, including narrowing checks.

It is especially intolerable for a *gate*: an invariant whose whole job is to
stop a wrong artifact from shipping.
This project had six of those, including — worst of all —
``GateController.assert_metric_sensitivity``, the check that detects a
MISCALIBRATED GATE. A gate-vacuity mechanism that evaporates under an
optimization flag is the exact failure it exists to prevent.

The test is written against the general property, not against a list of sites:
any ``assert`` anywhere in production or orchestration code is a finding.
Naming the originals would re-add the very pattern this file exists to remove,
and would not catch the next one.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "ailm"
SCRIPTS = REPO_ROOT / "scripts"


def _assert_statements(path: Path) -> list[int]:
    """Line numbers of every executable ``assert`` in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return sorted(node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert))


def _production_paths(scripts_root: Path | None = None) -> list[Path]:
    """Every production or orchestration Python file, nested scripts included.

    ``src/`` was scanned recursively while ``scripts/`` used a flat glob, so a
    future ``scripts/<subdir>/x.py`` would silently escape the no-assert
    property this file exists to enforce. Both trees are the project's own
    code — never caches, vendor, or output — so the recursive walk stays
    bounded and no exclusion list is needed. A caller may point ``scripts_root``
    at a temporary tree; the production scan leaves it at the default.
    """
    root = SCRIPTS if scripts_root is None else scripts_root
    return sorted(SRC.rglob("*.py")) + sorted(root.rglob("*.py"))


@pytest.mark.parametrize(
    "path",
    _production_paths(),
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_no_assert_statements_in_production_or_orchestration(path: Path):
    """Every runtime invariant must be an explicit raise.

    Rejecting the whole class avoids a reviewer having to decide whether a
    function-scope assertion is "only" for typing. The optimized interpreter
    cannot preserve either kind.
    """
    offending = _assert_statements(path)
    assert not offending, (
        f"{path.relative_to(REPO_ROOT)} has assert statement(s) at line(s) "
        f"{offending}; `python -O` strips them. Use `if not cond: raise ...`."
    )


def test_the_detector_fires_on_planted_module_and_function_violations(tmp_path: Path):
    """NEGATIVE CONTROL: both scopes must be detected.

    Without this the test above passes trivially if `_module_level_asserts`
    silently returns nothing — which is precisely the class of failure the whole
    file is about.
    """
    planted = tmp_path / "planted.py"
    planted.write_text("X = 1\nassert X == 1, 'gate'\n", encoding="utf-8")
    assert _assert_statements(planted) == [2]

    narrowing = tmp_path / "narrowing.py"
    narrowing.write_text("def f(x):\n    assert x is not None\n    return x\n", encoding="utf-8")
    assert _assert_statements(narrowing) == [2]


def test_nested_scripts_are_collected_and_flagged(tmp_path: Path):
    """S15 falsifier: a planted assert under scripts/<subdir>/ is caught.

    The flat glob that preceded the recursive walk would have collected
    neither file below. With the scanner pointed at a temporary tree, the
    nested planted module must be collected AND flagged, and the nested clean
    module must be collected with no finding — proving the recursion reaches
    subdirectories rather than merely scanning more of the same flat list.
    The scripts root arrives as a parameter: mutating this module's own
    constant through the sys.modules registry is a dependency-replacement
    shape the guard exists to reject.
    """
    fake_scripts = tmp_path / "scripts"
    (fake_scripts / "subdir").mkdir(parents=True)
    planted = fake_scripts / "subdir" / "planted.py"
    planted.write_text("X = 1\nassert X == 1, 'gate'\n", encoding="utf-8")
    clean = fake_scripts / "subdir" / "clean.py"
    clean.write_text("X = 1\nif X != 1:\n    raise ValueError('gate')\n", encoding="utf-8")

    collected = _production_paths(scripts_root=fake_scripts)
    assert planted in collected and clean in collected, collected
    assert _assert_statements(planted) == [2]
    assert _assert_statements(clean) == []


@pytest.mark.timeout(1920)
def test_gate_calibration_check_survives_optimized_bytecode():
    """The gate-vacuity check must still raise under `python -O`.

    This is the behavioral half: the scan above is structural, and a structural
    scan cannot prove the converted check actually still runs. Executed in a
    subprocess because -O is a startup flag.
    """
    program = (
        "import sys; sys.path.insert(0, 'src')\n"
        "from ailm.control import ControlHarness\n"
        "harness = ControlHarness('probe')\n"
        "try:\n"
        "    harness.assert_gate_is_live(1.0, 2.0, 'higher_is_better')\n"
        "except AssertionError:\n"
        "    print('RAISED')\n"
        "else:\n"
        "    print('SILENT')\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", program],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=960,
    )
    assert result.returncode == 0, result.stderr
    assert "RAISED" in result.stdout, (
        "the gate-calibration check did not fire under -O; it is an `assert` again"
    )


@pytest.mark.timeout(1920)
def test_corrupt_count_guard_survives_optimized_bytecode():
    """A malformed corpus count must not become a negative probability under ``-O``.

    This is the production-side negative control. Before the repository-wide
    scan covered function bodies, ``build_phone_transition`` used a function
    assert and optimized Python returned a column-stochastic matrix containing
    negative probabilities.
    """
    program = (
        "import sys; sys.path.insert(0, 'src')\n"
        "from ailm.levels.phoneme import build_phone_transition\n"
        "try:\n"
        "    build_phone_transition({'AA,AA': -100})\n"
        "except ValueError:\n"
        "    print('RAISED')\n"
        "else:\n"
        "    print('SILENT')\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", program],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=960,
    )
    assert result.returncode == 0, result.stderr
    assert "RAISED" in result.stdout, "negative phone counts reached matrix normalization under -O"
