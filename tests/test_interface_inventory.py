"""Two interfaces exist on purpose. This file is what keeps them from becoming two models.

AILM has exactly two human-facing interfaces, and the difference between them is
capability, not duplication:

``src/ailm/dashboard/``
    The canonical interactive inspector. Levels 1 to 6, bidirectional chat,
    speech in and out, matrix heatmaps. Needs a local Python server
    (``python -m ailm.dashboard``) because the filtering runs in ``engine.py``.

``output/web/ailm.html``
    A generated and ignored local demo. It opens from ``file://`` and is a
    strict subset: the letter filter, the generation policy and the Level 4
    partition. Built by ``scripts/build_web_ailm.py`` from
    ``scripts/export_web_model.py``; it is not a tracked or distributed
    publication artifact.

The risk is not that both exist. It is that the standalone page reimplements the
filter in JavaScript, so it *could* drift into being a second model that agrees
with nothing. These tests pin the two to one source and fail if a third
interface appears without a decision behind it. Both release profiles run the
exporter and page builder before pytest; a missing generated interface is a
failure, never a clean-checkout skip.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_MODEL = PROJECT_ROOT / "output" / "web" / "ailm_model.json"
WEB_PAGE = PROJECT_ROOT / "output" / "web" / "ailm.html"

#: Every interface the project ships, as ``(path, what it is)``. A new entry here
#: is a deliberate act; a new interface without one fails
#: :func:`test_no_undeclared_interface_has_appeared`.
DECLARED_INTERFACES: tuple[tuple[str, str], ...] = (
    (
        "src/ailm/dashboard/static/index.html",
        "canonical interactive inspector, Levels 1-6, served by ailm.dashboard",
    ),
    (
        "output/web/ailm.html",
        "generated ignored local demo, zero-dependency subset",
    ),
)


def _web_model() -> dict:
    assert WEB_MODEL.is_file(), (
        "generated web model is missing; run scripts/export_web_model.py. "
        "Release profiles must build it before pytest rather than accepting a skip."
    )
    return json.loads(WEB_MODEL.read_text(encoding="utf-8"))


def test_the_dashboard_and_the_page_share_one_model_builder():
    """Not two implementations of the transition, one function used twice.

    This is the check that makes the second interface safe. If the dashboard ever
    grows its own builder, the two would drift while both looked correct, and the
    standalone page would be reporting a model the paper does not describe.
    """
    from ailm.dashboard import engine
    from ailm.levels.letter import build_letter_transition

    assert engine.build_letter_transition is build_letter_transition, (
        "the dashboard no longer uses the shipped letter builder; the standalone "
        "page and the dashboard would now be two models rather than two views"
    )
    assert engine.build_unigram_a is build_letter_transition


def test_every_interface_reads_the_same_alphabet():
    """A shared alphabet is the minimum condition for the views to be comparable."""
    from ailm.dashboard import engine
    from ailm.levels.letter import IDX_TO_ALPHA, N_SYMBOLS

    canonical = [IDX_TO_ALPHA[i] for i in range(N_SYMBOLS)]
    assert engine.ALPHA_LIST == canonical
    assert _web_model()["alphabet"] == canonical


def test_the_exported_transition_matches_the_shipped_one_to_its_declared_precision():
    """The export rounds; it must not alter.

    ``PRECISION`` in the exporter is six decimals, so the largest permitted
    disagreement is half a unit in the last place. Anything larger means the page
    is filtering through a matrix the project did not build.
    """
    from ailm.levels.letter import build_letter_transition, load_brown_counts

    shipped = np.asarray(build_letter_transition(load_brown_counts()))[0]
    exported = np.asarray(_web_model()["letter_transition"])
    assert exported.shape == shipped.shape
    assert float(np.abs(shipped - exported).max()) <= 5e-7


def test_the_page_carries_only_claims_the_ledger_holds():
    """Every number the page displays is recorded, not typed into the markup."""
    from ailm.claims import load_ledger

    ledger = load_ledger(PROJECT_ROOT)
    missing = [label for label in _web_model()["claims"] if ledger.get(label) is None]
    assert missing == [], f"the web model carries claims the ledger does not hold: {missing}"


def test_the_standalone_page_inlines_the_exact_current_model():
    assert WEB_PAGE.is_file(), "run scripts/build_web_ailm.py after exporting the web model"
    model = WEB_MODEL.read_text(encoding="utf-8")
    page = WEB_PAGE.read_text(encoding="utf-8")
    assert "__MODEL_JSON__" not in page
    assert page.count(model) == 1, "standalone page does not inline the exact exported model once"


def _standalone_accessibility_defects(page: str) -> list[str]:
    """Independent source audit for the generated page's accessible contract."""
    defects: list[str] = []
    if not re.search(r'<label\b[^>]*\bfor="txt"[^>]*>', page):
        defects.append("filter input has no label")

    for element_id in ("selfcheck", "filter-results", "out"):
        tag = re.search(rf'<[^>]+\bid="{re.escape(element_id)}"[^>]*>', page)
        if tag is None or 'aria-live="polite"' not in tag.group(0):
            defects.append(f"{element_id} is not a polite live region")

    bars = re.search(r'<[^>]+\bid="bars"[^>]*>', page)
    if bars is None or 'role="list"' not in bars.group(0):
        defects.append("probability bars are not a named list")
    if 'role="listitem"' not in page or 'aria-label="Next symbol ${symbol}' not in page:
        defects.append("probability bars have no text equivalent")
    if '"outside lexicon"' not in page or "[${status}]" not in page:
        defects.append("word membership is conveyed only by color")
    return defects


def test_the_standalone_demo_has_accessible_results_with_a_planted_negative_control():
    page = WEB_PAGE.read_text(encoding="utf-8")
    assert _standalone_accessibility_defects(page) == []

    planted = (
        page.replace('for="txt"', 'for="wrong-input"', 1)
        .replace('role="listitem"', 'role="presentation"', 1)
        .replace('"outside lexicon"', '"nonlex"', 1)
    )
    defects = _standalone_accessibility_defects(planted)
    assert "filter input has no label" in defects
    assert "probability bars have no text equivalent" in defects
    assert "word membership is conveyed only by color" in defects


def test_the_standalone_demo_is_generated_and_ignored_not_a_distributed_surface():
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "output/web/ailm.html"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    tracked_dashboard = subprocess.run(
        ["git", "check-ignore", "--quiet", "src/ailm/dashboard/static/index.html"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    assert ignored.returncode == 0, "the generated local demo is no longer ignored"
    assert tracked_dashboard.returncode == 1, (
        "the canonical dashboard source is unexpectedly ignored"
    )


def test_no_undeclared_interface_has_appeared():
    """A third interface is a decision, not an accident.

    The project has been through one round of accidental duplication already. The
    cost of a third is not the file, it is that a reader cannot tell which one the
    paper is about.
    """
    _assert_declared_interfaces(PROJECT_ROOT)


def _assert_declared_interfaces(root: Path) -> None:
    declared = {path for path, _why in DECLARED_INTERFACES}
    skip = ("/output/", "/refs/", "/.venv/", "/.git/", "/htmlcov/", "/node_modules/")
    found = {
        str(candidate.relative_to(root))
        for candidate in root.rglob("*.html")
        if candidate.relative_to(root).parts[0] != ".omo"
        and not any(part in f"/{candidate.relative_to(root)}" for part in skip)
    }
    assert found <= declared, (
        f"undeclared HTML interface(s): {sorted(found - declared)}. If this is "
        f"intentional, add it to DECLARED_INTERFACES with what distinguishes it "
        f"from the two that already exist."
    )


@pytest.mark.parametrize("relative", ["src/new_ui/index.html", "src/.omo/product.html"])
def test_operator_evidence_is_ignored_but_undeclared_product_html_fails(tmp_path, relative):
    for path, _why in DECLARED_INTERFACES:
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("<html>declared interface</html>", encoding="utf-8")
    proof = tmp_path / ".omo/evidence/attempt/bounded.html"
    proof.parent.mkdir(parents=True)
    proof.write_text("<html>operator proof</html>", encoding="utf-8")
    _assert_declared_interfaces(tmp_path)
    product = tmp_path / relative
    product.parent.mkdir(parents=True, exist_ok=True)
    product.write_text("<html>undeclared product interface</html>", encoding="utf-8")
    with pytest.raises(AssertionError, match="undeclared HTML interface") as caught:
        _assert_declared_interfaces(tmp_path)
    assert relative in str(caught.value)
    assert ".omo/evidence" not in str(caught.value)


def test_the_declared_interfaces_still_exist():
    """The inventory is only worth having while it describes real files."""
    for path, _why in DECLARED_INTERFACES:
        target = PROJECT_ROOT / path
        assert target.exists(), f"declared interface {path} is missing"
