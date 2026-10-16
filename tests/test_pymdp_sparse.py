"""The sparse-path claim: what pymdp actually does, and how the prose may say it.

The manuscript rested the trie rejection on the sentence "pymdp takes dense
arrays and has no sparse path." That is false as written, and it appeared in
seven places across the source, the decision record and three manuscript
sections. pymdp's JAX backend contains sparse contraction machinery —
``factor_dot`` carries a ``JAXSparse`` overload dispatching to
``spm_dot_sparse`` — and the transition contraction the word level uses is
already sparse-capable. What blocks the path is the ``Agent`` API: it rejects
sparse arrays at construction and does not preserve sparsity across the
filtering loop.

The conclusion survives the correction and only its stated reason changes, which
is exactly why this needed fixing rather than ignoring: an argument that reaches
the right answer through a false premise cannot be checked by anyone, including
its author.

Two things are pinned here. First, the fact — measured, through
``scripts/probe_pymdp_sparse.py``, against the real Stage 3 artifact. Second, the
prose — a general check that no file states the unqualified claim again. The
guard is written against the *class* of statement rather than the exact wording
removed, because a check naming the old sentence would pass on any rephrasing of
it, and would also reprint the very sentence it exists to delete.

Timeouts: the repo default (60 s) cannot even import jax on cold caches, and
pytest-timeout runs per test — so the heavy imports happen at module
collection time, outside any per-test budget, and the module carries the
documented module-level mark for the bounded pymdp walks themselves.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

pytestmark = pytest.mark.timeout(900)

#: Files whose prose a reader takes as a statement about pymdp.
PROSE_ROOTS = ("src", "docs", "manuscript")

#: Where the claim is discussed at length rather than asserted in passing. The
#: research queue records the correction, this test explains it, and the probe
#: demonstrates it; all three necessarily quote the false sentence.
DISCUSSION_SITES = (
    "docs/research_queue.md",
    "scripts/probe_pymdp_sparse.py",
    "tests/test_pymdp_sparse.py",
)


def _prose_files() -> list[Path]:
    files: list[Path] = []
    for root in PROSE_ROOTS:
        for suffix in ("*.py", "*.md"):
            files.extend(
                path
                for path in (REPO_ROOT / root).rglob(suffix)
                if "__pycache__" not in path.parts
                and str(path.relative_to(REPO_ROOT)) not in DISCUSSION_SITES
            )
    return sorted(files)


# ── The fact ─────────────────────────────────────────────────────────────────


def test_a_sparse_transition_reproduces_the_dense_agent_exactly():
    """POSITIVE: after the generated three-site patch, sparse B matches dense.

    This is what makes the restriction an interface boundary rather than a limit
    of the method. If the sparse agent could not reproduce the dense one, the
    correction would overstate what pymdp has, and the original sentence would
    stand.

    NEGATIVE CONTROL: the patches must be *necessary*. An unpatched sparse
    construction has to fail, or the probe would be demonstrating nothing — a
    sparse path that works without patches would mean the public API was never
    the blocker, which is a different (and larger) correction than the one made.
    """
    import probe_pymdp_sparse as probe

    blocked = probe.check_unpatched_failure_modes()
    assert blocked, "no unpatched failure reproduced; the Agent API is not the blocker"
    for site in blocked:
        assert "BCOO" in site["exception"], (
            f"{site['site']} failed for an unrelated reason: {site['exception']}"
        )

    result = probe.run_equivalence_probe()
    assert result["max_diff"] == 0.0, (
        f"sparse and dense posteriors differ by {result['max_diff']}; "
        f"the equivalence the correction rests on does not hold"
    )
    assert result["all_supports_match"], "the induced trie-node supports differ between the two"


def test_the_sparse_contraction_machinery_is_where_the_correction_says_it_is():
    """The claim is about pymdp's source, so it is checked against pymdp's source.

    Line numbers are deliberately not asserted. They drift with every release
    and pin nothing worth pinning; what must hold is that the overload exists
    and that the one-step method the word level runs is the one that never
    touches a densification site.
    """
    import inspect

    from pymdp import inference, maths

    maths_source = inspect.getsource(maths)
    assert "JAXSparse" in maths_source, "the sparse overload's type is gone from maths"
    assert "def spm_dot_sparse" in maths_source, "the sparse contraction routine is gone"
    assert re.search(
        r"@factor_dot\.register|def factor_dot\(\s*\n?\s*M: JAXSparse", maths_source
    ), "factor_dot no longer carries a sparse overload"

    # NEGATIVE CONTROL: `fpi` must be a one-step method. If it were a sequence
    # method, the densification sites in `inference.py` WOULD lie on this
    # project's path, the sparse capability would be nominal, and the original
    # sentence would be closer to true than the correction.
    assert "fpi" in inference.ONE_STEP_METHODS, (
        "fpi is no longer a one-step method; the densification sites may now lie "
        "on the shipped path, which would change what the correction may claim"
    )


# ── The prose ────────────────────────────────────────────────────────────────


def test_no_file_states_the_unqualified_claim():
    """A sparse-path denial must name the API it is true of.

    The general property: any sentence saying pymdp lacks a sparse path has to
    carry the qualifier that makes it true — that the boundary is the public
    ``Agent`` API — within the same sentence. A bare denial is false, and it was
    false in seven places at once because each was a copy of the one before it.

    NEGATIVE CONTROL: the qualified form must PASS, or this would be a check
    that no file may mention sparsity at all, which would be satisfied by
    deleting the discussion rather than correcting it.
    """
    denial = re.compile(
        r"(?:has|have|with|is|are|takes?)\s+(?:dense\s+arrays?\s+(?:and|with)\s+)?no\s+sparse\s+path"
        r"|no\s+sparse\s+path\s+(?:is\s+)?(?:reachable|available|exists)",
        re.I,
    )
    qualifier = re.compile(r"\bAgent\b|\bpublic\s+API\b", re.I)

    offenders: dict[str, list[str]] = {}
    for path in _prose_files():
        text = path.read_text(encoding="utf-8")
        unqualified = []
        for match in denial.finditer(text):
            # The qualifier may sit either side of the denial within the
            # sentence; take a generous window rather than parsing sentences.
            window = text[max(0, match.start() - 160) : match.end() + 160]
            if not qualifier.search(window):
                unqualified.append(" ".join(window.split())[:120])
        if unqualified:
            offenders[str(path.relative_to(REPO_ROOT))] = unqualified
    assert not offenders, (
        f"unqualified sparse-path denials — pymdp has the machinery, the Agent API "
        f"is the boundary: {offenders}"
    )

    # NEGATIVE CONTROL: the qualified sentences the correction installed must
    # exist and must survive this check, or it passes vacuously.
    qualified_sites = [
        path for path in _prose_files() if denial.search(path.read_text(encoding="utf-8"))
    ]
    assert qualified_sites, (
        "no file discusses the sparse path at all; the check has nothing to constrain"
    )


def test_the_correction_is_recorded_where_a_reader_will_look():
    """A corrected claim with no record is a claim that will be re-derived.

    Decision 58 holds the boundary, the two patch sites and the measured
    equivalence. Without it the next reader who sees `B` allocated densely has
    the same conversation from the beginning.
    """
    decisions = (REPO_ROOT / "docs" / "decisions.md").read_text(encoding="utf-8")
    assert "## 58 —" in decisions, "decision 58 is gone"
    entry = decisions[decisions.index("## 58 —") :]
    entry = entry[: entry.index("\n## ") if "\n## " in entry else len(entry)]
    for required in ("spm_dot_sparse", "validate_normalization", "compute_expected_state"):
        assert required in entry, f"decision 58 no longer names {required}"
    assert "0.000e+00" in entry, "decision 58 no longer records the measured equivalence"
    assert "probe_pymdp_sparse" in decisions, "the re-runnable probe is not referenced"
