"""Enforce the claims discipline against the REAL repository prose.

Why this file exists
--------------------
``ClaimsLedger.scan_prose`` existed from Stage 0 and was exercised only against
``tmp_path`` fixtures. Every test passed while ``claim:phoneme_count_39`` sat
unbacked in README.md, AGENTS.md, and docs/provenance/sources.md — the ledger
protocol was documented, tested in miniature, and never actually applied to the
files a reader would take as factual.

A gate that only runs on synthetic inputs constrains nothing about the shipped
artifact. This file points the scanner at the real tree.

Scope
-----
Prose a reader treats as factual: root Markdown, docs/, GNN/formal
documentation/specifications, and src/ docstrings. ``tests/`` is deliberately
excluded — test files carry illustrative tags (``claim:alpha_claim``) that are
fixtures, not assertions about the project.

Non-vacuity
-----------
``test_scanner_detects_an_injected_unbacked_claim`` is the positive control: it
plants a fake tag in a file inside the scanned set and requires the scanner to
flag it. Without that control, ``assert unbacked == []`` would also pass if
``_prose_files()`` silently returned nothing or the regex stopped matching.
"""

from __future__ import annotations

import ast
import collections
import functools
import importlib.util
import json
import re
from pathlib import Path
from typing import NamedTuple

import pytest

from ailm.claims import ClaimsLedger, load_ledger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEDGER_FILE = PROJECT_ROOT / "data" / "claims.json"
CMUDICT_COUNTS_FILE = PROJECT_ROOT / "vendor" / "cmudict_counts.json"
STAGE0_RECORDER = PROJECT_ROOT / "scripts" / "record_stage0_claims.py"

# The bare-literal pass is intentionally narrower than the tag-validity pass.
# Every exclusion is named here so expanding or shrinking the surface is a
# reviewable policy change, not an incidental glob result.  These files are NOT
# certified as number-free: their existing ``claim:`` tags are still checked by
# ``test_no_unbacked_prose_claims`` above.  They are deferred because line-oriented
# scanning cannot certify their historical/example literals without either
# migrating old evidence or turning the gate into a digit ban.
_BARE_QUANTITATIVE_EXCLUSIONS: dict[str, str] = {
    "CHANGELOG.md": (
        "append-only historical release record with substantial pre-ledger numeric debt"
    ),
    "docs/development_history.md": (
        "verbatim pre-1.0.0 changelog history moved out of CHANGELOG.md; the same "
        "append-only record with the same pre-ledger numeric debt"
    ),
    "TODO.md": ("planning and exploratory work queue; proposals are not recorded project results"),
    "docs/decisions.md": (
        "dated historical decision log; bare literals are existing migration debt, "
        "not silently classified as verified"
    ),
    "docs/research_queue.md": (
        "explicit exploratory scratch and falsifier queue; proposals are not recorded results"
    ),
    "docs/method/style_guide.md": (
        "contains deliberate BAD bare-number examples used to teach the claim protocol"
    ),
}


def _prose_files() -> list[Path]:
    """Every file whose claim tags are factual assertions about this project."""
    # Every root Markdown file is reader-visible, including dated handoffs and
    # reviews. Historical files may be explicit bare-number migration debt, but
    # any claim tag they do carry must still resolve to real evidence.
    files: list[Path] = sorted(PROJECT_ROOT.glob("*.md"))
    files.extend(sorted((PROJECT_ROOT / "docs").rglob("*.md")))
    # GNN model annotations report measured outcomes beside the structural
    # declaration. They previously carried claim tags outside this scan, so a
    # typo there could ship despite the repository-wide discipline being green.
    files.extend(sorted((PROJECT_ROOT / "gnn").rglob("*.md")))
    files.extend(sorted((PROJECT_ROOT / "src").rglob("*.py")))
    # formal/README.md documents the Lean layer and cites its recorded counts.
    # It was outside this set, so those counts were bare digits: correct on the
    # day they were written, with no mechanism to stay correct. docs/formal.md
    # already used claim: tags for the same numbers; the in-tree companion did
    # not, which is precisely the asymmetry a scan set decides.
    files.extend(sorted((PROJECT_ROOT / "formal").rglob("*.md")))
    return files


def _bare_quantitative_prose_files() -> list[Path]:
    """Reader-facing prose covered by the conservative bare-number scan.

    Root Markdown, documentation, GNN/formal Markdown and Python docstrings are
    covered.  The small exclusion inventory is explicit historical or example
    debt, not a claim that those files contain no unsupported numbers.
    """

    candidates = _bare_quantitative_candidates()
    return [
        path
        for path in candidates
        if path.relative_to(PROJECT_ROOT).as_posix() not in _BARE_QUANTITATIVE_EXCLUSIONS
    ]


def _bare_quantitative_candidates() -> list[Path]:
    """All public prose surfaces eligible for bare-quantity enforcement."""

    files = sorted(PROJECT_ROOT.glob("*.md"))
    files.extend(sorted((PROJECT_ROOT / "docs").rglob("*.md")))
    files.extend(sorted((PROJECT_ROOT / "gnn").rglob("*.md")))
    files.extend(sorted((PROJECT_ROOT / "formal").rglob("*.md")))
    files.extend(sorted((PROJECT_ROOT / "src").rglob("*.py")))
    return files


# ── The scanned set must be non-empty ─────────────────────────────────────────


def test_prose_file_set_is_non_empty():
    """Guard against a vacuous scan: an empty file list would pass every check."""
    files = _prose_files()
    assert len(files) >= 5, (
        f"Only {len(files)} prose files discovered — the scan set is too small "
        "to be meaningful. Check _prose_files() globbing."
    )


def test_prose_set_includes_the_top_level_docs():
    """The reader-facing files specifically must be in scope."""
    names = {p.name for p in _prose_files()}
    for required in ("README.md", "AGENTS.md", "HANDOFF.md"):
        assert required in names, f"{required} missing from the scanned prose set"


def test_prose_set_includes_every_root_markdown_file():
    """A new root report cannot acquire claim tags outside the validity scan."""

    root_markdown = {path.resolve() for path in PROJECT_ROOT.glob("*.md")}
    scanned = {path.resolve() for path in _prose_files()}
    assert root_markdown <= scanned


def test_prose_set_includes_the_formal_layer_docs():
    """`formal/README.md` cites recorded counts and must be scanned for them.

    Pinned by path rather than by name: the top-level README.md would otherwise
    satisfy a name-only check and the in-tree companion could drop back out of
    scope unnoticed.
    """
    relative = {p.relative_to(PROJECT_ROOT).as_posix() for p in _prose_files()}
    assert "formal/README.md" in relative, (
        "formal/README.md is outside the scanned prose set, so the Lean layer's "
        "recorded counts could be hand-typed there without the scanner objecting."
    )


def test_prose_set_includes_gnn_specs_and_guidance():
    """Measured claims inside a declarative spec are still reader-facing claims."""
    relative = {p.relative_to(PROJECT_ROOT).as_posix() for p in _prose_files()}
    required = {
        "gnn/README.md",
        "gnn/AGENTS.md",
        "gnn/models/L5_clause.gnn.md",
        "gnn/models/L6_narrative.gnn.md",
    }
    missing = required - relative
    assert missing == set(), f"GNN factual prose outside the claims scan: {sorted(missing)}"


# ── POSITIVE CONTROL: the scanner must be able to fail ───────────────────────


def test_scanner_detects_an_injected_unbacked_claim(tmp_path):
    """POSITIVE CONTROL — an unbacked tag in a scanned file must be reported.

    Proves the assertion in test_no_unbacked_prose_claims is load-bearing rather
    than green because nothing is being examined.
    """
    ledger = ClaimsLedger(tmp_path / "claims.json")
    planted = tmp_path / "fake_readme.md"
    planted.write_text(
        "The model reaches claim:this_label_is_deliberately_unbacked nats.",
        encoding="utf-8",
    )
    unbacked = ledger.scan_prose(planted)
    assert "this_label_is_deliberately_unbacked" in unbacked, (
        "scan_prose failed to flag a planted unbacked claim — the scanner is "
        "broken, so the real-prose assertion below would be vacuous."
    )


def test_scanner_accepts_a_backed_claim(tmp_path):
    """NEGATIVE CONTROL — a backed tag must NOT be reported as unbacked."""
    ledger = ClaimsLedger(tmp_path / "claims.json")
    ledger.record("a_real_label", 1.23, run_id="control", source="test")
    planted = tmp_path / "fake_readme.md"
    planted.write_text("Value is claim:a_real_label units.", encoding="utf-8")
    assert ledger.scan_prose(planted) == [], (
        "scan_prose reported a backed claim as unbacked — it would fail on "
        "correct prose, making the discipline unusable."
    )


# ── Bare quantitative syntax: close the tag-avoidance direction ───────────────


def test_bare_quantitative_scanner_detects_each_declared_high_signal_class(tmp_path):
    """POSITIVE CONTROLS: every grammar the gate claims to cover can fail.

    The controls use factual prose, not isolated regex fragments, and include a
    line after a code fence in the companion exclusion test below.  That prevents
    a scanner that simply stops at the first fence from reading as successful.
    """

    ledger = ClaimsLedger(tmp_path / "claims.json")
    planted = tmp_path / "bare_claims.md"
    planted.write_text(
        "\n".join(
            (
                "Accuracy reached 71%.",
                "Held-out loss improved by 0.14 nats/sentence over 30 documents.",
                "The rebuild used 17 MB and took 2 minutes.",
                "The server sustained 53 calls per minute.",
                "The paired test gave p = 0.03 at n = 40.",
                "Measured wall clock was 28:08.",
            )
        ),
        encoding="utf-8",
    )

    findings = ledger.scan_bare_quantitative_claims(planted)
    observed = {(finding.category, finding.literal) for finding in findings}
    expected = {
        ("percentage", "71%"),
        ("scientific-unit", "0.14 nats/sentence"),
        ("scientific-unit", "30 documents"),
        ("operational-unit", "17 MB"),
        ("operational-unit", "2 minutes"),
        ("operational-rate", "53 calls per minute"),
        ("statistic", "p = 0.03"),
        ("statistic", "n = 40"),
        ("clock-duration", "28:08"),
    }
    assert observed == expected, (
        f"bare-claim grammar drifted; missing={expected - observed}, "
        f"unexpected={observed - expected}"
    )
    assert all(finding.path == planted and finding.line_number > 0 for finding in findings)


def test_bare_quantitative_scanner_excludes_only_documented_non_prose_forms(tmp_path):
    """Controls for code/equations plus measured source-description fields.

    A naked scalar is also planted deliberately.  Its survival documents the
    conservative boundary: this is a high-signal claims scanner, not proof that
    every untagged number in the repository is harmless.
    """

    ledger = ClaimsLedger(tmp_path / "claims.json")
    planted = tmp_path / "classified_literals.md"
    planted.write_text(
        """Library v1.2.3 was checked on 2026-08-23 for Stage 6 and Level 4.
Prior notation [@source2025] writes `$p = 0.01$`; the unexplained scalar is 3.14159.
Inline configuration `--cov-fail-under=95` and `timeout(300)` is source syntax.

- **File:** `reference.pdf` (146 pages)
  The archived source occupies 90 MB.
- **Extent:** 42 documents
- **Population:** 30 sentences
- **Exclusions:** 12 tokens
- **Source:** https://example.test/archive; corpus contains 9 documents
- **Version:** 2.4.1 (2025-06-01)

$$
p = 0.02,\\qquad n = 30
$$

```text
The example reports 88%, p = 0.04, and takes 3 minutes.
```

The phrase can wrap after Stage
3 perplexity without becoming a measured result.
The tagged result is claim:stage1_letter_perplexity nats.
""",
        encoding="utf-8",
    )

    findings = ledger.scan_bare_quantitative_claims(planted)
    assert {(finding.category, finding.literal) for finding in findings} == {
        ("operational-unit", "146 pages"),
        ("operational-unit", "90 MB"),
        ("scientific-unit", "42 documents"),
        ("scientific-unit", "30 sentences"),
        ("scientific-unit", "12 tokens"),
        ("scientific-unit", "9 documents"),
    }


def test_bare_quantitative_scanner_resumes_after_a_fence(tmp_path):
    """A fence exclusion must not disable scanning for the rest of the file."""

    ledger = ClaimsLedger(tmp_path / "claims.json")
    planted = tmp_path / "after_fence.md"
    planted.write_text("```text\naccuracy 88%\n```\nMeasured accuracy was 71%.\n", encoding="utf-8")

    findings = ledger.scan_bare_quantitative_claims(planted)
    assert [(finding.line_number, finding.literal) for finding in findings] == [(4, "71%")]


def test_bare_quantitative_scanner_scans_only_real_python_docstrings(tmp_path):
    """AST selection catches docstrings without treating executable strings as prose."""

    ledger = ClaimsLedger(tmp_path / "claims.json")
    source = tmp_path / "module.py"
    source.write_text(
        "\n".join(
            (
                '"""Accuracy was 71%."""',
                'EXECUTABLE = "Accuracy was 82%."',
                "",
                "def measured():",
                '    """The run took 3 minutes."""',
                '    return "The ignored string says 94%."',
            )
        )
        + "\n",
        encoding="utf-8",
    )

    findings = ledger.scan_bare_quantitative_claims(source)
    assert [(finding.line_number, finding.literal) for finding in findings] == [
        (1, "71%"),
        (5, "3 minutes"),
    ]


def test_bare_quantitative_scanner_rejects_unknown_source_type(tmp_path):
    """Unsupported formats fail closed instead of receiving a partial scan."""

    ledger = ClaimsLedger(tmp_path / "claims.json")
    source = tmp_path / "notes.txt"
    source.write_text("Accuracy was 71%.", encoding="utf-8")

    with pytest.raises(ValueError, match="Markdown and Python docstrings only"):
        ledger.scan_bare_quantitative_claims(source)


def test_bare_quantitative_scanner_rejects_unparseable_python(tmp_path):
    """Malformed source cannot be waved through as a docstring-free module."""

    ledger = ClaimsLedger(tmp_path / "claims.json")
    source = tmp_path / "broken.py"
    source.write_text('"""Accuracy was 71%.\n', encoding="utf-8")

    with pytest.raises(ValueError, match="cannot parse Python source"):
        ledger.scan_bare_quantitative_claims(source)


def test_bare_scan_surface_and_exclusions_are_explicit():
    """No eligible prose file may fall out of the gate accidentally."""

    candidates = {
        path.relative_to(PROJECT_ROOT).as_posix() for path in _bare_quantitative_candidates()
    }
    scanned = {
        path.relative_to(PROJECT_ROOT).as_posix() for path in _bare_quantitative_prose_files()
    }
    excluded = set(_BARE_QUANTITATIVE_EXCLUSIONS)

    assert all(_BARE_QUANTITATIVE_EXCLUSIONS.values()), "every exclusion needs a reason"
    assert candidates - scanned == excluded
    assert scanned.isdisjoint(excluded)
    for required in (
        "docs/gate_design.md",
        "docs/design/stage4_case.md",
        "docs/reference/output_conventions.md",
        "docs/guides/quickstart.md",
        "docs/troubleshooting.md",
        "README.md",
        "AGENTS.md",
        "formal/README.md",
        "gnn/models/L5_clause.gnn.md",
        "src/ailm/call_cache.py",
        "src/ailm/benchmarks/corpora.py",
    ):
        assert required in scanned, f"actionable claim surface dropped out: {required}"


# ── THE ACTUAL GATE ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def prose_claim_ledger() -> ClaimsLedger:
    """One real read-only ledger shared by the per-file prose checks."""
    return load_ledger(PROJECT_ROOT)


@pytest.mark.skipif(
    not LEDGER_FILE.exists(),
    reason=(
        "data/claims.json absent. Run scripts/record_stage0_claims.py and "
        "scripts/record_stage1_claims.py."
    ),
)
@pytest.mark.parametrize(
    "prose_path",
    _prose_files(),
    ids=lambda path: path.relative_to(PROJECT_ROOT).as_posix(),
)
def test_no_unbacked_prose_claims(prose_path: Path, prose_claim_ledger: ClaimsLedger):
    """Every claim:<label> in real prose must have a ledger entry.

    Fix by recording the measurement (scripts/record_stage*_claims.py) — never
    by deleting the tag from prose to make the gate green.
    """
    # Each source retains the ordinary per-test timeout. The complete inventory
    # is unchanged, but an aggregate filesystem-read delay no longer spends one
    # test's budget on every source, and any refusal names the offending file.
    unbacked = sorted(set(prose_claim_ledger.scan_prose(prose_path)))

    assert unbacked == [], (
        f"Unbacked prose claims in {prose_path.relative_to(PROJECT_ROOT)}: {unbacked}\n"
        f"These labels appear in project prose but have no entry in "
        f"data/claims.json. Record them from the run that produced the value:\n"
        f"  .venv/bin/python scripts/record_stage0_claims.py\n"
        f"  .venv/bin/python scripts/record_stage1_claims.py"
    )


def test_no_bare_high_signal_quantitative_claims_in_current_public_prose():
    """Measured prose uses ledger labels instead of unsupported numeric literals.

    A finding with no existing ledger label should lose its numerical precision,
    not gain a made-up label.  The exclusion inventory above is debt disclosure:
    this assertion certifies only the current documentation surface it names.
    """

    ledger = load_ledger(PROJECT_ROOT)
    findings = ledger.scan_bare_quantitative_claims(*_bare_quantitative_prose_files())
    rendered = [
        (
            f"{finding.path.relative_to(PROJECT_ROOT)}:{finding.line_number}: "
            f"{finding.category} {finding.literal!r} in {finding.line}"
        )
        for finding in findings
    ]
    assert rendered == [], "Bare high-signal quantitative claims:\n" + "\n".join(rendered)


@pytest.mark.skipif(not LEDGER_FILE.exists(), reason="data/claims.json absent")
def test_ledger_is_not_empty():
    """A ledger with no claims would make the scan trivially pass."""
    ledger = load_ledger(PROJECT_ROOT)
    assert len(ledger) > 0, "Claims ledger is empty — nothing is being tracked."


@pytest.mark.skipif(not LEDGER_FILE.exists(), reason="data/claims.json absent")
def test_all_five_cmudict_claims_match_the_committed_artifact():
    """Always-on claim evidence comes from shipped bytes, never an ignored input."""
    from ailm.corpus.cmudict import CANONICAL_ARPABET

    artifact = json.loads(CMUDICT_COUNTS_FILE.read_text(encoding="utf-8"))
    ledger = load_ledger(PROJECT_ROOT)
    expected = {
        "phoneme_count_39": len(artifact["phones"]),
        "phoneme_count_naive_unstripped": artifact["naive_phoneme_count"],
        "phoneme_count_comment_phantoms": artifact["comment_phantom_count"],
        "cmudict_base_word_count": artifact["source_base_words"],
        "cmudict_pronunciation_entry_count": artifact["source_pronunciation_entries"],
    }
    assert frozenset(artifact["phones"]) == CANONICAL_ARPABET
    for label, measured in expected.items():
        entry = ledger.get(label)
        assert entry is not None, (
            f"claim:{label} is used in prose but is unbacked. Run scripts/record_stage0_claims.py."
        )
        assert entry["value"] == measured, (
            f"claim:{label} says {entry['value']!r}, but the committed "
            f"vendor/cmudict_counts.json records {measured!r}"
        )


@pytest.mark.skipif(not LEDGER_FILE.exists(), reason="data/claims.json absent")
def test_stage0_recorder_rejects_a_planted_phantom_count_mismatch():
    """NEGATIVE CONTROL: the recorder must reject internally inconsistent evidence."""
    spec = importlib.util.spec_from_file_location("_stage0_claim_recorder", STAGE0_RECORDER)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import {STAGE0_RECORDER}")
    recorder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recorder)

    artifact = json.loads(CMUDICT_COUNTS_FILE.read_text(encoding="utf-8"))
    assert recorder.cmudict_claim_values(artifact)["phoneme_count_comment_phantoms"] == len(
        artifact["comment_phantom_tokens"]
    )

    planted = dict(artifact)
    planted["comment_phantom_count"] = artifact["comment_phantom_count"] + 1
    with pytest.raises(ValueError, match="does not equal len"):
        recorder.cmudict_claim_values(planted)


@pytest.mark.skipif(not LEDGER_FILE.exists(), reason="data/claims.json absent")
def test_every_ledger_entry_has_provenance():
    """Each entry must carry run_id and source — a bare value is not evidence."""
    ledger = load_ledger(PROJECT_ROOT)
    incomplete: list[str] = []
    for label in ledger.all_labels():
        entry = ledger.get(label) or {}
        if not entry.get("run_id") or not entry.get("source"):
            incomplete.append(label)
    assert incomplete == [], (
        f"Ledger entries missing run_id or source: {incomplete}. "
        "A recorded number without provenance cannot be re-derived."
    )


# ── Consumption surfaces beyond literal tag references ────────────────────────
#
# A recorded measurement can be genuinely spent without ever appearing as a
# literal ``claim:`` tag or a whole ``{{TOKEN}}`` in prose.  Three real
# consumption surfaces are invisible to a literal grep:
#
# 1. Generated manuscript table blocks: the prose cites ``{{BENCH_CTX_TABLES}}``
#    or ``{{LEXICAL_SWEEP_TABLE_ROWS}}``, and the labels rendered into those
#    rows are named nowhere else.
# 2. Explicit reads during derivation: ``generate_manuscript_variables`` reads
#    individual labels while computing derived quantities and display cells.
# 3. Figure modules that discover their tokens by prefix scan rather than by
#    name — ``_probes`` in ``figures/incontext_sweep.py`` matches the shape
#    ``BENCH_CTX_TOP1_K<rung>_<PROBE>`` — so no single token is named whole.
#
# The gate below unions all three surfaces with the literal references and
# judges each family by its dark share, so a family fails only when most of it
# is genuinely reached by nothing.


class _RecordingClaims(dict):
    """Ledger mapping that remembers explicit key reads during derivation.

    ``__getitem__`` and ``get`` record the key; iteration deliberately does
    not — the derivation loops over the whole ledger to emit one token per
    label, and occupying the token namespace is not a decision to cite a
    measurement.  This is an observation wrapper around the REAL validated
    ledger, not a mock: ``_load_claims`` runs unmodified and returns the real
    entries, every value the derivation consumes is the recorded one, and the
    only difference from an unobserved run is the set of keys remembered on
    the way through.
    """

    def __init__(self, entries: dict) -> None:
        super().__init__(entries)
        self.explicit_reads: set[str] = set()

    def __getitem__(self, key):
        if isinstance(key, str):
            self.explicit_reads.add(key)
        return super().__getitem__(key)

    def get(self, key, *default):
        if isinstance(key, str):
            self.explicit_reads.add(key)
        return super().get(key, *default)


_FIGURE_DIR = PROJECT_ROOT / "src" / "ailm" / "figures"

#: Uppercase fragments of token shape.  At least three characters so prose
#: debris such as ``"Agg"`` stays out, then filtered to pieces that contain an
#: underscore and reach five characters, so display text and composition keys
#: cannot become token prefixes: ``"G2P"`` and ``"L1_"`` are axis/file-path
#: words, and ``"STAGE2"`` is a family key interpolated as ``STAT_{prefix}``,
#: whose real consumption the ``STAT_`` stem already covers.
_TOKEN_PIECE_RE = re.compile(r"[A-Z][A-Z0-9_]{2,}")


def _figure_token_stems() -> set[str]:
    """Uppercase token fragments named literally by figure-module source.

    Figures consume tokens three ways: whole-name literals
    (``"BENCH_CTX_POSITIONS"``), constant prefixes inside f-strings
    (``f"BENCH_TOP1_WILSON_LOW_{model}"``), and prefix-scan regexes
    (``^BENCH_CTX_TOP1_K\\d+_([A-Z0-9]+)$``).  All three name a token through a
    string constant, and ``ast`` yields the constant parts of f-strings as
    constants of their own, so one walk over string constants recovers every
    token surface a figure can address — without importing matplotlib or
    executing any figure code.  A stem maps back to labels by prefix: a figure
    that scans ``BENCH_CTX_TOP1_K...`` genuinely consumes every rung and probe
    token that scan can return.
    """
    stems: set[str] = set()
    for path in sorted(_FIGURE_DIR.glob("*.py")):
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(module):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                stems.update(
                    piece
                    for piece in _TOKEN_PIECE_RE.findall(node.value)
                    if "_" in piece and len(piece) >= 5
                )
    return stems


class _GateData(NamedTuple):
    """Every consumption surface, kept separate so controls can audit each."""

    labels: frozenset[str]
    prose_used: frozenset[str]
    derivation_used: frozenset[str]
    figure_used: frozenset[str]
    used: frozenset[str]
    families: dict[str, list[str]]


#: The one real derivation call these tests share imports the model stack (JAX
#: reaches the generator through ``ailm.levels.phoneme``) and regenerates every
#: manuscript token, so they declare a bound far above the 60 s suite default.
_DERIVATION_TEST_TIMEOUT = 900


def _literal_prose_references(paths) -> set[str]:
    """Read literal claim references from the declared factual Markdown set."""
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    references = {token.lower() for token in re.findall(r"\{\{([A-Z0-9_]+)\}\}", text)}
    return references | set(re.findall(r"claim:([A-Za-z0-9_-]+)", text))


def test_consumption_reads_formal_prose_and_detects_removed_references(tmp_path):
    path = tmp_path / "formal" / "README.md"
    path.parent.mkdir()
    path.write_text("theorem count: claim:formal_module_control_theorem_count\n")
    assert _literal_prose_references([path]) == {"formal_module_control_theorem_count"}
    path.write_text("A description without a recorded count.\n")
    assert _literal_prose_references([path]) == set()
    assert PROJECT_ROOT / "formal" / "README.md" in _prose_files()


@functools.lru_cache(maxsize=1)
def _gate_data() -> _GateData:
    """Consume every real surface once and return the per-family raw data."""
    claims = json.loads(LEDGER_FILE.read_text(encoding="utf-8"))["claims"]
    # Consumption uses the same factual Markdown inventory as tag validity,
    # including root, GNN and formal documentation. Python strings are handled
    # by the derivation/figure surfaces below rather than counted as prose.
    prose_used = _literal_prose_references(path for path in _prose_files() if path.suffix == ".md")

    # (b) Labels explicitly read during one real derivation call.  The loader is
    # observed, never replaced: ``generate_manuscript_variables`` takes the
    # observer as its own ``claims_loader`` argument, so the real loader still
    # runs, validates, and returns the real ledger — no module attribute is
    # ever reassigned.
    import ailm.manuscript_variables as manuscript_variables

    observed_loads: list[_RecordingClaims] = []
    original_loader = manuscript_variables._load_claims

    def _observing_load(project_root: Path) -> dict:
        observed = _RecordingClaims(original_loader(project_root))
        observed_loads.append(observed)
        return observed

    manuscript_variables.generate_manuscript_variables(PROJECT_ROOT, claims_loader=_observing_load)
    assert observed_loads, "the derivation never loaded the claims mapping"
    reads: set[str] = set()
    for observed in observed_loads:
        reads |= observed.explicit_reads
    derivation_used = frozenset(reads & set(claims))

    # (c) Labels reachable through figure token scans.
    stems = _figure_token_stems()
    figure_used = frozenset(
        label for label in claims if any(label.upper().startswith(stem) for stem in stems)
    )

    families: dict[str, list[str]] = collections.defaultdict(list)
    for label in claims:
        families["_".join(label.split("_")[:2])].append(label)

    return _GateData(
        labels=frozenset(claims),
        prose_used=frozenset(prose_used),
        derivation_used=derivation_used,
        figure_used=figure_used,
        used=frozenset(prose_used | derivation_used | figure_used),
        families=dict(families),
    )


#: Families recorded from real runs and reported as measured-and-set-aside, so
#: their labels are dark by design.  One entry per family, citing the decision
#: that measured it and why the measurement stays out of the prose.  The
#: companion negative control fails any entry the ledger no longer supports,
#: so this list cannot quietly rot into a general exemption.
_MEASURED_AND_SET_ASIDE_FAMILIES: dict[str, str] = {
    "clause_word": (
        "the clause-to-word alpha sweep (decision 27 pre-registration, outcome "
        "corrected by decision 88): the non-selected sweep arms and the oracle "
        "and scrambled anchors are the measured record of why the channel stays "
        "out of the shipped stack, and the document-bootstrap cluster "
        "diagnostics of decision 89 bound an arm the manuscript describes "
        "qualitatively; the selected arm and its interval are the reported part"
    ),
    "generation_word": (
        "decision 78's third-degeneracy demonstration: the per-depth "
        "information gains, supports, and spreads instantiate the identity that "
        "action-independent transitions cancel the epistemic term; the "
        "manuscript states the degeneracy and the recorded policy-habit "
        "deviation, not each depth's number"
    ),
    "geometric_uniform": (
        "decision 47 measured the geometric-pooling rule against linear across "
        "all four message types and did not adopt it; the uniform-message arm "
        "is the no-information control of that comparison and stays in the "
        "ledger as part of the record of why the shipped rule is linear "
        "(decision 50)"
    ),
    "geometric_scrambled": (
        "decision 47's entropy-preserving scrambled-message control for the "
        "geometric-pooling comparison, recorded and set aside with the rule "
        "itself; the shipped combination rule stays linear (decision 50)"
    ),
    "stage4_adposition": (
        "decision 84's head-resolution census for the token-level adposition "
        "construct: the two cited disagreement quantities are the reported "
        "part, while the resolved/unresolved and role-agreement counts and "
        "their percentages are the audit surface that makes the citation "
        "verifiable. The construct itself is measured and set aside because "
        "English EWT supplies no morphological fidelity estimate for the "
        "ADP-token label (decision 84), and decision 63 shows the stratum's "
        "margin reverses and must never be reported as a confirmed effect"
    ),
    "stage2_g2p": (
        "the Stage 2 arm ledger covers the shipped interleave gate (the "
        "manuscript's central table), the split diagnostics, decision 74's "
        "family-disjoint bound, decision 102's family-grouped partition, and "
        "the decision 29 latent-emission arm; the remaining labels are the "
        "bounded rank rows and sweep/subsample diagnostics that make those "
        "reported parts auditable, not additional results"
    ),
    "wild_stage4": (
        "the wild-cluster bootstrap's Stage 4 arms (the sensitivity lane beside "
        "the clustered reanalysis): each arm's verdict-agreement flag is the "
        "reported part and the CI endpoints are the audit surface that makes "
        "the agreement checkable against the clustered intervals"
    ),
}


@pytest.mark.timeout(_DERIVATION_TEST_TIMEOUT)
def test_no_recorded_measurement_family_reaches_nothing():
    """A measurement mostly unreached is worse than one nobody made.

    The cluster-robust reanalysis was run at 10,000 replicates across five arms
    and wrote 99 claims to the ledger. Not one of them appeared in the
    manuscript, none was registered in `REQUIRED_CLAIMS`, and two sections
    stated in prose that no cluster-robust variant had been computed. The
    analysis existed, the paper denied it, and nothing connected the two.

    The first revision of this guard required only that SOME label of a family
    appear in prose, so the largest families passed while almost entirely
    dark — and even the dark count was wrong, because two real consumption
    paths are invisible to a literal grep: generated manuscript table blocks
    and figure modules that discover tokens by prefix.  The consumed set is
    therefore the union of three surfaces: literal ``{{TOKEN}}`` and
    ``claim:`` references in prose; labels explicitly read during one real
    ``generate_manuscript_variables`` call, observed through a recording
    mapping; and labels reachable through the token stems figure modules name
    literally, including their prefix scans.

    A family of five or more members fails when more than half of it is dark —
    reached by no surface.  A family measured and set aside on purpose is
    exempted by name in ``_MEASURED_AND_SET_ASIDE_FAMILIES``, one reason per
    family; ``test_set_aside_allowlist_entries_still_describe_dark_families``
    is the control that keeps those entries honest.

    NEGATIVE CONTROL: some families must be large, or the threshold would make
    this vacuous.
    """
    data = _gate_data()
    large = {name: labels for name, labels in data.families.items() if len(labels) >= 5}
    assert len(large) >= 10, "too few large families for this check to constrain anything"

    mostly_dark = {}
    for name, labels in large.items():
        if name in _MEASURED_AND_SET_ASIDE_FAMILIES:
            continue
        dark = sum(1 for label in labels if label not in data.used)
        if dark * 2 > len(labels):
            mostly_dark[name] = f"{dark} of {len(labels)} labels unreached"
    assert not mostly_dark, (
        "recorded measurement families that no consumption surface reaches — "
        "neither prose reference, explicit derivation read, nor figure token "
        f"scan: {mostly_dark}. Either report the result or record why it was "
        "measured and set aside in _MEASURED_AND_SET_ASIDE_FAMILIES."
    )


@pytest.mark.timeout(_DERIVATION_TEST_TIMEOUT)
def test_set_aside_allowlist_entries_still_describe_dark_families():
    """NEGATIVE CONTROL on ``_MEASURED_AND_SET_ASIDE_FAMILIES``.

    An allowlist entry asserts a family is dark on purpose, so the entry must
    match the ledger exactly: the family must exist, be large enough for the
    threshold to apply to it, and still be above that threshold.  When a
    set-aside family is later reported — most of its labels consumed by prose,
    the derivation, or a figure — the exemption stops describing reality and
    this control fails, forcing the entry's removal before the list can
    quietly cover some future dark family instead.
    """
    data = _gate_data()
    absent = sorted(set(_MEASURED_AND_SET_ASIDE_FAMILIES) - set(data.families))
    assert not absent, f"allowlisted families with no ledger entry at all: {absent}"
    stale = {}
    for name in _MEASURED_AND_SET_ASIDE_FAMILIES:
        labels = data.families[name]
        dark = sum(1 for label in labels if label not in data.used)
        if len(labels) < 5 or dark * 2 <= len(labels):
            stale[name] = f"{dark} of {len(labels)} labels unreached"
    assert not stale, (
        "stale _MEASURED_AND_SET_ASIDE_FAMILIES entries — the family is no "
        f"longer a gated, mostly-dark measurement: {stale}. Remove the entry."
    )
