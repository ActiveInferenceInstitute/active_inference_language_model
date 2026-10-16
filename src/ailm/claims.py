"""Claims ledger — the single most important guard in AILM.

Every quantitative claim in any README, docstring, or manuscript section must
have a backing entry in data/claims.json, recorded from the run that produced it.
A test fails if any prose claim tagged claim:<label> lacks a ledger entry.

This prevents hand-typed numbers from drifting away from measured ones — a failure
mode that has burned this project's codebase before.

Usage:
    ledger = load_ledger(project_root)
    ledger.record("phoneme_count_39", 39, run_id="stage0_init", source="test_cmudict.py")
    assert ledger.verify_label("phoneme_count_39")

Since schema v2 every entry may also carry an epistemic ``status`` — one of
``CLAIM_STATUSES`` (confirmatory / exploratory / descriptive / analytic) — so a
reader or a test can distinguish a pre-registered result from a test-selected
sweep without reading prose. The field is optional on write and inherited on
re-record; ``scripts/assign_claim_status.py`` assigns it to any entry missing
one, and ``tests/test_claims_status.py`` requires the real ledger to be fully
classified.

In prose (README, docstrings, manuscript):
    "The CMUdict phoneme inventory has claim:phoneme_count_39 canonical phonemes."
    The scan_prose() method finds these tags and verifies they're backed.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

# Pattern for claim references in prose: claim:<label>
_CLAIM_PATTERN = re.compile(r"claim:([A-Za-z0-9_\-]+)")
_LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*\Z")

# ``scan_prose`` proves that tags are backed, but it cannot object when prose
# avoids the protocol altogether by spelling a measured value as a digit.  The
# patterns below close that one-sided hole conservatively.  They intentionally
# name only forms with quantitative semantics in the text itself: percentages,
# measurement/population units, operational resource units, clock durations and
# conventional statistical assignments.  A naked scalar such as ``3.14159`` is
# not classified.  Treating every digit as a claim would turn dates, versions,
# stage identifiers and numbered prose into an unreviewable false-positive list.
_BARE_NUMBER = r"(?<![A-Za-z0-9_])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
_BARE_QUANTITATIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "percentage",
        re.compile(
            rf"{_BARE_NUMBER}\s*(?:%|percent(?:age)?(?:\s+points?)?)(?![A-Za-z])",
            re.IGNORECASE,
        ),
    ),
    (
        "scientific-unit",
        re.compile(
            rf"{_BARE_NUMBER}\s*(?:"
            r"nats?(?:/(?:position|sentence|token))?|bits?|"
            r"perplexity(?:\s+units?)?|positions?|tokens?|sentences?|"
            r"documents?|replicates?"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "operational-unit",
        re.compile(
            rf"{_BARE_NUMBER}\s*(?:"
            r"Ki?B|Mi?B|Gi?B|KB|MB|GB|milliseconds?|ms|seconds?|minutes?|"
            r"hours?|calls?|files?|pages?"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "operational-rate",
        re.compile(
            rf"{_BARE_NUMBER}\s+(?:calls?\s+)?(?:a|per)\s+"
            r"(?:millisecond|second|minute|hour)s?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "statistic",
        re.compile(
            r"(?<![A-Za-z0-9_])(?:p|q|n)\s*(?:<=|>=|=|<|>|≤|≥)\s*"
            r"(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?",
            re.IGNORECASE,
        ),
    ),
    ("clock-duration", re.compile(r"(?<![\d:])\d{1,3}:\d{2}(?![:\d])")),
)

_FENCE_START = re.compile(r"^\s*(`{3,}|~{3,})")
_INLINE_CODE = re.compile(r"(`+).*?\1")
_INLINE_MATH = re.compile(r"(?<!\\)\$(?!\$).*?(?<!\\)\$|\\\(.*?\\\)")
_INLINE_DISPLAY_MATH = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]")
_CITATION = re.compile(r"\[@[^\]]+\]")
_LINK_TARGET = re.compile(r"\]\([^)]*\)")
_IMAGE_ATTRIBUTES = re.compile(r"\{#[^}\n]*\bwidth\s*=\s*\d+(?:\.\d+)?%[^}\n]*\}")
_EXEMPT_LITERAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Dates and publication years are provenance, not measured outcomes.
    re.compile(r"\b(?:19|20)\d{2}(?:[-/]\d{1,2}[-/]\d{1,2})?\b"),
    # Semantic/library versions carry at least two dots.  Ordinary decimals do not.
    re.compile(r"\bv?\d+\.\d+(?:\.\d+)+\b", re.IGNORECASE),
    # Stage/level and source locators are identifiers.  Their associated result,
    # when one is stated, remains visible to the quantitative patterns.
    re.compile(
        r"\b(?:Stage|Level|L|Decision|entry|Figure|Fig\.?|Table|Equation|Eq\.?|Section)"
        r"\s*[- ]?\d+(?:\.\d+)*[A-Za-z]?\b",
        re.IGNORECASE,
    ),
)
#: On-disk schema tag for the ledger file.
_SCHEMA = "ailm-claims-v2"

#: The four epistemic statuses a claim may carry (schema v2, optional field).
#:
#: - ``confirmatory``: a prediction whose registration commit strictly predates
#:   its outcome commit per ``data/registration_provenance.json`` and that was
#:   scored under its registered rule. A PRE-REGISTRATION heading in
#:   docs/decisions.md is not sufficient: the commit order is the operative
#:   rule, and it is machine-checked against Git history.
#: - ``exploratory``: test-selected sweep results and subsample arms — anything
#:   whose headline was chosen on the evidence it is reported on.
#: - ``descriptive``: measured quantities carrying no inferential claim
#:   (counts, sizes, versions, diagnostics, un-registered measurements).
#: - ``analytic``: closed-form or derived quantities (entropy floors, uniform
#:   baselines, complexity arithmetic) — no sampling involved.
CLAIM_STATUSES = frozenset({"confirmatory", "exploratory", "descriptive", "analytic"})

_DESCRIPTION = (
    "Every quantitative claim in README, docstrings, or manuscript "
    "must have an entry here linking it to the run that produced it."
)


def _default_data() -> dict[str, Any]:
    """Return a fresh, empty ledger payload."""
    return {"_schema": _SCHEMA, "_description": _DESCRIPTION, "claims": {}}


class ClaimBatch(NamedTuple):
    """One measured run's claims and the provenance every entry in it carries.

    A recorder that splits its output across several run ids passes one of these
    per run id to :meth:`ClaimsLedger.replace_claims`, which commits all of them
    in a single write.

    ``status`` is the batch's epistemic status (one of :data:`CLAIM_STATUSES`),
    or ``None`` to record no status. ``None`` never erases: a label re-recorded
    without a status keeps the status it already has, because the status
    classifies the *design* of the measurement — a property of the arm, not of
    the run — and must survive a recorder replay unchanged.
    """

    values: Mapping[str, Any]
    run_id: str
    source: str
    status: str | None = None


class BareQuantitativeClaim(NamedTuple):
    """One high-signal quantitative literal found outside the claim protocol.

    ``category`` names which conservative grammar matched; ``literal`` is the
    exact matched text, while ``line`` preserves enough context for a human to
    decide whether to replace it with a recorded ``claim:`` label or remove an
    unsupported precision claim.  The scanner does not guess a ledger label from
    the number: two measurements can share a value without sharing provenance.
    """

    path: Path
    line_number: int
    category: str
    literal: str
    line: str


def _python_docstring_lines(path: Path) -> list[tuple[int, str]]:
    """Return physical source lines that belong to actual Python docstrings.

    AST ranges distinguish module, class and function docstrings from comments,
    ordinary string constants and executable code.  Scanning the physical lines
    preserves actionable file locators and avoids the line-number drift that
    decoded escape sequences can introduce.
    """

    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        raise ValueError(f"cannot parse Python source while scanning docstrings: {path}") from error

    line_numbers: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            continue
        end_line = first.end_lineno
        if end_line is None:  # pragma: no cover - guaranteed by supported Python versions
            raise ValueError(f"docstring node in {path} has no end position")
        line_numbers.update(range(first.lineno, end_line + 1))

    source_lines = source.splitlines()
    return [(number, source_lines[number - 1]) for number in sorted(line_numbers)]


def _quantitative_source_lines(path: Path) -> list[tuple[int, str]]:
    """Return supported reader-facing prose lines with their real locators."""

    if path.suffix.casefold() == ".md":
        return list(enumerate(path.read_text(encoding="utf-8").splitlines(), 1))
    if path.suffix.casefold() == ".py":
        return _python_docstring_lines(path)
    raise ValueError(
        f"bare quantitative claim scanning supports Markdown and Python docstrings only; got {path}"
    )


def _masked_quantitative_lines(path: Path) -> list[tuple[int, str, str]]:
    """Return prose lines after explicit non-claim forms are masked.

    The exclusions are syntactic, not a claim that the excluded material is
    quantitatively trustworthy:

    - fenced/inline code and display/inline equations are executable or formal
      notation, which have their own tests;
    - semantic versions, dates, and stage/level identifiers name artifacts
      rather than report measurements;
    - citations and link targets name evidence rather than report measurements.
      No list-item heading exempts
      its whole value or continuation: numeric ``Extent``, ``Population``,
      ``Exclusions``, ``File`` and even ``Source`` descriptions remain visible.

    Markdown is scanned as written.  Python is parsed first and only AST-proven
    module, class and function docstrings enter this line-oriented pass.
    """

    lines: list[tuple[int, str, str]] = []
    fence_marker: str | None = None
    display_math_end: str | None = None
    wrapped_identifier = False

    previous_line_number: int | None = None
    for line_number, original in _quantitative_source_lines(path):
        if previous_line_number is not None and line_number != previous_line_number + 1:
            # A gap in Python source means one docstring ended and another began.
            # No fence, equation or wrapped-list state may leak between them.
            fence_marker = None
            display_math_end = None
            wrapped_identifier = False
        previous_line_number = line_number
        stripped = original.lstrip()

        if fence_marker is not None:
            if stripped.startswith(fence_marker):
                fence_marker = None
            continue
        fence = _FENCE_START.match(original)
        if fence is not None:
            fence_marker = fence.group(1)
            continue

        if display_math_end is not None:
            if display_math_end in original:
                display_math_end = None
            continue
        if stripped.startswith("$$") and stripped.count("$$") == 1:
            display_math_end = "$$"
            continue
        if stripped.startswith(r"\[") and r"\]" not in stripped:
            display_math_end = r"\]"
            continue
        equation = re.search(r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?)\}", original)
        if equation is not None:
            display_math_end = rf"\end{{{equation.group(1)}}}"
            continue

        masked = original
        if wrapped_identifier:
            masked = re.sub(r"^\s*\d+(?:\.\d+)*[A-Za-z]?\b", " ", masked, count=1)
        wrapped_identifier = bool(re.search(r"\b(?:Stage|Level)\s*$", original, re.IGNORECASE))

        masked = _INLINE_DISPLAY_MATH.sub(" ", masked)
        masked = _INLINE_MATH.sub(" ", masked)
        masked = _INLINE_CODE.sub(" ", masked)
        masked = _CITATION.sub(" ", masked)
        masked = _LINK_TARGET.sub("]", masked)
        # Pandoc image widths are page-layout instructions, not quantitative
        # prose.  Leaving them visible produced dozens of false findings and
        # made the manuscript scanner look stricter than it was.
        masked = _IMAGE_ATTRIBUTES.sub(" ", masked)
        for pattern in _EXEMPT_LITERAL_PATTERNS:
            masked = pattern.sub(" ", masked)
        lines.append((line_number, original, masked))

    return lines


def scan_bare_quantitative_claims(*paths: str | Path) -> list[BareQuantitativeClaim]:
    """Find high-signal quantitative prose without opening or mutating a ledger.

    This is the read-only scanner behind
    :meth:`ClaimsLedger.scan_bare_quantitative_claims`. Keeping it independent of
    ledger construction matters for validation: opening a legacy v1 ledger with
    :class:`ClaimsLedger` upgrades its schema on disk, whereas a prose audit must
    never rewrite the evidence it is inspecting.
    """

    findings: list[BareQuantitativeClaim] = []
    for raw_path in paths:
        path = Path(raw_path)
        for line_number, original, masked in _masked_quantitative_lines(path):
            matches: list[tuple[int, int, str, str]] = []
            for category, pattern in _BARE_QUANTITATIVE_PATTERNS:
                matches.extend(
                    (match.start(), match.end(), category, match.group(0))
                    for match in pattern.finditer(masked)
                )
            accepted: list[tuple[int, int, str, str]] = []
            for candidate in sorted(
                matches, key=lambda item: (item[0], -(item[1] - item[0]), item[2])
            ):
                start, end, _category, _literal = candidate
                if any(
                    start < prior_end and prior_start < end
                    for prior_start, prior_end, *_ in accepted
                ):
                    continue
                accepted.append(candidate)
            for _start, _end, category, literal in sorted(accepted):
                finding = BareQuantitativeClaim(
                    path=path,
                    line_number=line_number,
                    category=category,
                    literal=literal,
                    line=original.strip(),
                )
                findings.append(finding)
                logger.warning(
                    "Bare quantitative claim in %s:%d (%s): %s",
                    path,
                    line_number,
                    category,
                    literal,
                )
    return findings


class ClaimsLedger:
    """Persistent claims ledger backed by a JSON file.

    The ledger maps claim labels to their backing entries. Each entry records
    the measured value, the run that produced it, and the source location.
    """

    def __init__(self, path: str | Path) -> None:
        """Load ledger from path, creating it if it doesn't exist.

        Args:
            path: Path to the JSON ledger file (typically data/claims.json).
        """
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = self._load()

    @staticmethod
    def _validate_text(field: str, value: object) -> str:
        """Validate non-empty provenance text without silently coercing it."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string, got {value!r}")
        return value

    @classmethod
    def _validate_entry(cls, label: object, entry: object) -> None:
        """Validate one complete wire entry, including finite JSON data."""
        if not isinstance(label, str) or _LABEL_PATTERN.fullmatch(label) is None:
            raise ValueError(f"claim labels must match [A-Za-z0-9][A-Za-z0-9_-]*, got {label!r}")
        if not isinstance(entry, dict):
            raise ValueError(f"claim {label!r} must be an object, got {type(entry).__name__}")
        missing = {"value", "run_id", "source"} - set(entry)
        if missing:
            raise ValueError(f"claim {label!r} is missing fields {sorted(missing)}")
        cls._validate_text(f"claim {label!r} run_id", entry["run_id"])
        cls._validate_text(f"claim {label!r} source", entry["source"])
        if "status" in entry and entry["status"] not in CLAIM_STATUSES:
            raise ValueError(
                f"claim {label!r} status must be one of {sorted(CLAIM_STATUSES)}, "
                f"got {entry['status']!r}"
            )
        try:
            json.dumps(entry["value"], allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"claim {label!r} value must be finite JSON data: {exc}") from exc
        if isinstance(entry["value"], float) and not math.isfinite(entry["value"]):
            raise ValueError(f"claim {label!r} value must be finite")

    def _write_atomic(self, payload: dict[str, Any]) -> None:
        """Atomically replace the ledger after a durable temporary-file write."""
        mode = self._path.stat().st_mode & 0o777 if self._path.exists() else 0o664
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(mode)
            os.replace(temporary, self._path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _load(self) -> dict[str, Any]:
        """Load the ledger from disk, repairing a structurally incomplete file.

        A schema-free skeleton (for example a seed containing just ``{}``) is
        brought up to the current shape in memory, because
        every accessor indexes ``_data["claims"]`` directly and would otherwise
        raise a bare ``KeyError``. Repair only ever *adds* the missing skeleton —
        no existing key is overwritten, so nothing recorded can be lost.

        Unparseable JSON, or a top-level value that is not an object, is a
        different situation: the file may hold data this code cannot interpret,
        so it raises instead of silently replacing it.

        Raises:
            ValueError: If the file is not valid JSON, or is valid JSON whose
                top-level value is not an object.
        """
        if not self._path.exists():
            initial = _default_data()
            self._write_atomic(initial)
            return initial

        text = self._path.read_text(encoding="utf-8")
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Claims ledger at {self._path} is not valid JSON ({exc}). "
                "Refusing to overwrite it — inspect or delete the file by hand so "
                "no recorded claim is discarded silently."
            ) from exc

        if not isinstance(raw, dict):
            raise ValueError(
                f"Claims ledger at {self._path} must contain a JSON object, got "
                f"{type(raw).__name__}. Refusing to overwrite it."
            )

        if "_schema" in raw and raw["_schema"] != _SCHEMA:
            raise ValueError(
                f"Claims ledger at {self._path} has schema {raw['_schema']!r}; "
                f"expected {_SCHEMA!r}. Refusing an implicit migration."
            )

        # Backfill only what is missing; never clobber existing content.
        repaired = False
        for key, default in _default_data().items():
            if key not in raw:
                raw[key] = default
                repaired = True
        if not isinstance(raw.get("claims"), dict):
            raise ValueError(
                f"Claims ledger at {self._path} has a 'claims' key of type "
                f"{type(raw['claims']).__name__}, expected an object. Refusing to "
                "overwrite it."
            )
        for label, entry in raw["claims"].items():
            self._validate_entry(label, entry)
        if repaired:
            self._write_atomic(raw)
            logger.info("Repaired ledger skeleton at %s", self._path)

        return raw

    def record(
        self,
        label: str,
        value: float | int | bool | str | dict,
        run_id: str,
        source: str,
        status: str | None = None,
    ) -> None:
        """Record a claim backed by a run.

        Args:
            label: Identifier used in prose as claim:<label>.
            value: The measured value (must be JSON-serializable).
            run_id: Identifier for the run that produced this value.
            source: File and/or function name that produced this value.
            status: Optional epistemic status, one of :data:`CLAIM_STATUSES`.
                ``None`` records no status for a new label and keeps the
                existing status for a re-recorded one.
        """
        self.record_many({label: value}, run_id=run_id, source=source, status=status)

    def remove_labels(self, labels: list[str] | tuple[str, ...]) -> None:
        """Atomically remove obsolete claim labels from the ledger."""
        if not isinstance(labels, (list, tuple)):
            raise ValueError("labels must be a list or tuple")
        unknown = [label for label in labels if not isinstance(label, str)]
        if unknown:
            raise ValueError(f"claim labels must be strings, got {unknown!r}")
        candidate = {**self._data, "claims": dict(self._data["claims"])}
        for label in labels:
            candidate["claims"].pop(label, None)
        self._write_atomic(candidate)
        self._data = candidate

    def _batch_entries(
        self,
        values: Mapping[str, Any],
        run_id: str,
        source: str,
        status: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Validate one batch and return its wire entries, changing no state."""
        resolved_run = self._validate_text("run_id", run_id)
        resolved_source = self._validate_text("source", source)
        if not isinstance(values, Mapping) or not values:
            raise ValueError("values must be a non-empty mapping of claim labels")

        entries: dict[str, dict[str, Any]] = {}
        for label, value in values.items():
            entry = {
                "value": value,
                "run_id": resolved_run,
                "source": resolved_source,
            }
            if status is not None:
                entry["status"] = status
            self._validate_entry(label, entry)
            entries[label] = entry
        return entries

    def _inherit_statuses(self, entries: dict[str, dict[str, Any]]) -> None:
        """Carry each existing status onto a re-record that names none.

        The status classifies the measurement's *design* (pre-registered,
        sweep-selected, descriptive, closed-form), which does not change when
        the same recorder re-runs. Without inheritance, every recorder replay
        would silently strip the field and the ledger-wide status gate would
        oscillate with whichever recorder ran last.
        """
        for label, entry in entries.items():
            if "status" in entry:
                continue
            prior = self._data["claims"].get(label)
            if isinstance(prior, dict) and "status" in prior:
                entry["status"] = prior["status"]

    def _commit(self, candidate: dict[str, Any]) -> None:
        """Prove the whole candidate serialises, then replace file and memory."""
        try:
            json.dumps(candidate, indent=2, allow_nan=False)
        except (TypeError, ValueError) as exc:  # pragma: no cover - entry guard narrows
            raise ValueError(f"claims batch is not finite JSON data: {exc}") from exc
        self._write_atomic(candidate)
        self._data = candidate

    def record_many(
        self,
        values: Mapping[str, Any],
        *,
        run_id: str,
        source: str,
        status: str | None = None,
    ) -> None:
        """Atomically record a batch from one measured run.

        Validation and JSON serialisation complete before the in-memory state or
        on-disk ledger changes.  A malformed value therefore cannot leave a
        recorder's earlier labels committed and its later labels absent.

        ``status`` (one of :data:`CLAIM_STATUSES`) applies to every label in the
        batch; ``None`` records no status for new labels and keeps the stored
        status for re-recorded ones — see :meth:`_inherit_statuses`.
        """
        entries = self._batch_entries(values, run_id, source, status)
        self._inherit_statuses(entries)
        candidate = {**self._data, "claims": dict(self._data["claims"])}
        candidate["claims"].update(entries)
        self._commit(candidate)
        logger.info(
            "Recorded %d claim(s) from run=%s source=%s",
            len(entries),
            run_id,
            source,
        )

    def replace_claims(
        self,
        *,
        remove: list[str] | tuple[str, ...] = (),
        batches: list[ClaimBatch] | tuple[ClaimBatch, ...] = (),
    ) -> None:
        """Clear labels and record batches as one atomic transition.

        A recorder that owns a namespace has to delete the labels its previous
        run produced and this run does not, then write this run's results.
        Spelling that as ``remove_labels`` followed by ``record_many`` leaves a
        window open: those are two separate whole-file replacements, so a crash
        or a kill between them commits the deletion without the replacement.
        The ledger is then short by exactly the retired labels *and* the new
        ones, and nothing on disk says so — the next reader sees a smaller
        ledger that is internally consistent.

        This method builds one candidate payload from both halves and hands it
        to a single :meth:`_write_atomic`, so the file observed by any other
        process is either entirely pre-run or entirely post-run.

        Removals apply first and batch entries second, so a label named in both
        ``remove`` and a batch ends up recorded with its new value.

        Args:
            remove: Labels to delete. Missing labels are ignored, matching
                :meth:`remove_labels`.
            batches: One :class:`ClaimBatch` per run id. Each batch's labels are
                recorded with that batch's ``run_id`` and ``source``.

        Raises:
            ValueError: If either argument has the wrong shape, if any batch
                fails the same validation :meth:`record_many` applies, if two
                batches claim the same label, or if the call would do nothing.
                Every check runs before any state changes.
        """
        if isinstance(remove, str) or not isinstance(remove, (list, tuple)):
            raise ValueError(f"remove must be a list or tuple of labels, got {remove!r}")
        not_strings = [label for label in remove if not isinstance(label, str)]
        if not_strings:
            raise ValueError(f"claim labels must be strings, got {not_strings!r}")
        if not isinstance(batches, (list, tuple)):
            raise ValueError(f"batches must be a list or tuple of ClaimBatch, got {batches!r}")

        entries: dict[str, dict[str, Any]] = {}
        for position, batch in enumerate(batches):
            if not isinstance(batch, ClaimBatch):
                raise ValueError(
                    f"batches[{position}] must be a ClaimBatch, got {type(batch).__name__}"
                )
            batch_entries = self._batch_entries(
                batch.values, batch.run_id, batch.source, batch.status
            )
            # Two run ids claiming one label is a recorder bug: whichever batch
            # lands last would silently decide the value's provenance.
            collisions = sorted(set(batch_entries) & set(entries))
            if collisions:
                raise ValueError(
                    f"batches[{position}] (run_id={batch.run_id!r}) repeats labels "
                    f"already claimed by an earlier batch: {collisions}"
                )
            entries.update(batch_entries)

        if not remove and not entries:
            raise ValueError("replace_claims needs at least one label to remove or record")

        # Inheritance reads the pre-call state on purpose: a label retired and
        # re-recorded in the same call is the same arm re-measured, so its
        # status survives unless the batch names a new one.
        self._inherit_statuses(entries)
        candidate = {**self._data, "claims": dict(self._data["claims"])}
        for label in remove:
            candidate["claims"].pop(label, None)
        candidate["claims"].update(entries)
        self._commit(candidate)
        logger.info(
            "Replaced %d claim(s) with %d claim(s) from %d run(s) in one write",
            len(remove),
            len(entries),
            len(batches),
        )

    def assign_status(self, statuses: Mapping[str, str]) -> None:
        """Set the epistemic status of existing claims in one atomic write.

        This is the classification channel, deliberately separate from
        :meth:`record_many`: it touches nothing but ``status``, so it can never
        manufacture a value, a run_id, or a source — assigning a status is a
        judgement about a measurement that already exists, not a measurement.

        Args:
            statuses: Mapping of existing claim label to a member of
                :data:`CLAIM_STATUSES`.

        Raises:
            ValueError: If the mapping is empty, names a label the ledger does
                not hold (a status for a claim that does not exist is a claim
                that does not exist), or carries an invalid status. Every check
                runs before any state changes.
        """
        if not isinstance(statuses, Mapping) or not statuses:
            raise ValueError("statuses must be a non-empty mapping of claim labels")
        unknown = sorted(label for label in statuses if label not in self._data["claims"])
        if unknown:
            raise ValueError(f"cannot assign a status to unrecorded claims: {unknown}")
        invalid = {
            label: status for label, status in statuses.items() if status not in CLAIM_STATUSES
        }
        if invalid:
            raise ValueError(f"statuses must be one of {sorted(CLAIM_STATUSES)}, got {invalid!r}")

        candidate = {**self._data, "claims": dict(self._data["claims"])}
        for label, status in statuses.items():
            candidate["claims"][label] = {**candidate["claims"][label], "status": status}
        self._commit(candidate)
        logger.info("Assigned epistemic status to %d claim(s)", len(statuses))

    def get(self, label: str) -> dict | None:
        """Retrieve a claim entry by label. Returns None if not found."""
        return self._data["claims"].get(label)

    def verify_label(self, label: str) -> bool:
        """Return True if label has a backing entry in the ledger."""
        return label in self._data["claims"]

    def scan_prose(self, *paths: str | Path) -> list[str]:
        """Scan prose files for claim:<label> tags and return unbacked labels.

        Args:
            *paths: Paths to prose files (README.md, docstrings, manuscript, etc.)

        Returns:
            List of label strings that appear in prose but have no ledger entry.
        """
        unbacked: list[str] = []
        for p in paths:
            text = Path(p).read_text(encoding="utf-8")
            for match in _CLAIM_PATTERN.finditer(text):
                label = match.group(1)
                if not self.verify_label(label):
                    unbacked.append(label)
                    logger.warning("Unbacked prose claim in %s: claim:%s", p, label)
        return unbacked

    def scan_bare_quantitative_claims(self, *paths: str | Path) -> list[BareQuantitativeClaim]:
        """Find high-signal quantitative prose that bypasses ``claim:`` tags.

        This is deliberately a conservative complement to :meth:`scan_prose`,
        not a general number detector.  It classifies bare percentages,
        scientific/population units, operational resource units and rates,
        conventional ``p``/``q``/``n`` assignments, and clock durations.  It
        does not flag naked scalars or spelled-out quantities, and it applies the
        explicit prose exclusions documented in
        :func:`_masked_quantitative_lines`.

        A match is actionable even when the same numeric value happens to exist
        somewhere in the ledger: prose must name the label so provenance, rather
        than numeric coincidence, connects the sentence to the measurement.

        Args:
            *paths: Markdown files or Python files whose docstrings to inspect.

        Returns:
            Structured findings in path, line and left-to-right match order.

        Raises:
            ValueError: If a path is neither Markdown nor Python, or if Python
                source cannot be parsed well enough to identify docstrings.
        """

        return scan_bare_quantitative_claims(*paths)

    def all_labels(self) -> list[str]:
        """Return all recorded claim labels."""
        return list(self._data["claims"].keys())

    def labels_owned_by(self, source: str) -> list[str]:
        """Return the labels whose recorded ``source`` names the same script.

        Ownership is compared on ``Path(...).name`` so an entry recorded under a
        legacy absolute path still belongs to the script that wrote it. This is
        the set a recorder may retire: a shared label prefix is a naming
        convention, not an ownership boundary, and a prefix-wide removal deletes
        another recorder's evidence on every run of this one.

        Args:
            source: The recorder's ``source`` string, typically its repo-relative
                script path.

        Returns:
            Labels in ledger order whose entry ``source`` has the same file name.

        Raises:
            ValueError: If ``source`` is empty or has no file name component.
        """
        owner = Path(self._validate_text("source", source)).name
        if not owner:
            raise ValueError(f"source must name a file, got {source!r}")
        return [
            label
            for label, entry in self._data["claims"].items()
            if Path(entry["source"]).name == owner
        ]

    def __len__(self) -> int:
        return len(self._data["claims"])


def load_ledger(base_dir: str | Path) -> ClaimsLedger:
    """Load the project claims ledger from base_dir/data/claims.json.

    Args:
        base_dir: Project root directory.

    Returns:
        ClaimsLedger instance (creates the file if it doesn't exist).
    """
    override = os.environ.get("AILM_CLAIMS_PATH", "").strip()
    if override:
        path = Path(override)
        if not path.is_absolute():
            raise ValueError(
                "AILM_CLAIMS_PATH must be absolute so recorder isolation cannot "
                f"depend on a subprocess working directory, got {override!r}"
            )
        return ClaimsLedger(path)
    return ClaimsLedger(Path(base_dir) / "data" / "claims.json")
