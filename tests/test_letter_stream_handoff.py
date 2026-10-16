"""Offline handoff tests for the full-corpus letter stream construction.

Every test drives the script's own construction path — ``construct_letter_dataset``,
``prepare_brown_stream``, and the fitted-data builders — on a small synthetic
stream injected as a parameter (the module's declared stream-source seam). No
network, no Brown download, no training, and no monkeypatching of the module
under test.
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from ailm.benchmarks.fitted.census import StreamCensus
from ailm.benchmarks.fitted.data import (
    canonicalize_pairs,
    carve_diagnostic_tail,
    refit_letter_bigram_counts,
    synthetic_letter_stream,
)
from ailm.benchmarks.fitted.receipt import DataReceiptError, verify_receipt
from ailm.benchmarks.prompts import build_prompt

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_fitted_probe_data.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("generate_fitted_probe_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # exec without registering: the loader only needs the module object, and
    # registering it would be a sys.modules mutation the dependency guard
    # rejects. Leaf-script modules do not need the registry entry.
    spec.loader.exec_module(module)
    return module


module = _load_script()
WINDOW = 6
GAP = 6
STREAM = synthetic_letter_stream(11, 480)
TARGETS = len(STREAM) - 1


def _args(**overrides):
    values = {
        "window": WINDOW,
        "gap": GAP,
        "diagnostic_size": 40_000,
        "diagnostic_stride_cap": 0,
        "epochs": 2,
        "seed": 20260910,
        "run_scope": "handoff-fixture",
    }
    values.update(overrides)
    return module.argparse.Namespace(**values)


def _split(args):
    return carve_diagnostic_tail(
        TARGETS,
        window=args.window,
        gap=args.gap,
        diagnostic_size=min(args.diagnostic_size, TARGETS // 4),
        diagnostic_stride_cap=args.diagnostic_stride_cap or None,
    )


def _expected_fit_positions(split):
    return range(1, split.fit_positions[1] + 1)


def _construct(tmp_path: Path, *, args=None, artifact_counts=None, **arg_overrides):
    args = args or _args(**arg_overrides)
    rc = module.construct_letter_dataset(
        STREAM,
        normalization="identity (pre-normalized fixture input)",
        source_bindings={},
        artifact_counts=artifact_counts,
        args=args,
        output_dir=tmp_path,
    )
    return rc, tmp_path / "letter-receipt.json"


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestCensusAndCounts:
    def test_census_totals_reconcile_independently(self, tmp_path: Path) -> None:
        rc, receipt_path = _construct(tmp_path)
        assert rc == 0
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        chunked = StreamCensus(window=WINDOW)
        for start in range(0, len(STREAM), module._CENSUS_CHUNK):
            chunked.feed(STREAM[start : start + module._CENSUS_CHUNK])
        whole = StreamCensus(window=WINDOW)
        whole.feed(STREAM)
        expected = whole.reconcile()
        # chunked feeding crosses chunk boundaries and agrees with one feed
        assert chunked.reconcile() == expected
        for key, value in expected.items():
            assert receipt["counts"][key] == value
        assert receipt["counts"]["supervised_targets"] == TARGETS
        assert receipt["counts"]["bigram_tokens"] == TARGETS

    def test_instance_counts_equal_supervised_targets(self, tmp_path: Path) -> None:
        rc, receipt_path = _construct(tmp_path)
        assert rc == 0
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        fit_lines = _lines(tmp_path / "letter-instances.jsonl")
        diagnostic_lines = _lines(tmp_path / "letter-diagnostic-instances.jsonl")
        separation = receipt["separation"]
        assert len(fit_lines) == separation["fit_positions"]
        assert len(diagnostic_lines) == separation["diagnostic_positions"]
        assert (
            len(fit_lines) + len(diagnostic_lines) + separation["gap_skipped"]
            == receipt["counts"]["supervised_targets"]
        )
        # the diagnostic file carries the carved tail, in positional order
        split = _split(_args())
        assert len(diagnostic_lines) == len(split.diagnostic_positions)

    def test_sft_lines_are_the_exact_scored_format(self, tmp_path: Path) -> None:
        rc, _receipt_path = _construct(tmp_path)
        assert rc == 0
        split = _split(_args())
        for name, positions in (
            ("letter-instances.jsonl", _expected_fit_positions(split)),
            ("letter-diagnostic-instances.jsonl", split.diagnostic_positions),
        ):
            lines = _lines(tmp_path / name)
            assert len(lines) == len(list(positions))
            for line, position in zip(lines, positions, strict=True):
                assert set(line) == {"prompt", "completion"}
                context = STREAM[max(0, position - WINDOW) : position]
                assert line["prompt"] == build_prompt("letter", context)
                assert line["completion"] == STREAM[position]


class TestSeam:
    def test_no_diagnostic_window_reaches_across_the_seam(self, tmp_path: Path) -> None:
        rc, _receipt_path = _construct(tmp_path)
        assert rc == 0
        split = _split(_args())
        fit_n, gap_skipped = (
            split.fit_positions[1],
            split.gap_positions[1] - split.gap_positions[0] + 1,
        )
        gap_hi = fit_n + gap_skipped
        for line, position in zip(
            _lines(tmp_path / "letter-diagnostic-instances.jsonl"),
            split.diagnostic_positions,
            strict=True,
        ):
            # the prompt embeds the context window; recover its reach from the
            # stream and require every reached position to be past the seam
            context = STREAM[max(0, position - WINDOW) : position]
            assert line["prompt"] == build_prompt("letter", context)
            earliest_reached = position - len(context)
            assert position >= gap_hi + 1
            assert earliest_reached > fit_n

    def test_gap_below_window_is_refused_fail_closed(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="seam invariant refused"):
            _construct(tmp_path, gap=2)

    def test_stride_cap_carves_a_declared_subsample(self, tmp_path: Path) -> None:
        args = _args(diagnostic_stride_cap=3)
        rc, receipt_path = _construct(tmp_path, args=args)
        assert rc == 0
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        split = _split(args)
        tail_size = len(split.diagnostic_positions)  # capped carve
        uncapped = carve_diagnostic_tail(
            TARGETS, window=WINDOW, gap=GAP, diagnostic_size=min(40_000, TARGETS // 4)
        )
        skipped = len(uncapped.diagnostic_positions) - tail_size
        assert receipt["separation"]["diagnostic_stride_skipped"] == skipped
        assert len(_lines(tmp_path / "letter-diagnostic-instances.jsonl")) == tail_size


class TestRefitCounts:
    def test_refit_counts_reconcile_with_the_fit_region(self, tmp_path: Path) -> None:
        rc, _receipt_path = _construct(tmp_path)
        assert rc == 0
        split = _split(_args())
        expected = canonicalize_pairs(refit_letter_bigram_counts(STREAM, split.fit_positions))
        written = json.loads((tmp_path / "letter-refit-bigram-counts.json").read_text("utf-8"))
        assert written == expected
        # manual cross-check: bigrams ending at each fit position
        manual: Counter[tuple[str, str]] = Counter()
        for i in _expected_fit_positions(split):
            manual[(STREAM[i - 1], STREAM[i])] += 1
        assert written == canonicalize_pairs(dict(manual))


class TestReceipt:
    def test_receipt_verifies_by_recomputation(self, tmp_path: Path) -> None:
        rc, receipt_path = _construct(tmp_path)
        assert rc == 0
        report = verify_receipt(receipt_path, data_dir=tmp_path)
        assert report["verified_files"] == 3
        assert report["supervised_targets"] == TARGETS

    def test_outputs_are_write_once(self, tmp_path: Path) -> None:
        rc, _receipt_path = _construct(tmp_path)
        assert rc == 0
        before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
        with pytest.raises(DataReceiptError, match="refusing to overwrite"):
            _construct(tmp_path)
        after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
        assert before == after


class TestArtifactReconciliationSeam:
    def test_matching_artifact_counts_construct(self, tmp_path: Path) -> None:
        census = StreamCensus(window=WINDOW)
        census.feed(STREAM)
        rc, _receipt_path = _construct(
            tmp_path, artifact_counts=canonicalize_pairs(census.bigram_counts)
        )
        assert rc == 0

    def test_mismatched_artifact_counts_refuse_before_any_write(self, tmp_path: Path) -> None:
        census = StreamCensus(window=WINDOW)
        census.feed(STREAM)
        wrong = canonicalize_pairs(census.bigram_counts)
        first_key = next(iter(wrong))
        wrong[first_key] += 1
        rc = module.construct_letter_dataset(
            STREAM,
            normalization="identity (pre-normalized fixture input)",
            source_bindings={},
            artifact_counts=wrong,
            args=_args(),
            output_dir=tmp_path,
        )
        assert rc == 1
        assert not (tmp_path / "letter-instances.jsonl").exists()
        assert not (tmp_path / "letter-diagnostic-instances.jsonl").exists()


class TestBrownStreamSeam:
    def test_fetch_failure_refuses(self, capsys: pytest.CaptureFixture[str]) -> None:
        def missing_zip() -> str:
            raise FileNotFoundError("pinned brown.zip is absent")

        prepared = module.prepare_brown_stream(stream_builder=missing_zip)
        assert prepared is None
        assert "authenticated Brown fetch failed" in capsys.readouterr().err

    def test_source_integrity_failure_refuses(self, capsys: pytest.CaptureFixture[str]) -> None:
        from ailm.corpus.brown import BrownSourceIntegrityError

        def tampered_zip() -> str:
            raise BrownSourceIntegrityError("bytes do not match the canonical pin")

        prepared = module.prepare_brown_stream(stream_builder=tampered_zip)
        assert prepared is None
        assert "authenticated Brown fetch failed" in capsys.readouterr().err

    def test_injected_stream_carries_the_recorded_normalization(self) -> None:
        prepared = module.prepare_brown_stream(stream_builder=lambda: STREAM)
        assert prepared is not None
        stream, normalization, bindings, recorded = prepared
        assert stream == STREAM
        artifact = json.loads(
            (Path(__file__).resolve().parents[1] / "vendor" / "brown_counts.json").read_text(
                "utf-8"
            )
        )
        assert normalization == artifact["provenance"]["normalization"]
        assert recorded == artifact["counts"]
        assert set(bindings) == {"brown_raw_source", "brown_counts_artifact"}
        for digest in bindings.values():
            assert digest.startswith("sha256:") and len(digest) == 71
