"""Foundation tests for the fitted-probe data/identity modules.

Offline, deterministic, mock-free: every test runs against real in-memory or
tmp_path fixtures, and the script tests run the real CLI as a subprocess with
true exit codes. No network, no Ollama, no training, no historical namespace
mutation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from ailm.benchmarks.constants import PROBE_MODEL_TAGS, SERVED_PROBES
from ailm.benchmarks.fitted import (
    FITTED_PROBES,
    CheckpointSpec,
    DataReceipt,
    DataReceiptError,
    FittedIdentityError,
    FittedProbeSpec,
    StreamCensus,
    build_lineage_chain,
    sha256_text,
    stride_aliasing_demo,
    validate_fitted_probes,
    verify_receipt,
    write_receipt,
)
from ailm.benchmarks.fitted.data import (
    build_g2p_instances,
    build_letter_instances,
    build_phoneme_stream_instances,
    canonicalize_pairs,
    carve_diagnostic_tail,
    g2p_train_test_pairs,
    reconcile_contexts,
    refit_letter_bigram_counts,
    synthetic_letter_stream,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORDER = PROJECT_ROOT / "scripts" / "generate_fitted_probe_data.py"
GOOD_DIGEST = "sha256:" + "a" * 64
OTHER_DIGEST = "sha256:" + "b" * 64


def _spec(**overrides: object) -> FittedProbeSpec:
    values: dict[str, object] = {
        "key": "smollm_fit",
        "lineage_family": "smollm",
        "display_name": "SmolLM 1.7B fitted (letter)",
        "tasks": ("letter",),
        "upstream_repo": "HuggingFaceTB/SmolLM-1.7B",
        "upstream_revision": GOOD_DIGEST,
        "precision": "bf16",
        "served_tag": "ailm-fit/smollm-letter:Q4_K_M",
    }
    values.update(overrides)
    return FittedProbeSpec(**values)  # type: ignore[arg-type]


class TestIdentity:
    def test_empty_registry_and_untouched_served_surface(self) -> None:
        assert FITTED_PROBES == ()
        assert [probe.key for probe in SERVED_PROBES] == ["gemma3", "qwen25", "smollm"]
        assert PROBE_MODEL_TAGS["smollm"] == "smollm:1.7b-base-v0.2-q4_K_M"

    def test_unknown_lineage_family_rejected(self) -> None:
        with pytest.raises(FittedIdentityError, match="not a registered probe key"):
            validate_fitted_probes(
                (_spec(lineage_family="gpt4"),),
                registered_keys=frozenset({"smollm"}),
                registered_tags=frozenset({"gemma3:4b"}),
            )

    def test_served_tag_collision_rejected(self) -> None:
        with pytest.raises(FittedIdentityError, match="collides with a registered probe tag"):
            validate_fitted_probes(
                (_spec(served_tag="gemma3:4b"),),
                registered_keys=frozenset({"smollm"}),
                registered_tags=frozenset({"gemma3:4b"}),
            )

    def test_malformed_digest_rejected(self) -> None:
        with pytest.raises(FittedIdentityError):
            validate_fitted_probes(
                (_spec(upstream_revision="sha256:xyz"),),
                registered_keys=frozenset({"smollm"}),
                registered_tags=frozenset(set()),
            )

    def test_duplicate_key_rejected(self) -> None:
        with pytest.raises(FittedIdentityError, match="duplicate fitted key"):
            validate_fitted_probes(
                (_spec(), _spec(served_tag="ailm-fit/other:Q8_0")),
                registered_keys=frozenset({"smollm"}),
                registered_tags=frozenset(set()),
            )

    def test_tasks_must_be_declared_subset(self) -> None:
        with pytest.raises(FittedIdentityError, match="ordered subset"):
            validate_fitted_probes(
                (_spec(tasks=("letter", "bogus")),),  # type: ignore[arg-type]
                registered_keys=frozenset({"smollm"}),
                registered_tags=frozenset(set()),
            )

    def test_lineage_chain_requires_all_stages_in_order(self) -> None:
        with pytest.raises(FittedIdentityError, match="missing stages"):
            build_lineage_chain({"upstream": GOOD_DIGEST})
        with pytest.raises(FittedIdentityError, match="unknown stages"):
            build_lineage_chain(
                {
                    stage: GOOD_DIGEST
                    for stage in ("upstream", "adapter", "fused", "converted", "served")
                }
                | {"behavioral": GOOD_DIGEST}
            )
        with pytest.raises(FittedIdentityError, match="adapter"):
            stages = {
                "upstream": GOOD_DIGEST,
                "adapter": "nothex",
                "fused": GOOD_DIGEST,
                "converted": GOOD_DIGEST,
                "served": GOOD_DIGEST,
            }
            build_lineage_chain(stages)
        chain = build_lineage_chain(
            {
                stage: GOOD_DIGEST
                for stage in ("upstream", "adapter", "fused", "converted", "served")
            }
        )
        assert [stage for stage, _ in chain] == [
            "upstream",
            "adapter",
            "fused",
            "converted",
            "served",
        ]

    def test_digest_helpers_deterministic(self) -> None:
        assert sha256_text("abc") == sha256_text("abc")
        assert sha256_text("abc") != sha256_text("abd")


class TestCensus:
    def test_exact_counts_match_pair_counter(self) -> None:
        stream = synthetic_letter_stream(7, 200)
        census = StreamCensus(window=5)
        census.feed(stream)
        totals = census.reconcile()
        assert totals["supervised_targets"] == 199
        expected = Counter(zip(stream, stream[1:], strict=False))
        assert dict(census.bigram_counts) == dict(expected)

    def test_window_truncation_at_stream_start(self) -> None:
        census = StreamCensus(window=3)
        census.feed("abcd")
        totals = census.reconcile()
        assert totals["supervised_targets"] == 3
        assert census.context_occurrences["a"] == 1  # first target's context is just "a"

    def test_chunked_feeding_matches_single_feed(self) -> None:
        stream = synthetic_letter_stream(11, 150)
        single = StreamCensus(window=4)
        single.feed(stream)
        chunked = StreamCensus(window=4)
        for i in range(0, len(stream), 7):
            chunked.feed(stream[i : i + 7])
        assert single.reconcile() == chunked.reconcile()
        assert single.bigram_counts == chunked.bigram_counts

    def test_stride2_aliasing_counterexample(self) -> None:
        census_freq, strided_freq = stride_aliasing_demo()
        assert census_freq[("a", "b")] > 0 and census_freq[("b", "a")] > 0
        assert set(strided_freq) == {("b", "a")}
        assert census_freq != strided_freq


class TestData:
    def test_letter_instances_exact_and_ordered(self) -> None:
        instances = build_letter_instances("abcd", window=2)
        assert [r["position_index"] for r in instances] == [1, 2, 3]
        assert instances[0]["target"] == "b" and instances[0]["context_window"] == "a"
        assert instances[2]["context_window"] == "bc"

    def test_carve_reconciles_and_keeps_sets_disjoint(self) -> None:
        split = carve_diagnostic_tail(100, window=10, gap=10, diagnostic_size=20)
        counts = split.counts()
        assert sum(counts.values()) == 100
        fit = set(range(split.fit_positions[0], split.fit_positions[1] + 1))
        diag = set(split.diagnostic_positions)
        gap = set(range(split.gap_positions[0], split.gap_positions[1] + 1))
        assert not (fit & diag) and not (fit & gap) and not (diag & gap)
        assert min(diag) > max(gap)

    def test_carve_stride_cap_thins_diagnostic_only(self) -> None:
        full = carve_diagnostic_tail(100, window=10, gap=10, diagnostic_size=20)
        capped = carve_diagnostic_tail(
            100, window=10, gap=10, diagnostic_size=20, diagnostic_stride_cap=2
        )
        assert capped.fit_positions == full.fit_positions
        assert len(capped.diagnostic_positions) < len(full.diagnostic_positions)
        assert set(capped.diagnostic_positions) <= set(full.diagnostic_positions)

    def test_carve_refuses_invalid_shapes(self) -> None:
        with pytest.raises(ValueError):
            carve_diagnostic_tail(100, window=10, gap=0, diagnostic_size=20)
        with pytest.raises(ValueError):
            carve_diagnostic_tail(5, window=10, gap=10, diagnostic_size=20)

    def test_carve_seam_invariant_exact_gap_accepted(self) -> None:
        split = carve_diagnostic_tail(100, window=10, gap=10, diagnostic_size=20)
        assert split.gap_positions == (71, 80)
        assert split.diagnostic_positions[0] == 81

    def test_carve_seam_invariant_below_gap_refused(self) -> None:
        with pytest.raises(ValueError, match="seam invariant refused"):
            carve_diagnostic_tail(100, window=10, gap=9, diagnostic_size=20)

    def test_carve_refuses_sub_window_window(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 1"):
            carve_diagnostic_tail(100, window=0, gap=10, diagnostic_size=20)

    def test_refit_counts_are_fit_region_only(self) -> None:
        stream = "ab" * 8 + "xy"  # tail contributes bigrams absent from the fit region
        split = carve_diagnostic_tail(len(stream) - 1, window=2, gap=2, diagnostic_size=4)
        refit = refit_letter_bigram_counts(stream, split.fit_positions)
        fit_hi = split.fit_positions[1]
        expected = Counter(zip(stream[:fit_hi], stream[1 : fit_hi + 1], strict=False))
        assert refit == dict(expected)

    @pytest.mark.timeout(180)
    def test_g2p_split_reuse_honors_stride(self) -> None:
        pairs = [(f"word{i}", "A B C") for i in range(10)]
        train, test = g2p_train_test_pairs(pairs, test_every=5)
        assert [word for word, _ in test] == ["word0", "word5"]
        assert len(train) == 8

    def test_g2p_instances_are_strict_1to1(self) -> None:
        with pytest.raises(ValueError, match="not 1:1"):
            build_g2p_instances([("abc", ["AA", "AE"])])
        instances = build_g2p_instances([("ab", ["AA", "AE"])])
        assert instances == [{"pair_index": 0, "word": "ab", "gold": ["AA", "AE"]}]

    def test_phoneme_stream_instances_reset_per_pair(self) -> None:
        train = [("ab", ["AA", "AE"]), ("cde", ["B", "D", "K"])]
        instances = build_phoneme_stream_instances(train)
        assert instances[0] == {"pair_index": 0, "position": 1, "prefix": "AA", "target": "AE"}
        assert instances[1] == {"pair_index": 1, "position": 1, "prefix": "B", "target": "D"}
        assert instances[2] == {"pair_index": 1, "position": 2, "prefix": "B D", "target": "K"}
        reconciliation = reconcile_contexts(instances, key="prefix")
        assert reconciliation["supervised_targets"] == 3

    def test_canonicalize_pairs_uses_vendored_key_format(self) -> None:
        assert canonicalize_pairs({("a", "b"): 2}) == {"a,b": 2}


class TestReceipt:
    def _receipt(self, files: dict[str, str]) -> DataReceipt:
        return DataReceipt(
            dataset="letter",
            run_scope="unit-scope",
            policy={"window": 1, "gap": 2, "epochs": 2, "seed": 1, "diagnostic_stride_cap": 0},
            source_bindings={"declared_stream": GOOD_DIGEST},
            separation={"fit_positions": 5, "gap_skipped": 2, "diagnostic_positions": 3},
            counts={"supervised_targets": 10, "bigram_tokens": 10, "distinct_contexts": 7},
            files=files,
            generator="test",
        )

    def test_receipt_seam_invariant_below_gap_rejected(self) -> None:
        with pytest.raises(DataReceiptError, match="seam invariant refused"):
            DataReceipt(
                dataset="letter",
                run_scope="s",
                policy={"window": 40, "gap": 10, "epochs": 2, "seed": 1},
                source_bindings={"x": GOOD_DIGEST},
                separation={"fit_positions": 10, "gap_skipped": 0, "diagnostic_positions": 0},
                counts={"supervised_targets": 10, "bigram_tokens": 10},
                files={},
                generator="test",
            )

    def test_receipt_seam_invariant_exact_gap_accepted(self) -> None:
        receipt = self._receipt({})
        assert receipt.policy["gap"] >= receipt.policy["window"]
        assert receipt.separation["gap_skipped"] == receipt.policy["gap"]

    def test_round_trip_write_load_verify(self, tmp_path: Path) -> None:
        (tmp_path / "a.jsonl").write_text("one\ntwo\n", encoding="utf-8")
        (tmp_path / "b.jsonl").write_text("three\n", encoding="utf-8")
        from ailm.benchmarks.fitted import sha256_digest

        receipt = self._receipt(
            {
                "a.jsonl": sha256_digest((tmp_path / "a.jsonl").read_bytes()),
                "b.jsonl": sha256_digest((tmp_path / "b.jsonl").read_bytes()),
            }
        )
        path = write_receipt(tmp_path / "receipt.json", receipt)
        loaded = DataReceipt.from_dict(json.loads(path.read_text(encoding="utf-8")))
        assert loaded.run_scope == "unit-scope"
        report = verify_receipt(path, data_dir=tmp_path)
        assert report["verified_files"] == 2
        assert report["supervised_targets"] == 10

    def test_verify_catches_tampering(self, tmp_path: Path) -> None:
        (tmp_path / "a.jsonl").write_text("one\n", encoding="utf-8")
        from ailm.benchmarks.fitted import sha256_digest

        receipt = self._receipt({"a.jsonl": sha256_digest((tmp_path / "a.jsonl").read_bytes())})
        path = write_receipt(tmp_path / "receipt.json", receipt)
        (tmp_path / "a.jsonl").write_text("tampered\n", encoding="utf-8")
        with pytest.raises(DataReceiptError, match="digest mismatch"):
            verify_receipt(path, data_dir=tmp_path)

    def test_verify_catches_missing_file(self, tmp_path: Path) -> None:
        receipt = self._receipt({"missing.jsonl": GOOD_DIGEST})
        path = write_receipt(tmp_path / "receipt.json", receipt)
        with pytest.raises(DataReceiptError, match="missing file"):
            verify_receipt(path, data_dir=tmp_path)

    def test_write_refuses_overwrite(self, tmp_path: Path) -> None:
        path = write_receipt(tmp_path / "receipt.json", self._receipt({}))
        with pytest.raises(DataReceiptError, match="refusing to overwrite"):
            write_receipt(path, self._receipt({}))

    def test_non_reconciling_receipt_rejected(self) -> None:
        with pytest.raises(DataReceiptError, match="separation"):
            self._receipt({}).__class__(
                dataset="letter",
                run_scope="s",
                policy={"window": 2},
                source_bindings={"x": GOOD_DIGEST},
                separation={"fit_positions": 1, "gap_skipped": 0, "diagnostic_positions": 0},
                counts={"supervised_targets": 10, "bigram_tokens": 10},
                files={},
                generator="test",
            )

    def test_malformed_digest_rejected(self) -> None:
        with pytest.raises(DataReceiptError, match="sha256"):
            self._receipt({"f": "deadbeef"})

    def test_from_dict_rejects_unknown_and_missing_fields(self) -> None:
        base = json.loads(json.dumps(self._receipt({}).to_dict()))
        with pytest.raises(DataReceiptError, match="unknown fields"):
            DataReceipt.from_dict({**base, "extra": 1})
        with pytest.raises(DataReceiptError, match="missing fields"):
            DataReceipt.from_dict({k: v for k, v in base.items() if k != "counts"})

    def test_checkpoint_spec_validation_and_round_trip(self) -> None:
        checkpoint = CheckpointSpec(
            dataset="letter",
            data_receipt_digest=GOOD_DIGEST,
            adapter_path="adapters/letter.safetensors",
            optimizer_step=10,
            epochs_completed=1,
            trainer="mlx_lm.lora",
        )
        assert CheckpointSpec.from_dict(checkpoint.to_dict()) == checkpoint
        with pytest.raises(DataReceiptError, match="data_receipt_digest"):
            CheckpointSpec(
                dataset="letter",
                data_receipt_digest="bad",
                adapter_path="a",
                optimizer_step=0,
                epochs_completed=0,
                trainer="t",
            )
        with pytest.raises(DataReceiptError, match="optimizer_step"):
            CheckpointSpec(
                dataset="letter",
                data_receipt_digest=GOOD_DIGEST,
                adapter_path="a",
                optimizer_step=-1,
                epochs_completed=0,
                trainer="t",
            )


def _run_script(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RECORDER), *args], capture_output=True, text=True, timeout=960
    )


class TestScript:
    @pytest.mark.timeout(1920)
    def test_letter_fixture_end_to_end_and_immutability(self, tmp_path: Path) -> None:
        stream_path = tmp_path / "stream.txt"
        stream_path.write_text(synthetic_letter_stream(3, 600), encoding="utf-8")
        out = tmp_path / "out"
        first = _run_script(
            [
                "--dataset",
                "letter",
                "--stream-file",
                str(stream_path),
                "--run-scope",
                "fixture-scope",
                "--output-dir",
                str(out),
            ]
        )
        assert first.returncode == 0, first.stderr
        report = json.loads(first.stdout)
        assert report["verified_files"] == 3
        assert report["supervised_targets"] == 599
        second = _run_script(
            [
                "--dataset",
                "letter",
                "--stream-file",
                str(stream_path),
                "--run-scope",
                "fixture-scope",
                "--output-dir",
                str(out),
            ]
        )
        assert second.returncode == 2
        assert "refusing to overwrite" in second.stderr

    @pytest.mark.timeout(1920)
    def test_letter_full_corpus_gate_refuses(self, tmp_path: Path) -> None:
        result = _run_script(
            [
                "--dataset",
                "letter",
                "--stream-source",
                "nltk-brown",
                "--run-scope",
                "fixture-scope",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        assert result.returncode == 2
        assert "gated" in result.stderr

    @pytest.mark.timeout(1920)
    def test_letter_below_gap_refuses_at_cli(self, tmp_path: Path) -> None:
        stream_path = tmp_path / "stream.txt"
        stream_path.write_text(synthetic_letter_stream(5, 600), encoding="utf-8")
        result = _run_script(
            [
                "--dataset",
                "letter",
                "--stream-file",
                str(stream_path),
                "--run-scope",
                "fixture-scope",
                "--output-dir",
                str(tmp_path / "out"),
                "--gap",
                "10",
            ]
        )
        assert result.returncode == 2
        assert "seam invariant refused" in result.stderr

    @pytest.mark.timeout(1920)
    def test_letter_exact_gap_runs_end_to_end(self, tmp_path: Path) -> None:
        stream_path = tmp_path / "stream.txt"
        stream_path.write_text(synthetic_letter_stream(6, 600), encoding="utf-8")
        result = _run_script(
            [
                "--dataset",
                "letter",
                "--stream-file",
                str(stream_path),
                "--run-scope",
                "fixture-scope",
                "--output-dir",
                str(tmp_path / "out"),
                "--gap",
                "40",
            ]
        )
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["verified_files"] == 3

    @pytest.mark.timeout(1920)
    def test_g2p_real_vendored_source(self, tmp_path: Path) -> None:
        result = _run_script(
            [
                "--dataset",
                "g2p",
                "--run-scope",
                "fixture-scope",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        receipt_path = tmp_path / "out" / "g2p-receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["dataset"] == "g2p"
        assert receipt["separation"]["fit_positions"] == receipt["counts"]["supervised_targets"]
        assert report["supervised_targets"] == receipt["counts"]["supervised_targets"]
        assert set(receipt["source_bindings"]) == {"cmudict_counts"}
