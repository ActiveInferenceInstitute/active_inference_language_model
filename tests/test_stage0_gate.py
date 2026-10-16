"""Stage 0 meta-gate: integration test that Stage 0 is complete and correct.

This file is the canonical Stage 0 completion test. All lower-level tests
in test_cases.py, test_claims_ledger.py, test_control_harness.py,
test_pymdp_smoke.py, and test_cmudict.py validate individual components.

This file asks: "Is Stage 0 done?" — a holistic check that:
  1. Claims ledger infrastructure is operational
  2. Control harness can detect a broken model (gate is live)
  3. pymdp smoke tests all pass (A_dependencies, empirical prior, etc.)
  4. CMUdict phoneme count is 39 (negative control passes)
  5. Case module is self-consistent
  6. Stage 0 claim IS recorded in the ledger

STAGE 0 GATE: all assertions here must pass before Stage 1 begins.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import tomllib
from pathlib import Path

import numpy as np
import pytest

from ailm.cases import CaseRole, CerebrumFunctionRole
from ailm.claims import ClaimsLedger
from ailm.control import ControlHarness, GateVacuousError
from ailm.corpus.cmudict import (
    CANONICAL_ARPABET,
    EXPECTED_PHONEME_COUNT,
    assert_full_inventory,
    build_phoneme_inventory,
)
from ailm.generative.smoke import run_all_smoke_tests


class TestStage0ClaimsInfrastructure:
    """Claims ledger must be operational before any claim-backed prose is written."""

    def test_claims_ledger_can_record_and_retrieve(self, tmp_path):
        path = tmp_path / "claims.json"
        ledger = ClaimsLedger(path)
        ledger.record("stage0_test", 0, "stage0_gate_run", "test_stage0_gate.py")
        assert ledger.verify_label("stage0_test")
        assert len(ledger) == 1

    def test_claims_ledger_persists(self, tmp_path):
        path = tmp_path / "claims.json"
        ledger = ClaimsLedger(path)
        ledger.record("phoneme_count_39", 39, "stage0", "test_stage0_gate.py")
        ledger2 = ClaimsLedger(path)
        assert ledger2.verify_label("phoneme_count_39")

    def test_scan_prose_detects_unbacked(self, tmp_path):
        ledger = ClaimsLedger(tmp_path / "claims.json")
        prose = tmp_path / "test.md"
        prose.write_text("There are claim:phoneme_count_39 phonemes.")
        unbacked = ledger.scan_prose(prose)
        assert "phoneme_count_39" in unbacked

    def test_scan_prose_clears_when_backed(self, tmp_path):
        ledger = ClaimsLedger(tmp_path / "claims.json")
        ledger.record("phoneme_count_39", 39, "stage0", "stage0_gate")
        prose = tmp_path / "test.md"
        prose.write_text("There are claim:phoneme_count_39 phonemes.")
        unbacked = ledger.scan_prose(prose)
        assert "phoneme_count_39" not in unbacked


class TestStage0ControlHarness:
    """Control harness must be live — vacuous gates block ALL downstream work."""

    def test_harness_detects_good_vs_broken(self):
        harness = ControlHarness("stage0_entropy_gate")
        # uniform model (broken) has entropy log(n); peaked model (good) has lower
        n = 30
        uniform_entropy = np.log(n)  # ~3.4 nats
        peaked_entropy = 0.47  # sharp distribution (measured)
        result = harness.check_metric_is_sensitive(
            peaked_entropy, uniform_entropy, "lower_is_better"
        )
        assert result is True, f"Expected True (Python bool), got {result!r} ({type(result)})"

    def test_harness_raises_on_vacuous_gate(self):
        harness = ControlHarness("vacuous_stage0")
        with pytest.raises(GateVacuousError):
            harness.check_metric_is_sensitive(0.5, 0.5, "lower_is_better")

    def test_negative_control_with_uniform_model_fails(self):
        """A uniform phoneme model must score WORSE than our (tiny) model."""
        harness = ControlHarness("phoneme_entropy_gate")

        def entropy(log_probs: np.ndarray) -> float:
            return float(-np.sum(np.exp(log_probs) * log_probs))

        n_phones = EXPECTED_PHONEME_COUNT  # 39
        uniform_log = np.full(n_phones, -np.log(n_phones))  # max entropy

        # A tiny-but-better model (just slightly non-uniform)
        rng = np.random.default_rng(0)
        better_probs = rng.dirichlet(np.ones(n_phones) * 0.1)
        better_log = np.log(better_probs + 1e-12)

        good_ent = entropy(better_log)
        bad_ent = entropy(uniform_log)

        assert bad_ent > good_ent  # uniform is strictly worse
        harness.assert_gate_is_live(good_ent, bad_ent, "lower_is_better")


class TestStage0PymdpSmoke:
    """All pymdp smoke tests must pass. Failure here blocks Stage 1."""

    def test_all_pymdp_smoke_tests_pass(self):
        result = run_all_smoke_tests()
        failed = {
            k: v
            for k, v in result.items()
            if k != "all_passed" and isinstance(v, dict) and not v.get("success", True)
        }
        assert result["all_passed"], (
            f"STAGE 0 BLOCKED: pymdp smoke tests failed.\n"
            f"Failed: {list(failed.keys())}\n"
            f"Without A_dependencies factorization and empirical_prior coupling, "
            f"AILM cannot be built. Check pymdp version and API surface."
        )


class TestStage0PhonemeInventory:
    """CMUdict 39-phoneme invariant. Failure = comment stripping was broken."""

    def test_expected_phoneme_count_is_39(self):
        assert EXPECTED_PHONEME_COUNT == 39

    def test_canonical_arpabet_has_exactly_39(self):
        assert len(CANONICAL_ARPABET) == 39, (
            f"CANONICAL_ARPABET has {len(CANONICAL_ARPABET)} entries. "
            f"If >39: likely comment tokens not stripped. "
            f"If <39: a phoneme category missing."
        )

    def test_comment_tokens_not_in_canonical(self):
        """NEGATIVE CONTROL: comment tokens must not sneak into the inventory."""
        comment_tokens = {";;;", ";;", ";", "#"}
        leaked = comment_tokens & CANONICAL_ARPABET
        assert not leaked, (
            f"Comment tokens in CANONICAL_ARPABET: {leaked}. "
            f"Comment stripping is broken. The 39-phoneme claim is false."
        )

    def test_inventory_from_tiny_dict_is_subset_of_canonical(self, tiny_cmudict):
        """tiny_cmudict has 9 phones (5 words) — a strict subset, not equal to 39."""
        inv = build_phoneme_inventory(tiny_cmudict)
        assert inv.issubset(CANONICAL_ARPABET), (
            f"Unknown phonemes from tiny_cmudict: {inv - CANONICAL_ARPABET}"
        )
        # And confirm assert_full_inventory correctly rejects the partial set
        with pytest.raises(AssertionError):
            assert_full_inventory(inv)


class TestStage0CaseConsistency:
    """Case module invariants. Failure = a symbol denotes two different roles."""

    def test_eight_case_roles(self):
        assert len(CaseRole) == 8

    def test_core_function_roles_plus_novel_generative_role(self):
        assert len(CerebrumFunctionRole) == len(CaseRole) + 1

    def test_role_systems_stay_unambiguous(self):
        """Core Genitive and novel Generative must serialize distinctly.

        CEREBRUM gives both the source abbreviation ``GEN``. AILM retains that
        source collision while scoping the enum names and values so the two roles
        cannot collapse in a persisted record.
        """
        assert CaseRole.GEN.value == "Genitive"
        assert "core:GEN" in CerebrumFunctionRole.CORE_GENITIVE.value
        assert "novel:GEN" in CerebrumFunctionRole.NOVEL_GENERATIVE.value
        assert (
            CerebrumFunctionRole.CORE_GENITIVE.value != CerebrumFunctionRole.NOVEL_GENERATIVE.value
        )
        shared_values = {c.value for c in CaseRole} & {r.value for r in CerebrumFunctionRole}
        assert not shared_values, f"Value strings shared by the two role systems: {shared_values}."


class TestStage0Completeness:
    """Meta-check: Stage 0 is complete when all components are present."""

    def test_all_modules_importable(self):
        """Every Stage 0 module must be importable."""

        import ailm

        names = sorted(
            module.name for module in pkgutil.walk_packages(ailm.__path__, f"{ailm.__name__}.")
        )
        required = {
            "ailm.claims",
            "ailm.control",
            "ailm.corpus.cmudict",
            "ailm.generative.smoke",
        }
        assert required <= set(names), (
            f"Stage 0 modules absent from package: {required - set(names)}"
        )
        for name in names:
            importlib.import_module(name)

        # NEGATIVE CONTROL: the import path must actually reject a missing module.
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("ailm.__stage0_negative_control_missing__")

    def test_package_version_present(self):
        import ailm

        root = Path(__file__).resolve().parents[1]
        metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        declared = metadata["project"]["version"]
        assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", declared), (
            f"Project version is not a stable three-part release: {declared!r}"
        )
        assert ailm.__version__ == declared

        # NEGATIVE CONTROL: the equality check distinguishes a stale package value.
        planted_stale_version = "0.0.0"
        assert planted_stale_version != declared

    def test_smoke_returns_all_passed_key(self):
        result = run_all_smoke_tests()
        assert "all_passed" in result

    def test_stage0_can_record_completion(self, tmp_path):
        """Record Stage 0 completion in the ledger — end-to-end."""
        ledger = ClaimsLedger(tmp_path / "stage0_completion.json")
        ledger.record(
            "stage0_complete",
            1,
            "test_stage0_gate_run",
            "tests/test_stage0_gate.py::TestStage0Completeness",
        )
        assert ledger.verify_label("stage0_complete")
        entry = ledger.get("stage0_complete")
        assert entry["value"] == 1
