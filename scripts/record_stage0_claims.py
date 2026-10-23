#!/usr/bin/env python3
"""Record Stage 0 quantitative claims into data/claims.json.

Stage 0's prose (README, AGENTS, docs/provenance) references
claim:phoneme_count_39, but the Stage 0 tests only ever recorded that label into
throwaway tmp_path ledgers — the real data/claims.json never received it. The
prose tag was therefore unbacked from the day it was written.

This script derives the value from the actual CMUdict inventory and persists it,
so tests/test_claims_discipline.py can enforce that no prose claim is unbacked.

Run:  .venv/bin/python scripts/record_stage0_claims.py
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ailm.claims import load_ledger  # noqa: E402
from ailm.corpus.cmudict import (  # noqa: E402
    CANONICAL_ARPABET,
    EXPECTED_PHONEME_COUNT,
    validate_cmudict_source_metadata,
)
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402

RUN_ID = "stage0_inventory"
SOURCE = "scripts/record_stage0_claims.py"


def cmudict_claim_values(artifact: Mapping[str, object]) -> dict[str, int]:
    """Validate the committed CMUdict artifact and derive its five ledger claims.

    Raises:
        ValueError: If the artifact cannot support all five claims exactly.
    """
    required_inventory_fields = {
        "phones",
        "source_base_words",
        "source_pronunciation_entries",
        "naive_phoneme_count",
        "comment_phantom_count",
        "comment_phantom_tokens",
    }
    missing_inventory_fields = sorted(required_inventory_fields - set(artifact))
    if missing_inventory_fields:
        raise ValueError(f"artifact lacks required inventory fields: {missing_inventory_fields}")
    validate_cmudict_source_metadata(artifact)

    phones = artifact["phones"]
    phantom_tokens = artifact["comment_phantom_tokens"]
    if not isinstance(phones, list) or not all(isinstance(phone, str) for phone in phones):
        raise ValueError("artifact phones must be a list of strings")
    if not isinstance(phantom_tokens, list) or not all(
        isinstance(token, str) for token in phantom_tokens
    ):
        raise ValueError("artifact comment_phantom_tokens must be a list of strings")
    if len(phones) != len(set(phones)):
        raise ValueError("artifact phones contains duplicate symbols")
    if len(phantom_tokens) != len(set(phantom_tokens)):
        raise ValueError("artifact comment_phantom_tokens contains duplicates")

    integer_fields = (
        "source_base_words",
        "source_pronunciation_entries",
        "naive_phoneme_count",
        "comment_phantom_count",
    )
    for field in integer_fields:
        value = artifact[field]
        if type(value) is not int or value < 0:
            raise ValueError(f"artifact {field} must be a non-negative integer")

    measured_inventory = frozenset(phones)
    measured = len(measured_inventory)
    if measured != EXPECTED_PHONEME_COUNT:
        raise ValueError(
            f"len(artifact phones) = {measured} but "
            f"EXPECTED_PHONEME_COUNT = {EXPECTED_PHONEME_COUNT}"
        )
    if measured_inventory != CANONICAL_ARPABET:
        raise ValueError("artifact phones differ from CANONICAL_ARPABET")

    phantom_count = artifact["comment_phantom_count"]
    if phantom_count != len(phantom_tokens):
        raise ValueError(
            "artifact comment_phantom_count does not equal len(comment_phantom_tokens)"
        )
    if measured_inventory & set(phantom_tokens):
        raise ValueError("artifact comment phantom tokens overlap CANONICAL_ARPABET")

    naive_count = artifact["naive_phoneme_count"]
    if naive_count - phantom_count != measured:
        raise ValueError(
            "artifact naive/canonical inventory difference is not explained "
            "exactly by inline-comment tokens"
        )

    base_words = artifact["source_base_words"]
    pronunciation_entries = artifact["source_pronunciation_entries"]
    if base_words == 0 or pronunciation_entries < base_words:
        raise ValueError(
            "artifact pronunciation-entry extent must cover a non-empty base-word extent"
        )

    return {
        "phoneme_count_39": measured,
        "phoneme_count_naive_unstripped": naive_count,
        "phoneme_count_comment_phantoms": phantom_count,
        "cmudict_base_word_count": base_words,
        "cmudict_pronunciation_entry_count": pronunciation_entries,
    }


def main() -> int:
    # Derived from the shipped inventory, not hand-typed. If CANONICAL_ARPABET
    # ever drifts from EXPECTED_PHONEME_COUNT this fails here rather than
    # silently recording a stale number.
    import json

    artifact_path = _PROJECT_ROOT / "vendor" / "cmudict_counts.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    try:
        recorded = cmudict_claim_values(artifact)
    except ValueError as exc:
        print(
            f"REFUSING TO RECORD: {artifact_path}: {exc}. "
            "Regenerate it with scripts/generate_phoneme_counts.py.",
            file=sys.stderr,
        )
        return 1

    ledger = load_ledger(_PROJECT_ROOT)

    # Derive the role counts too rather than trusting the literals above.
    from ailm.cases import CaseRole, CerebrumFunctionRole

    recorded.update(
        {
            "stage0_case_roles": len(CaseRole),
            "stage0_function_roles": len(CerebrumFunctionRole),
        }
    )

    # Alphabet sizes, derived from the shipped level constants so the manuscript
    # can token-reference them instead of hard-coding 30/27/26 in prose.
    from ailm.levels.letter import ALPHA_LIST, N_SYMBOLS
    from ailm.levels.word import LETTERS, N_OBS

    if len(ALPHA_LIST) != N_SYMBOLS:
        print(
            f"REFUSING TO RECORD: letter.N_SYMBOLS = {N_SYMBOLS} but "
            f"len(ALPHA_LIST) = {len(ALPHA_LIST)}.",
            file=sys.stderr,
        )
        return 1
    if len(LETTERS) + 1 != N_OBS:
        print(
            f"REFUSING TO RECORD: word.N_OBS = {N_OBS} but len(LETTERS) + 1 = "
            f"{len(LETTERS) + 1} (letters plus the end-of-word marker).",
            file=sys.stderr,
        )
        return 1
    recorded["stage0_letter_alphabet_size"] = N_SYMBOLS
    recorded["stage0_word_alphabet_size"] = N_OBS
    recorded["stage0_word_letters"] = len(LETTERS)

    # The coverage floor the suite is gated at, read from pyproject so the
    # reproducibility section cannot silently disagree with the actual gate.
    import tomllib

    pyproject = tomllib.loads((_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    recorded["stage0_coverage_floor_pct"] = int(
        pyproject["tool"]["coverage"]["report"]["fail_under"]
    )

    # ── Numeric environment ──
    # The manuscript's build-determinism caveat rests on pymdp computing in
    # float32 under JAX defaults. Import the same path the models use, THEN read
    # the live flag, so a pymdp or ailm import that flipped it would be seen.
    import importlib.metadata
    import platform

    import jax
    import pymdp  # noqa: F401 - imported for its config side effects, if any

    x64_enabled = bool(jax.config.jax_enable_x64)
    if x64_enabled:
        print(
            "REFUSING TO RECORD: jax_enable_x64 is True. Every float32 tolerance "
            "and analytic-anchor statement in the manuscript assumes JAX's "
            "default float32; a float64 run changes the numeric story, which "
            "needs a rewrite, not a re-render.",
            file=sys.stderr,
        )
        return 1
    recorded["env_jax_enable_x64"] = x64_enabled
    recorded["env_jax_version"] = jax.__version__
    recorded["env_jaxlib_version"] = importlib.metadata.version("jaxlib")
    recorded["env_pymdp_version"] = importlib.metadata.version("inferactively-pymdp")
    recorded["env_platform"] = platform.platform()
    recorded["env_machine"] = platform.machine()

    ledger.record_many(recorded, run_id=RUN_ID, source=SOURCE)

    print(f"Recorded {len(recorded)} Stage 0 claims into data/claims.json:\n")
    width = max(len(k) for k in recorded)
    for label, value in recorded.items():
        print(f"  {label:<{width}} = {value}")
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: ``ClaimsLedger`` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results, with no error on either side.
    with single_writer_lock(ledger_lock_path(_PROJECT_ROOT)):
        raise SystemExit(main())
