"""Contracts for the held-out stream artifact and what it makes measurable.

The artifact's whole value is that it is the *denominator* Stage 3's numbers are
a fraction of. That only holds if it is the same text, under the same split, as
the stream those numbers were scored on. If the two ever drift, the coverage
percentage and the selection ratio computed from them would both be about a
population neither artifact describes, and nothing else in the suite would
notice.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ailm.corpus.brown import resolve_brown_source_zip
from ailm.levels.word import load_word_artifact, validate_heldout_stream_artifact

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STREAM = PROJECT_ROOT / "vendor" / "brown_heldout_stream.json"


def _module(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_{name}", PROJECT_ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def payload() -> dict:
    if not STREAM.exists():
        pytest.skip(
            "vendor/brown_heldout_stream.json absent; run scripts/generate_heldout_stream.py"
        )
    return json.loads(STREAM.read_text(encoding="utf-8"))


@pytest.mark.raw_corpus_regeneration
def test_the_artifact_reproduces_from_its_generator(payload: dict) -> None:
    """Byte-level reproduction, the same standard every other vendored file meets."""
    try:
        resolve_brown_source_zip()
    except FileNotFoundError:
        pytest.skip("pinned Brown archive absent; raw-source reproduction cannot run offline")
    assert payload == _module("generate_heldout_stream").build_artifact()


def test_filtering_the_stream_reproduces_eval_tokens_element_for_element(payload: dict) -> None:
    """The agreement invariant, and the only reason this artifact can be trusted.

    Element-for-element rather than by count or by multiset: two streams can
    hold the same tokens in a different order and be different text, and order
    is what the perplexity is computed over.
    """
    artifact = load_word_artifact()
    vocabulary = set(artifact.word_freq)
    covered = [token for token in payload["held_tokens"] if token in vocabulary]
    assert covered == artifact.eval_tokens


def test_stream_validator_binds_document_identity_partition_and_token_alignment(
    payload: dict,
) -> None:
    artifact = load_word_artifact()
    assert validate_heldout_stream_artifact(payload, artifact) == payload["held_tokens"]
    assert payload["documents"]["ids"] == artifact.document_ids
    assert payload["documents"]["split_indices"] == {
        "train": artifact.train_document_indices,
        "heldout": artifact.heldout_document_indices,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("duplicate_id", "documents.ids disagrees"),
        ("cross_arm", "split_indices disagrees"),
        ("training_token_alias", "non-heldout document"),
        ("token_count_drift", "counts disagree"),
        ("digest_drift", "document_identity_sha256 disagrees"),
    ),
)
def test_stream_validator_rejects_planted_document_provenance_corruption(
    payload: dict, mutation: str, message: str
) -> None:
    """NEGATIVE CONTROLS: the denominator cannot be relabeled by count alone."""
    corrupted = copy.deepcopy(payload)
    if mutation == "duplicate_id":
        corrupted["documents"]["ids"][1] = corrupted["documents"]["ids"][0]
    elif mutation == "cross_arm":
        corrupted["documents"]["split_indices"]["heldout"][0] = corrupted["documents"][
            "split_indices"
        ]["train"][0]
    elif mutation == "training_token_alias":
        corrupted["documents"]["held_token_document_indices"][0] = corrupted["documents"][
            "split_indices"
        ]["train"][0]
    elif mutation == "token_count_drift":
        mapping = corrupted["documents"]["held_token_document_indices"]
        first, second = corrupted["documents"]["split_indices"]["heldout"][:2]
        mapping[max(index for index, value in enumerate(mapping) if value == first)] = second
    elif mutation == "digest_drift":
        corrupted["provenance"]["document_identity_sha256"] = "0" * 64
    else:  # pragma: no cover - parametrization owns the finite mutation set
        raise AssertionError(mutation)
    with pytest.raises(ValueError, match=message):
        validate_heldout_stream_artifact(corrupted, load_word_artifact())


def test_the_two_artifacts_agree_on_how_many_tokens_were_held_out(payload: dict) -> None:
    """A cheaper check that fails first when the split constant moves."""
    assert payload["provenance"]["held_tokens"] == load_word_artifact().provenance["held_tokens"]
    assert (
        payload["provenance"]["train_document_fraction"]
        == load_word_artifact().provenance["train_document_fraction"]
    )


def test_the_stream_is_unfiltered(payload: dict) -> None:
    """NON-VACUITY: an artifact equal to eval_tokens would make every claim trivial.

    If the stream were accidentally written already filtered, the agreement test
    above would still pass, the selection ratio would be exactly 1, and the
    section would report "no selection effect" on the strength of having
    compared a population with itself.
    """
    artifact = load_word_artifact()
    assert len(payload["held_tokens"]) > len(artifact.eval_tokens)
    vocabulary = set(artifact.word_freq)
    assert any(token not in vocabulary for token in payload["held_tokens"])


@pytest.mark.raw_corpus_regeneration
def test_the_recorder_writes_only_its_own_namespace(tmp_path: Path) -> None:
    """Namespace ownership: this recorder clears stale ``stage3_oov_`` labels.

    A label written under a different Stage 3 prefix would be retired by
    whichever recorder owns that prefix, and the loss would be silent.

    The recorder refuses to write unless the vendored stream reproduces from the
    pinned raw Brown archive, so this control needs that ignored input and runs
    in the full profile's raw-corpus replay step rather than the portable suite.
    """
    try:
        resolve_brown_source_zip()
    except FileNotFoundError:
        pytest.skip("pinned Brown archive absent; the recorder cannot reproduce the stream offline")
    module = _module("record_heldout_stream_claims")
    assert module.SOURCE == "scripts/record_heldout_stream_claims.py"
    ledger_path = tmp_path / "claims.json"
    ledger_path.write_text(json.dumps({"_schema": "ailm-claims-v2", "claims": {}}))
    import os

    os.environ["AILM_CLAIMS_PATH"] = str(ledger_path)
    try:
        assert module.main() == 0
    finally:
        del os.environ["AILM_CLAIMS_PATH"]
    written = json.loads(ledger_path.read_text())["claims"]
    assert written, "the recorder produced no claims"
    assert all(label.startswith("stage3_oov_") for label in written)


def test_the_excluded_remainder_is_where_the_text_gets_harder(payload: dict) -> None:
    """The finding, pinned as a direction rather than as a value.

    Pinning the numbers would duplicate the ledger and break on any corpus
    change; pinning the direction is what the manuscript's qualification
    actually rests on. If the excluded tokens were *easier* than the covered
    ones, the section's disclosure would be backwards.
    """
    from ailm.levels.word import _letter_bigram_perplexity, build_letter_bigram_bar

    artifact = load_word_artifact()
    vocabulary = set(artifact.word_freq)
    excluded = [token for token in payload["held_tokens"] if token not in vocabulary]
    transition, start = build_letter_bigram_bar(artifact)
    covered_pp, _ = _letter_bigram_perplexity(artifact.eval_tokens, transition, start)
    excluded_pp, _ = _letter_bigram_perplexity(excluded, transition, start)
    assert excluded_pp > covered_pp, (
        "the excluded tokens are easier for the lexicon-free comparator than the "
        "covered ones; the manuscript's coverage qualification is stated backwards"
    )


def test_the_coverage_effect_survives_matching_on_length(payload: dict) -> None:
    """The claim the manuscript now rests on, pinned where the old one was not.

    The previous guard asserted ``excluded_pp > covered_pp`` on the raw
    populations. That was true, stayed true throughout, and pinned nothing about
    the *reason* — so the prose could attribute the gap to word length, be wrong,
    and stay green. A guard on a conclusion is not a guard on the reasoning that
    reached it.

    What the paragraph now claims is that the effect is not length: the excluded
    side is harder within length buckets, so it cannot be explained away by the
    two populations having different length distributions.
    """
    from collections import defaultdict

    from ailm.levels.word import _letter_bigram_perplexity, build_letter_bigram_bar

    artifact = load_word_artifact()
    vocabulary = set(artifact.word_freq)
    transition, start = build_letter_bigram_bar(artifact)

    buckets: dict[int, dict[str, list[str]]] = defaultdict(lambda: {"covered": [], "excluded": []})
    for token in payload["held_tokens"]:
        side = "covered" if token in vocabulary else "excluded"
        buckets[len(token)][side].append(token)

    compared = 0
    excluded_harder = 0
    for _length, sides in sorted(buckets.items()):
        if len(sides["covered"]) < 30 or len(sides["excluded"]) < 30:
            continue
        compared += 1
        covered_pp, _ = _letter_bigram_perplexity(sides["covered"], transition, start)
        excluded_pp, _ = _letter_bigram_perplexity(sides["excluded"], transition, start)
        excluded_harder += int(excluded_pp > covered_pp)

    assert compared >= 5, f"only {compared} comparable length buckets; the check is thin"
    assert excluded_harder > compared / 2, (
        f"the excluded tokens are harder in only {excluded_harder} of {compared} length "
        f"buckets. The manuscript claims the coverage effect survives length-matching; "
        f"if it does not, that paragraph is wrong and decision 55 with it."
    )


def test_length_standardization_reconstructs_direct_per_emission_scores() -> None:
    """NEGATIVE CONTROL: token weights must fail on planted unequal lengths.

    The recorder reports per-emission perplexity, so a token of length ``L``
    contributes ``L + 1`` observations. On equal-length buckets, token weights
    and emission weights coincide and the old bug is invisible. These planted
    populations contain two unequal lengths with the same token counts: the
    corrected ``n_L * (L + 1)`` aggregation must reconstruct direct scoring,
    while the former token-weighted aggregation must not.
    """
    from ailm.levels.word import _letter_bigram_perplexity

    module = _module("record_heldout_stream_claims")
    n_obs = 27
    transition = np.full((n_obs, n_obs), 1.0 / n_obs, dtype=np.float64)
    transition[:, 0] = 0.1 / (n_obs - 1)
    transition[0, 0] = 0.9
    start = np.full(n_obs, 0.5 / (n_obs - 1), dtype=np.float64)
    start[0] = 0.5

    covered = ["aa"] * 30 + ["aaaaaa"] * 30
    excluded = ["bb"] * 30 + ["bbbbbb"] * 30
    result = module._length_stratified(covered, excluded, transition, start)

    direct_covered, _ = _letter_bigram_perplexity(covered, transition, start)
    direct_excluded, _ = _letter_bigram_perplexity(excluded, transition, start)
    assert result["stage3_oov_matched_covered_perplexity"] == pytest.approx(
        direct_covered, rel=1e-12
    )
    assert result["stage3_oov_matched_excluded_perplexity"] == pytest.approx(
        direct_excluded, rel=1e-12
    )

    short_pp, _ = _letter_bigram_perplexity(covered[:30], transition, start)
    long_pp, _ = _letter_bigram_perplexity(covered[30:], transition, start)
    former_token_weighted = math.exp((math.log(short_pp) + math.log(long_pp)) / 2.0)
    assert not math.isclose(former_token_weighted, direct_covered, rel_tol=1e-6), (
        "the planted unequal-length control cannot distinguish token weighting "
        "from the required per-emission reconstruction"
    )
