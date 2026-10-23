"""Stage 3 gate — the word level, and what admits it.

The gate
--------
Stage 3 is admitted by beating a **letter-bigram model over the identical
alphabet, trained on the identical tokens, scored on the identical positions**.
Metric: per-emission perplexity, where an emission is a letter or the
end-of-word symbol, over held-out in-vocabulary Brown tokens.

The declared "perplexity < 500 on PTB-1k" was VACUOUS — the bar itself clears it
by claim:stage3_declared_gate_slack times — and named a corpus this project does
not have. ``docs/decisions.md`` entry 14 records the restatement;
``test_the_declared_gate_was_vacuous`` re-derives that verdict rather than
trusting the prose.

Why the bar is not the whole guard
----------------------------------
Level 2 taught that beating the baseline is necessary, not sufficient. At Level 3
that lesson is sharper. The lexicon is so much information that **three of the
four prior mis-wirings still beat the bar**, including one that inverts the
frequency ranking so the rarest words are predicted most confidently. That is
asserted here as a fact about the control set, not left as a remark:
``test_the_bar_does_not_catch_the_prior_miswirings``.

What pins the level is the **path identity** — the probability the model assigns
to a word's whole emission path must equal that word's unigram probability —
checked between a live pymdp agent and an independent closed form that imports
neither pymdp nor jax. Every surviving mis-wiring changes the word prior, so
every one of them fails it.

Claims discipline
-----------------
This file **verifies**; ``scripts/record_stage3_claims.py`` writes. See
``docs/decisions.md`` entry 10 and ``tests/test_claims_single_writer.py``.
"""

from __future__ import annotations

import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from ailm.claims import load_ledger
from ailm.control import ControlHarness
from ailm.corpus.brown import (
    BROWN_WORD_ARTIFACT_SCHEMA,
    brown_document_identity_sha256,
    brown_partition_input_sha256,
    brown_source_metadata,
)
from ailm.gnn import assert_conforms, check_conformance, describe_shapes, load_spec
from ailm.levels.word import (
    END_INDEX,
    N_OBS,
    build_emission,
    build_letter_bigram_bar,
    build_position_transition,
    build_word_agent,
    build_word_prior,
    build_word_transition,
    load_word_artifact,
    posterior_word_support,
    word_perplexity,
)
from ailm.verification.oracle import (
    letter_bigram_perplexity,
    word_path_logprob,
    word_path_perplexity,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = PROJECT_ROOT / "vendor" / "brown_word_counts.json"

_SKIP_NO_ARTIFACT = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason="vendor/brown_word_counts.json absent; run scripts/generate_word_counts.py",
)

pytestmark = _SKIP_NO_ARTIFACT


@pytest.fixture(scope="module")
def artifact():
    return load_word_artifact()


@pytest.fixture(scope="module")
def prior(artifact):
    return build_word_prior(artifact)


@pytest.fixture(scope="module")
def agent(artifact):
    return build_word_agent(artifact)


@pytest.fixture(scope="module")
def scores(artifact) -> dict[str, float]:
    """Model and bar over the identical position set."""
    model_ppl, n_model = word_perplexity(artifact)
    transition, start = build_letter_bigram_bar(artifact)
    bar_ppl, n_bar = letter_bigram_perplexity(artifact.eval_tokens, transition, start)
    return {
        "model": model_ppl,
        "bar": bar_ppl,
        "n_model": n_model,
        "n_bar": n_bar,
    }


# ══ THE GATE ══════════════════════════════════════════════════════════════════


def test_stage3_beats_the_letter_bigram_bar(scores):
    """(1) The level must beat the strongest model that does not know the lexicon."""
    assert scores["model"] < scores["bar"], (
        f"Word model {scores['model']:.4f} does not beat the letter-bigram bar "
        f"{scores['bar']:.4f}. Knowing the lexicon must buy something."
    )


def test_the_bar_is_scored_on_identical_positions(scores):
    """A comparator on a different position set is not a comparator (decision 2)."""
    assert scores["n_model"] == scores["n_bar"], (
        f"Model scored {scores['n_model']} emissions, bar {scores['n_bar']}. The "
        f"two numbers describe different quantities and must not be compared."
    )


def test_the_declared_gate_was_vacuous(scores):
    """POSITIVE CONTROL on the restatement: re-derive the old gate's verdict.

    ``docs/decisions.md`` entry 14 says the declared "perplexity < 500" is
    vacuous. Prose asserting a verdict is not the verdict; this measures it.
    """
    declared = 500.0
    assert scores["bar"] < declared, (
        f"The bar {scores['bar']:.4f} does NOT clear the declared threshold "
        f"{declared}, so the vacuity finding recorded in decisions.md entry 14 "
        f"no longer holds and the entry needs revisiting."
    )


def test_the_gate_is_sound_not_unreachable(artifact, scores):
    """The ceiling must clear the bar, or no correct model could pass (decision 3)."""
    held_freq = Counter(artifact.eval_tokens)
    n_tokens = len(artifact.eval_tokens)
    n_emissions = sum(len(w) + 1 for w in artifact.eval_tokens)
    ceiling = math.exp(
        -sum(math.log(held_freq[w] / n_tokens) for w in artifact.eval_tokens) / n_emissions
    )
    assert ceiling < scores["bar"], (
        f"The achievable ceiling {ceiling:.4f} does not beat the bar "
        f"{scores['bar']:.4f}: the gate would be UNREACHABLE."
    )
    assert scores["model"] >= ceiling, (
        f"The model {scores['model']:.4f} beats the held-out oracle ceiling "
        f"{ceiling:.4f}, which is impossible for a closed-vocabulary word model. "
        f"Something is scoring a different position set."
    )


# ══ THE IDENTITY THAT PINS THE LEVEL ══════════════════════════════════════════


@pytest.mark.timeout(300)
def test_path_probability_equals_the_word_unigram(artifact, agent, prior):
    """(2) A live pymdp walk must reproduce the closed-form path probability.

    Every factor except ``D[0]`` is deterministic, so the probability of a whole
    emission path collapses to the word's prior mass. The closed form is derived
    in ``ailm.verification.oracle``, which imports neither pymdp nor jax — an
    auditor sharing code with the thing it audits inherits the flaw it audits for.

    This is what licenses computing the full-set statistics in closed form: it
    proves the closed form describes the *agent*, not merely itself.
    """
    emission = np.asarray(agent.A[0])[0]
    position_start = np.zeros(artifact.n_positions, dtype=np.float32)
    position_start[0] = 1.0

    # Warm-up / pre-compile JAX for the infer_states path with the expected shape.
    # The test samples len(vocab) // 120 words (~16), each iterating letter-by-letter
    # through a pymdp infer_states call. On CPU this pays XLA compilation per distinct
    # belief shape — set a persistent cache dir so subsequent runs reuse compiled HLO.
    import os

    os.environ.setdefault(
        "JAX_PERSISTENT_CACHE_DIR", os.path.expanduser("~/.cache/jax_compilation")
    )
    warmup_belief = [
        jnp.array(np.asarray(agent.D[0]).reshape(1, -1)),
        jnp.array(position_start[np.newaxis]),
    ]
    _ = agent.infer_states([jnp.array([[0]])], warmup_belief)

    stride = max(1, len(artifact.vocab) // 120)
    worst = 0.0
    for word in artifact.vocab[::stride]:
        belief = [
            jnp.array(np.asarray(agent.D[0]).reshape(1, -1)),
            jnp.array(position_start[np.newaxis]),
        ]
        live = 0.0
        for obs in [ord(ch) - 97 for ch in word] + [END_INDEX]:
            q_word = np.asarray(belief[0]).ravel()
            q_pos = np.asarray(belief[1]).ravel()
            predictive = float(np.einsum("wp,w,p->", emission[obs], q_word, q_pos))
            live += math.log(max(predictive, 1e-300))
            qs = agent.infer_states([jnp.array([obs])], belief)
            belief = agent.update_empirical_prior(jnp.array([[0, 0]]), qs)
        closed = word_path_logprob(word, artifact.word_to_index[word], prior)
        worst = max(worst, abs(live - closed))

    assert worst < 1e-4, (
        f"pymdp and the closed form disagree by {worst:.3e} nats. The recorded "
        f"Stage 3 statistics are computed in closed form and would no longer "
        f"describe the agent."
    )


def test_the_path_identity_is_not_vacuous(artifact, prior):
    """NON-VACUITY: the identity must FAIL on a perturbed prior.

    An identity that holds for every prior would pin nothing. This plants a
    change in one word's mass and requires the check to notice.
    """
    perturbed = prior.copy()
    perturbed[0] *= 0.5
    perturbed /= perturbed.sum()
    word = artifact.vocab[0]
    correct = word_path_logprob(word, 0, prior)
    planted = word_path_logprob(word, 0, perturbed)
    assert abs(correct - planted) > 1e-3, (
        "Halving a word's prior mass did not change its path log-probability, so "
        "the identity does not constrain the prior at all."
    )


def test_posterior_support_is_the_trie_node(artifact, agent):
    """(3) The factorization must INDUCE the trie the design document specifies.

    Decision 13 claims the two encodings are the same model in different
    coordinates. That is checkable: after observing a prefix, the words with
    posterior mass must be exactly the vocabulary words carrying that prefix.
    """
    for prefix in ("th", "the", "co", "s"):
        support = set(posterior_word_support(agent, artifact, prefix))
        expected = {w for w in artifact.vocab if w.startswith(prefix)}
        assert support == expected, (
            f"After {prefix!r} the posterior is supported on {len(support)} words "
            f"but the trie node holds {len(expected)}. Missing: "
            f"{sorted(expected - support)[:5]}; spurious: "
            f"{sorted(support - expected)[:5]}."
        )


def test_the_trie_correspondence_is_not_vacuous(artifact, agent):
    """NON-VACUITY: a prefix must actually narrow the support.

    If every prefix returned the whole vocabulary the equality above would still
    hold for the empty-prefix case and prove nothing about inference.
    """
    broad = posterior_word_support(agent, artifact, "t")
    narrow = posterior_word_support(agent, artifact, "the")
    assert len(narrow) < len(broad) < len(artifact.vocab), (
        f"Support did not narrow with a longer prefix: 't'->{len(broad)}, "
        f"'the'->{len(narrow)}, vocabulary {len(artifact.vocab)}."
    )


# ══ STRUCTURAL INVARIANTS, EACH ON THE RIGHT AXIS ════════════════════════════


def test_emission_is_normalized_over_observations_not_over_words(artifact):
    """A matrix summing to 1 along *some* axis is not a validated matrix.

    The emission must be a distribution over observations for every
    ``(word, position)`` pair. Asserting the sum on the wrong axis is a green
    check on a transposed model — the Stage 2 lesson, one level up.
    """
    emission = build_emission(artifact)
    assert emission.shape == (N_OBS, len(artifact.vocab), artifact.n_positions)
    np.testing.assert_allclose(
        emission.sum(axis=0),
        np.ones((len(artifact.vocab), artifact.n_positions)),
        atol=1e-9,
        err_msg="A[:, w, p] must be a distribution over observations",
    )


def test_emission_spells_each_word_and_then_ends(artifact):
    """The emission IS the lexicon's spelling; check it against the words."""
    emission = build_emission(artifact)
    for word in (artifact.vocab[0], artifact.vocab[len(artifact.vocab) // 2]):
        index = artifact.word_to_index[word]
        for position in range(artifact.n_positions):
            expected = ord(word[position]) - 97 if position < len(word) else END_INDEX
            assert emission[:, index, position].argmax() == expected, (
                f"{word!r} at position {position} emits "
                f"{emission[:, index, position].argmax()}, expected {expected}"
            )


def test_b_word_is_exactly_the_identity(artifact):
    """A word must not become a different word part-way through being spelled."""
    transition = build_word_transition(artifact)
    np.testing.assert_allclose(
        transition,
        np.eye(len(artifact.vocab)),
        atol=1e-12,
        err_msg="B[0] must be the identity over word identity",
    )


def test_b_position_advances_and_is_not_time_reversed(artifact):
    """The position transition is the +1 shift; its transpose spells backwards."""
    transition = build_position_transition(artifact)
    n_pos = artifact.n_positions
    for position in range(n_pos - 1):
        assert transition[position + 1, position] == 1.0, (
            f"position {position} must advance to {position + 1}"
        )
    assert transition[n_pos - 1, n_pos - 1] == 1.0, "the last position must absorb"
    assert not np.allclose(transition, transition.T), (
        "The position transition equals its own transpose, so the time-reversal "
        "negative control cannot distinguish anything."
    )
    np.testing.assert_allclose(transition.sum(axis=0), np.ones(n_pos), atol=1e-12)


def test_word_prior_is_the_only_learned_probability(artifact, prior):
    """The level's whole content is the lexicon's frequencies."""
    assert prior.shape == (len(artifact.vocab),)
    assert prior.sum() == pytest.approx(1.0)
    assert not np.allclose(prior, 1.0 / len(artifact.vocab), atol=1e-4), (
        "The word prior is indistinguishable from uniform — the vendored "
        "frequencies were dropped during parsing."
    )
    assert np.all(np.diff(prior) <= 1e-12), (
        "The vocabulary must stay in descending frequency order; the recorder's "
        "stride sampling assumes it."
    )


# ══ NEGATIVE CONTROLS — one per mis-wiring ════════════════════════════════════


def _miswired_priors(artifact, prior) -> dict[str, np.ndarray]:
    groups = Counter(len(w) for w in artifact.vocab)
    length = np.array([1.0 / (groups[len(w)] * len(groups)) for w in artifact.vocab])
    return {
        "uniform": np.full(len(artifact.vocab), 1.0 / len(artifact.vocab)),
        "length": length / length.sum(),
        "reversed": prior[::-1].copy(),
    }


@pytest.mark.parametrize("name", ["uniform", "length", "reversed"])
def test_prior_miswirings_fail_the_path_identity(artifact, prior, name):
    """Each mis-wired prior must FAIL the identity that pins the level.

    These are the controls that matter: all three beat the accuracy bar, so the
    identity is the only guard standing between them and a green gate.
    """
    wrong = _miswired_priors(artifact, prior)[name]
    word = artifact.vocab[0]
    correct = word_path_logprob(word, 0, prior)
    under_miswiring = word_path_logprob(word, 0, wrong)
    assert abs(correct - under_miswiring) > 1e-3, (
        f"The {name} mis-wiring produces the same path probability as the "
        f"shipped prior, so the identity does not catch it."
    )


def test_the_bar_does_not_catch_the_prior_miswirings(artifact, prior, scores):
    """A CONTROL ON THE CONTROL SET, and the sharpest finding at this level.

    Three of the four mis-wirings beat the bar. Recording that as an executable
    assertion rather than a remark is what stops a future reader concluding that
    ``assert model < bar`` is a sufficient gate. If this test ever fails because
    a mis-wiring stopped clearing the bar, the *finding* changed and
    ``docs/decisions.md`` entry 14 needs updating — it is not a license to drop
    the identity check.
    """
    survived = {
        name: word_perplexity(artifact, word_prior=wrong)[0]
        for name, wrong in _miswired_priors(artifact, prior).items()
    }
    clearing = {n: v for n, v in survived.items() if v < scores["bar"]}
    assert set(clearing) == {"uniform", "length", "reversed"}, (
        f"Expected all three prior mis-wirings to clear the bar "
        f"{scores['bar']:.4f}; got {survived}. The claim in decisions.md entry 14 "
        f"that the bar is necessary-but-not-sufficient rests on this."
    )
    for name, value in clearing.items():
        assert value > scores["model"], (
            f"The {name} mis-wiring ({value:.4f}) scores better than the shipped "
            f"model ({scores['model']:.4f}), which would mean the shipped prior "
            f"is not the maximum-likelihood one."
        )


def test_time_reversed_emission_is_caught_by_the_bar(artifact):
    """The one mis-wiring the bar DOES catch, kept so the contrast is measured."""
    total = sum(artifact.word_freq.values())
    reversed_paths = {w[::-1]: c / total for w, c in artifact.word_freq.items()}
    n_emissions = sum(len(w) + 1 for w in artifact.eval_tokens)
    logp = sum(math.log(max(reversed_paths.get(w, 0.0), 1e-12)) for w in artifact.eval_tokens)
    reversed_ppl = math.exp(-logp / n_emissions)
    transition, start = build_letter_bigram_bar(artifact)
    bar, _ = letter_bigram_perplexity(artifact.eval_tokens, transition, start)
    assert reversed_ppl > bar, (
        f"The time-reversed emission scores {reversed_ppl:.4f}, which beats the "
        f"bar {bar:.4f}. It was the one mis-wiring a baseline comparison caught."
    )


def test_miswiring_registry_rejects_duplicate_and_extra_analytic_arms():
    """A set conversion must not hide a duplicate that changes prose counts."""
    from ailm.levels.word import STAGE3_MISWIRINGS, validate_stage3_miswirings

    with pytest.raises(ValueError, match="duplicate keys"):
        validate_stage3_miswirings((*STAGE3_MISWIRINGS, STAGE3_MISWIRINGS[0]))
    with pytest.raises(ValueError, match="exactly the analytic-infinite"):
        validate_stage3_miswirings(
            (*STAGE3_MISWIRINGS, type(STAGE3_MISWIRINGS[0])("other_infinite", "other", None))
        )


def test_out_of_vocabulary_raises_rather_than_smoothing(artifact, prior):
    """The closed-vocabulary bound is the finding, not an inconvenience.

    Returning a smoothed number for a token the model assigns no probability to
    would report a result for an undefined input.
    """
    with pytest.raises(ValueError, match="out of vocabulary"):
        word_path_perplexity(["zzzznotaword"], artifact.word_to_index, prior)


def test_the_gate_is_live_under_the_control_harness(artifact, scores):
    """The gate must DISCRIMINATE: a mis-wired model has to land on the wrong side.

    ``assert_gate_is_live`` raises ``GateVacuousError`` when the good and the
    broken value sit on the same side of the bar, so this is the gate's own
    liveness proof rather than a second pass of the positive assertion.
    """
    harness = ControlHarness("stage3_word")
    uniform = np.full(len(artifact.vocab), 1.0 / len(artifact.vocab))
    broken, _ = word_perplexity(artifact, word_prior=uniform)
    harness.assert_gate_is_live(
        good_value=scores["model"],
        bad_value=broken,
        direction="lower_is_better",
    )


# ══ SCOPE, STATED RATHER THAN HIDDEN ══════════════════════════════════════════


def test_coverage_is_reported_and_is_not_total(artifact):
    """Every Stage 3 number describes a sublanguage; the share is the honest number."""
    assert 0.0 < artifact.coverage_pct < 100.0, (
        f"Coverage {artifact.coverage_pct} is not a proper fraction — either the "
        f"vocabulary bound vanished or the held-out stream is mis-counted."
    )
    assert len(artifact.eval_tokens) < artifact.provenance["held_tokens"], (
        "Every held-out token is in vocabulary, which cannot be true for a "
        "bounded lexicon and means the restriction is not being applied."
    )


def test_stage3_perplexity_is_not_comparable_with_stage1(artifact):
    """The two alphabets differ, so the two numbers must never share a table."""
    assert N_OBS == 27
    from ailm.levels.letter import N_SYMBOLS

    assert N_SYMBOLS == 30 and N_SYMBOLS != N_OBS, (
        "Level 1 and Level 3 now share an alphabet size; if that is deliberate "
        "the 'not comparable' warnings in the docs and the oracle need revisiting."
    )


# ══ GNN CONFORMANCE ═══════════════════════════════════════════════════════════


def test_live_agent_conforms_to_its_spec(agent):
    """The shipped two-factor agent must satisfy gnn/models/L3_word.gnn.md."""
    assert_conforms(load_spec("L3_word"), describe_shapes(agent))


#: The intended semantic reason each on-disk mutation must be rejected for,
#: pinned from the violation text the conformance checker emits against the
#: live agent. A non-empty violation list alone is satisfied by ANY defect
#: the mutation drags along; the fragment binds the rejection to the property
#: the mutation targets. The conformance suite pins its own controls to the
#: same checker, so the two readers cannot drift apart silently.
L3_MUTATION_EXPECTATIONS = {
    "frozen_position": (
        "B_f1: declared Identity(21) in InitialParameterization but the live "
        "matrix is NOT the identity"
    ),
    "drifting_word": "the live matrix IS the identity, so it carries no learned content",
    "transposed_factors": (
        "declared dims (27, 21, 2000) but the live canonical dims are (27, 2000, 21)"
    ),
    "unsized_factor": "Every hidden factor needs exactly one belief of its own size",
}


@pytest.mark.parametrize("mutation", sorted(L3_MUTATION_EXPECTATIONS))
def test_spec_mutations_are_rejected_for_their_intended_reason(agent, mutation):
    """NEGATIVE CONTROLS on conformance itself, pinned to the intended reason.

    Each mutation under ``gnn/mutations/`` is a spec that must NOT describe
    the shipped agent — and must be rejected FOR the property it targets.
    """
    violations = check_conformance(load_spec(f"L3_word@{mutation}"), describe_shapes(agent))
    joined = "\n".join(violations)
    assert violations, (
        f"L3_word@{mutation} passed conformance against the correct live agent, "
        f"so that check is vacuous for the property the mutation targets."
    )
    assert L3_MUTATION_EXPECTATIONS[mutation] in joined, (
        f"L3_word@{mutation} was rejected, but not for its intended reason, so "
        f"the rejection no longer measures the defect the mutation carries: {joined}"
    )


_L3_DISTINCT_PAIRS = [
    (intended, other)
    for intended in sorted(L3_MUTATION_EXPECTATIONS)
    for other in sorted(L3_MUTATION_EXPECTATIONS)
    if intended != other
]


@pytest.mark.parametrize("intended,other", _L3_DISTINCT_PAIRS)
def test_a_different_l3_mutation_cannot_satisfy_the_intended_expectation(agent, intended, other):
    """FALSIFIER over the distinct shipped mutations, not every possible fault.

    For every ordered pair of DIFFERENT on-disk mutations, the OTHER
    mutation's violations must not contain THIS mutation's expected reason;
    only the pairs that exist are collected, so the run carries no skips. If
    a fragment matched a different defect too, the pin above would measure
    nothing. The identical pair is pinned directly by the rejection test.
    """
    violations = check_conformance(load_spec(f"L3_word@{other}"), describe_shapes(agent))
    joined = "\n".join(violations)
    assert violations, "the unrelated mutation must still be rejected, or the pair is vacuous"
    assert L3_MUTATION_EXPECTATIONS[intended] not in joined, (
        f"L3_word@{other}'s violations satisfy L3_word@{intended}'s expected "
        f"reason, so the fragment does not pin the intended semantic defect: {joined}"
    )


# ══ CLAIMS LEDGER — verify, never write ══════════════════════════════════════


def test_stage3_claims_verified(artifact, scores):
    """Recorded Stage 3 values must match a fresh measurement of this run."""
    ledger = load_ledger(PROJECT_ROOT)
    expected = {
        "stage3_word_perplexity": round(scores["model"], 4),
        "stage3_bigram_baseline": round(scores["bar"], 4),
        "stage3_eval_positions": scores["n_model"],
        "stage3_eval_tokens": len(artifact.eval_tokens),
        "stage3_vocab_size": len(artifact.vocab),
        "stage3_coverage_pct": round(artifact.coverage_pct, 2),
    }
    for label, value in expected.items():
        entry = ledger.get(label)
        assert entry is not None, (
            f"{label} is absent. Run scripts/record_stage3_claims.py — this test may not write it."
        )
        assert entry["source"].startswith("scripts/"), (
            f"{label} was last written by {entry['source']!r}. Recorders write; tests verify."
        )
        assert entry["value"] == pytest.approx(value, rel=1e-4), (
            f"Ledger {label}={entry['value']!r} != fresh measurement {value!r}."
        )


# ══ ARTIFACT / BUILDER VALIDATION PATHS (error branches) ════════════════════════


def _minimal_word_payload() -> dict:
    """A valid-against-__init__ payload used to exercise validation branches."""
    document_ids = ["train-doc", "heldout-doc"]
    return {
        "schema": BROWN_WORD_ARTIFACT_SCHEMA,
        "provenance": {
            "corpus": "NLTK Brown",
            "raw_source": brown_source_metadata(),
            "normalization": "lowercased; tokens of ASCII letters only",
            "target_train_document_fraction": 0.9,
            "train_document_fraction": 0.5,
            "train_token_fraction": 0.9,
            "split": (
                "category-stratified revision-bound SHA-256 document split; "
                "not by word type (closed vocabulary)"
            ),
            "vocab_size": 2,
            "vocab_bound": 2,
            "lexicon_source": "brown-frequency-ranked",
            "max_word_len": 20,
            "types_dropped_for_length": 0,
            "total_alphabetic_tokens": 100,
            "train_tokens": 90,
            "held_tokens": 10,
            "train_types": 2,
            "documents": 2,
            "train_documents": 1,
            "heldout_documents": 1,
            "categories": 1,
            "document_identity_sha256": brown_document_identity_sha256(document_ids),
            "document_partition_input_sha256": brown_partition_input_sha256(
                document_ids, ["fiction", "fiction"]
            ),
        },
        "documents": {
            "ids": document_ids,
            "categories": ["fiction", "fiction"],
            "alphabetic_token_counts": [90, 10],
            "split_indices": {"train": [0], "heldout": [1]},
            "eval_token_document_indices": [1, 1],
        },
        "word_freq": {"cat": 60, "sat": 30},
        "bar_train_freq": {"cat": 60, "sat": 30},
        "eval_tokens": ["cat", "sat"],
    }


def _load_word_generator():
    path = PROJECT_ROOT / "scripts" / "generate_word_counts.py"
    spec = importlib.util.spec_from_file_location("_generate_word_counts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_word_artifact_rejects_an_unknown_schema():
    """The artifact must refuse a different-generation schema loudly."""
    from ailm.levels.word import WordArtifact

    payload = _minimal_word_payload()
    payload["schema"] = "ailm-word-counts-v0"
    with pytest.raises(ValueError, match="Unexpected artifact schema"):
        WordArtifact(payload)


def test_word_artifact_rejects_an_empty_vocabulary():
    from ailm.levels.word import WordArtifact

    payload = _minimal_word_payload()
    payload["word_freq"] = {}
    payload["bar_train_freq"] = {}
    with pytest.raises(ValueError, match="empty vocabulary"):
        WordArtifact(payload)


def test_word_artifact_rejects_an_empty_eval_stream():
    from ailm.levels.word import WordArtifact

    payload = _minimal_word_payload()
    payload["eval_tokens"] = []
    with pytest.raises(ValueError, match="empty evaluation"):
        WordArtifact(payload)


def test_word_artifact_rejects_an_out_of_vocabulary_eval_token():
    from ailm.levels.word import WordArtifact

    payload = _minimal_word_payload()
    payload["eval_tokens"] = ["dog"]  # not in word_freq
    with pytest.raises(ValueError, match="out-of-vocabulary"):
        WordArtifact(payload)


@pytest.mark.parametrize("token", [["cat"], 3, None, True])
def test_word_artifact_rejects_non_word_eval_entries_with_a_controlled_error(token: object):
    from ailm.levels.word import WordArtifact

    payload = _minimal_word_payload()
    payload["eval_tokens"] = [token]
    with pytest.raises(ValueError, match=r"eval_tokens\[0\]"):
        WordArtifact(payload)


def test_word_artifact_rejects_a_missing_vocab_key_for_eval_token():
    from ailm.levels.word import WordArtifact

    payload = _minimal_word_payload()
    payload["eval_tokens"] = ["CAT"]  # case differs from word_freq key 'cat'
    with pytest.raises(ValueError, match="out-of-vocabulary"):
        WordArtifact(payload)


@pytest.mark.parametrize("count", [0, -1, 1.5, True, "3"])
def test_word_artifact_rejects_non_positive_or_non_integer_counts(count: object):
    from ailm.levels.word import WordArtifact

    payload = _minimal_word_payload()
    payload["word_freq"]["cat"] = count
    with pytest.raises(ValueError, match="positive integer count"):
        WordArtifact(payload)


@pytest.mark.parametrize("word", ["Cat", "cat2", "", "café"])
def test_word_artifact_rejects_words_outside_the_declared_alphabet(word: str):
    from ailm.levels.word import WordArtifact

    payload = _minimal_word_payload()
    payload["word_freq"] = {word: 60, "sat": 30}
    with pytest.raises(ValueError, match="lowercase ASCII-letter"):
        WordArtifact(payload)


def test_word_artifact_rejects_frequency_order_and_cross_table_drift():
    from ailm.levels.word import WordArtifact

    unordered = _minimal_word_payload()
    unordered["word_freq"] = {"sat": 30, "cat": 60}
    with pytest.raises(ValueError, match="descending count"):
        WordArtifact(unordered)

    mismatch = _minimal_word_payload()
    mismatch["bar_train_freq"]["cat"] = 59
    with pytest.raises(ValueError, match="does not match bar_train_freq"):
        WordArtifact(mismatch)


def test_word_artifact_rejects_schema_and_provenance_cardinality_drift():
    from ailm.levels.word import WordArtifact

    missing = _minimal_word_payload()
    del missing["bar_train_freq"]
    with pytest.raises(ValueError, match="missing=.*bar_train_freq"):
        WordArtifact(missing)

    extra = _minimal_word_payload()
    extra["undeclared"] = 1
    with pytest.raises(ValueError, match="unexpected=.*undeclared"):
        WordArtifact(extra)

    extra_provenance = _minimal_word_payload()
    extra_provenance["provenance"]["undeclared"] = 1
    with pytest.raises(ValueError, match="provenance fields.*unexpected=.*undeclared"):
        WordArtifact(extra_provenance)

    wrong_vocab = _minimal_word_payload()
    wrong_vocab["provenance"]["vocab_size"] = 3
    with pytest.raises(ValueError, match="vocab_size=.*cardinality"):
        WordArtifact(wrong_vocab)

    wrong_types = _minimal_word_payload()
    wrong_types["provenance"]["train_types"] = 3
    with pytest.raises(ValueError, match="train_types=.*cardinality"):
        WordArtifact(wrong_types)


def test_word_artifact_rejects_false_or_internally_inconsistent_provenance():
    from ailm.levels.word import WordArtifact

    wrong_corpus = _minimal_word_payload()
    wrong_corpus["provenance"]["corpus"] = "unknown"
    with pytest.raises(ValueError, match="provenance corpus"):
        WordArtifact(wrong_corpus)

    wrong_fraction = _minimal_word_payload()
    wrong_fraction["provenance"]["target_train_document_fraction"] = "0.9"
    with pytest.raises(ValueError, match="target_train_document_fraction"):
        WordArtifact(wrong_fraction)

    wrong_document_fraction = _minimal_word_payload()
    wrong_document_fraction["provenance"]["train_document_fraction"] = 0.4
    with pytest.raises(ValueError, match="train_document_fraction disagrees"):
        WordArtifact(wrong_document_fraction)

    wrong_token_fraction = _minimal_word_payload()
    wrong_token_fraction["provenance"]["train_token_fraction"] = 0.8
    with pytest.raises(ValueError, match="train_token_fraction disagrees"):
        WordArtifact(wrong_token_fraction)

    wrong_total = _minimal_word_payload()
    wrong_total["provenance"]["total_alphabetic_tokens"] = 101
    with pytest.raises(ValueError, match="token cardinalities disagree"):
        WordArtifact(wrong_total)

    wrong_drop = _minimal_word_payload()
    wrong_drop["provenance"]["types_dropped_for_length"] = 1
    with pytest.raises(ValueError, match="types_dropped_for_length"):
        WordArtifact(wrong_drop)


def test_word_artifact_rejects_missing_or_moving_brown_source_identity():
    """NEGATIVE CONTROL: valid counts cannot be relabeled from mutable Brown."""
    from ailm.levels.word import WordArtifact

    missing_revision = _minimal_word_payload()
    del missing_revision["provenance"]["raw_source"]["source_revision"]
    with pytest.raises(ValueError, match="lacks pinned raw-source metadata"):
        WordArtifact(missing_revision)

    moving_branch = _minimal_word_payload()
    moving_branch["provenance"]["raw_source"]["source_url"] = (
        "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip"
    )
    with pytest.raises(ValueError, match="disagrees with the pin"):
        WordArtifact(moving_branch)


def test_word_artifact_rejects_duplicate_or_relabeled_document_identity():
    """NEGATIVE CONTROL: counts cannot conceal duplicated or reordered sources."""
    from ailm.levels.word import WordArtifact

    duplicated = _minimal_word_payload()
    duplicated["documents"]["ids"] = ["same", "same"]
    with pytest.raises(ValueError, match="duplicate ids"):
        WordArtifact(duplicated)

    relabeled = _minimal_word_payload()
    relabeled["documents"]["ids"] = ["heldout-doc", "train-doc"]
    with pytest.raises(ValueError, match="identity digest disagrees"):
        WordArtifact(relabeled)

    coherently_recategorized = _minimal_word_payload()
    coherently_recategorized["documents"]["categories"] = ["news", "news"]
    # The hash-selected indices remain valid under this coordinated relabel;
    # only the partition-input digest can detect that its scientific strata moved.
    with pytest.raises(ValueError, match="partition-input digest disagrees"):
        WordArtifact(coherently_recategorized)


def test_canonical_loader_rejects_coordinated_category_relabel(tmp_path):
    """NEGATIVE CONTROL: an artifact cannot authenticate its own rewritten strata."""
    from ailm.levels.word import WordArtifact

    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    categories = payload["documents"]["categories"]
    categories[:] = [
        "coordinated-relabel-0" if category == "science_fiction" else category
        for category in categories
    ]
    payload["provenance"]["document_partition_input_sha256"] = brown_partition_input_sha256(
        payload["documents"]["ids"], categories
    )
    # This coherent relabel preserves the revision-bound split and passes every
    # artifact-internal check. The external source projection is the only guard
    # exercised by the canonical loader below.
    WordArtifact(payload)
    rewritten = tmp_path / "coherently-recategorized.json"
    rewritten.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="pinned source projection"):
        load_word_artifact(rewritten)


def test_canonical_loader_rejects_coordinated_document_id_relabel(tmp_path):
    """NEGATIVE CONTROL: recomputing both embedded digests cannot rename a file."""
    from ailm.levels.word import WordArtifact

    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    document_ids = payload["documents"]["ids"]
    document_ids[0] = "planted_doc_0"
    categories = payload["documents"]["categories"]
    payload["provenance"]["document_identity_sha256"] = brown_document_identity_sha256(document_ids)
    payload["provenance"]["document_partition_input_sha256"] = brown_partition_input_sha256(
        document_ids, categories
    )
    WordArtifact(payload)
    rewritten = tmp_path / "coherently-reidentified.json"
    rewritten.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="pinned source projection"):
        load_word_artifact(rewritten)


def test_canonical_loader_rejects_a_self_consistent_target_protocol_relabel(tmp_path):
    """NEGATIVE CONTROL: nearest-count equivalence cannot redefine the target."""
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    payload["provenance"]["target_train_document_fraction"] = 0.899
    rewritten = tmp_path / "target-relabel.json"
    rewritten.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical protocol"):
        load_word_artifact(rewritten)


def test_canonical_loader_rejects_an_ambiguous_nested_source_declaration(tmp_path):
    """NEGATIVE CONTROL: no second moving source locator can shadow the pin."""
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    payload["provenance"]["raw_source"]["effective_source_url"] = (
        "https://example.invalid/moving.zip"
    )
    rewritten = tmp_path / "ambiguous-raw-source.json"
    rewritten.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected raw-source metadata"):
        load_word_artifact(rewritten)


def test_canonical_loader_rejects_eval_content_not_projected_from_denominator(tmp_path):
    """NEGATIVE CONTROL: valid vocabulary support cannot conceal token drift."""
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    original = payload["eval_tokens"][0]
    replacement = next(word for word in payload["word_freq"] if word != original)
    payload["eval_tokens"][0] = replacement
    rewritten = tmp_path / "mutated-evaluation-content.json"
    rewritten.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not reproduce eval_tokens"):
        load_word_artifact(rewritten)


def test_word_artifact_rejects_cross_arm_document_aliasing_and_non_exhaustive_splits():
    """NEGATIVE CONTROL: no Brown file may occur in both or neither arm."""
    from ailm.levels.word import WordArtifact

    overlap = _minimal_word_payload()
    overlap["documents"]["split_indices"]["heldout"] = [0, 1]
    with pytest.raises(ValueError, match="overlap"):
        WordArtifact(overlap)

    missing = _minimal_word_payload()
    missing["documents"]["split_indices"]["heldout"] = [0]
    with pytest.raises(ValueError, match="overlap|not exhaustive"):
        WordArtifact(missing)


def test_word_artifact_rejects_the_wrong_revision_bound_stratified_selection():
    """NEGATIVE CONTROL: disjointness alone does not implement the declared split."""
    from ailm.levels.word import WordArtifact

    payload = _minimal_word_payload()
    payload["documents"]["split_indices"] = {"train": [1], "heldout": [0]}
    payload["documents"]["eval_token_document_indices"] = [0, 0]
    with pytest.raises(ValueError, match="category-stratified revision-bound"):
        WordArtifact(payload)


def test_word_artifact_rejects_eval_tokens_mapped_to_training_or_overfull_documents():
    """NEGATIVE CONTROL: token provenance must remain aligned with held-out clusters."""
    from ailm.levels.word import WordArtifact

    training_alias = _minimal_word_payload()
    training_alias["documents"]["eval_token_document_indices"] = [0, 1]
    with pytest.raises(ValueError, match="non-heldout document"):
        WordArtifact(training_alias)

    overfull = _minimal_word_payload()
    overfull["documents"]["alphabetic_token_counts"] = [99, 1]
    overfull["provenance"]["train_tokens"] = 99
    overfull["provenance"]["held_tokens"] = 1
    overfull["provenance"]["train_token_fraction"] = 0.99
    with pytest.raises(ValueError, match="maps 2 evaluation tokens"):
        WordArtifact(overfull)


def test_custom_lexicon_counts_are_strict_and_never_silently_default(tmp_path):
    generator = _load_word_generator()
    valid = tmp_path / "valid.tsv"
    valid.write_text(
        "cat\t2\ncat\t3\ndog\naveryveryveryverylongword\t9\n",
        encoding="utf-8",
    )
    assert generator.load_custom_lexicon(valid) == {"cat": 5, "dog": 1}

    invalid_rows = ("cat many\n", "cat 0\n", "cat -1\n", "cat 2 extra\n")
    for index, row in enumerate(invalid_rows):
        invalid = tmp_path / f"invalid-{index}.tsv"
        invalid.write_text(row, encoding="utf-8")
        with pytest.raises(ValueError, match="positive-frequency|positive integer"):
            generator.load_custom_lexicon(invalid)


@pytest.mark.raw_corpus_regeneration
def test_custom_lexicon_artifact_uses_authenticated_brown_evaluation(tmp_path):
    from ailm.levels.word import WordArtifact

    generator = _load_word_generator()
    valid = tmp_path / "valid.tsv"
    valid.write_text(
        "cat\t2\ncat\t3\ndog\naveryveryveryverylongword\t9\n",
        encoding="utf-8",
    )
    artifact = generator.build_artifact(2, valid)
    validated = WordArtifact(artifact)
    assert validated.provenance["lexicon_source"] == "custom:valid.tsv"
    assert "/" not in validated.provenance["lexicon_source"]
    assert validated.provenance["types_dropped_for_length"] == 1
    assert validated.provenance["lexicon_frequency_total"] == 6
    assert len(validated.provenance["lexicon_sha256"]) == 64

    corrupted = artifact | {
        "provenance": {**artifact["provenance"], "lexicon_sha256": "not-a-digest"}
    }
    with pytest.raises(ValueError, match="lexicon_sha256"):
        WordArtifact(corrupted)


def test_word_generator_rejects_negative_or_non_integer_vocabulary_bounds():
    generator = _load_word_generator()
    for value in (-1, True, 1.5, "2000"):
        with pytest.raises(ValueError, match="vocab_size"):
            generator.build_artifact(value)


def test_load_word_artifact_raises_when_absent(tmp_path):
    from ailm.levels.word import load_word_artifact

    with pytest.raises(FileNotFoundError, match="generate_word_counts"):
        load_word_artifact(tmp_path / "nope.json")


def test_build_word_agent_rejects_an_oversized_vocabulary():
    """A vocabulary too large to instantiate must refuse with the arithmetic."""
    from ailm.levels.word import WordArtifact, build_word_agent

    # required = n*n*4 bytes; MAX_AGENT_VOCAB_BYTES = 1.2e9, so any vocab larger
    # than ~17.3k words trips the guard. Build a real oversized payload in memory
    # rather than patching the budget: the guard is about the artifact's size.
    n = 20000

    def alphabetic_word(index: int) -> str:
        digits = ["a"] * 4
        for position in range(3, -1, -1):
            digits[position] = chr(ord("a") + index % 26)
            index //= 26
        return "w" + "".join(digits)

    words = [alphabetic_word(index) for index in range(n)]
    frequencies = dict.fromkeys(words, 1)
    document_ids = ["train-doc", "heldout-doc"]
    eval_tokens = words[::20]
    payload = {
        "schema": BROWN_WORD_ARTIFACT_SCHEMA,
        "provenance": {
            "corpus": "NLTK Brown",
            "raw_source": brown_source_metadata(),
            "normalization": "lowercased; tokens of ASCII letters only",
            "target_train_document_fraction": 0.5,
            "train_document_fraction": 0.5,
            "train_token_fraction": 0.5,
            "split": (
                "category-stratified revision-bound SHA-256 document split; "
                "not by word type (closed vocabulary)"
            ),
            "vocab_size": n,
            "vocab_bound": n,
            "lexicon_source": "brown-frequency-ranked",
            "max_word_len": 20,
            "types_dropped_for_length": 0,
            "total_alphabetic_tokens": n * 2,
            "train_tokens": n,
            "held_tokens": n,
            "train_types": n,
            "documents": 2,
            "train_documents": 1,
            "heldout_documents": 1,
            "categories": 1,
            "document_identity_sha256": brown_document_identity_sha256(document_ids),
            "document_partition_input_sha256": brown_partition_input_sha256(
                document_ids, ["fiction", "fiction"]
            ),
        },
        "documents": {
            "ids": document_ids,
            "categories": ["fiction", "fiction"],
            "alphabetic_token_counts": [n, n],
            "split_indices": {"train": [0], "heldout": [1]},
            "eval_token_document_indices": [1] * len(eval_tokens),
        },
        "word_freq": frequencies,
        "bar_train_freq": frequencies,
        "eval_tokens": eval_tokens,
    }
    big = WordArtifact(payload)
    with pytest.raises(ValueError, match="needs .* GB for the"):
        build_word_agent(big)


def test_build_word_agent_rejects_a_wrong_shaped_word_prior(artifact):
    from ailm.levels.word import build_word_agent

    with pytest.raises(ValueError, match="word_prior must be"):
        build_word_agent(artifact, word_prior=np.ones(len(artifact.vocab) + 1))


def test_build_word_agent_rejects_a_wrong_shaped_emission(artifact):
    from ailm.levels.word import build_word_agent

    n_vocab = len(artifact.vocab)
    with pytest.raises(ValueError, match="emission must be"):
        build_word_agent(
            artifact,
            emission=np.zeros((31, n_vocab, artifact.n_positions)),  # wrong N_OBS=27
        )


def test_posterior_word_support_rejects_an_empty_prefix(artifact, agent):
    from ailm.levels.word import posterior_word_support

    with pytest.raises(ValueError, match="non-empty lowercase"):
        posterior_word_support(agent, artifact, "")


def test_posterior_word_support_rejects_non_letter_prefix(artifact, agent):
    from ailm.levels.word import posterior_word_support

    with pytest.raises(ValueError, match="non-empty lowercase"):
        posterior_word_support(agent, artifact, "C3")
