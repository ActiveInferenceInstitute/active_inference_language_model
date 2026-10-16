#!/usr/bin/env python3
"""Export the shipped AILM matrices for the browser build. Records nothing.

The browser runs the *same model*, not a description of one. AILM is conditional
probability tables plus a forward filter, so the whole of Level 1 and the whole of
Level 3's preference channel fit in a JSON file small enough to inline, and the
inference is a few lines of arithmetic. There is no approximation step here and no
second model: this dumps the arrays the Python build already uses.

That is the point of doing it this way. A demo that reimplements the model from
its description is a second implementation with its own bugs, and the first thing
it would lose is the property that makes this project worth reading -- that every
behavior is attributable to a named matrix. So the export carries the matrices
themselves, and the page carries a self-check that recomputes a recorded claim
from them and refuses to render if the value disagrees.

Writes ``output/web/ailm_model.json``. No claim label, no ledger lock.

Usage:
    .venv/bin/python scripts/export_web_model.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
from _artifact_io import atomic_write_text, sha256_authenticated_regular_file  # noqa: E402

from ailm.benchmarks import HELD_OUT_TEXT  # noqa: E402
from ailm.claims import load_ledger  # noqa: E402
from ailm.generation import (  # noqa: E402
    SPACE_INDEX,
    build_generative_letter_agent,
    efe_policy_posterior,
    lexicon_preference,
)
from ailm.levels.letter import (  # noqa: E402
    ALPHA_TO_IDX,
    IDX_TO_ALPHA,
    N_SYMBOLS,
    build_letter_transition,
    load_brown_counts,
)
from ailm.levels.word import build_word_prior, load_word_artifact  # noqa: E402

OUTPUT = REPO_ROOT / "output" / "web" / "ailm_model.json"
WORD_ARTIFACT = REPO_ROOT / "vendor" / "brown_word_counts.json"

#: Fixed trie locations that exercise a word boundary, the shipped high-gamma
#: greedy path, and two non-greedy branches. The browser self-check evaluates all
#: of them at both the recorded operating precision and the degenerate endpoint.
POLICY_PROBE_PREFIXES = ("", "t", "th", "the", "co", "s")
HIGH_GAMMA = 16.0

#: Decimal places kept per probability. Six is not cosmetic: the page recomputes
#: `stage1_letter_perplexity` from these arrays and compares against the recorded
#: value, so the export has to preserve enough precision for that check to pass on
#: its own terms rather than by a loosened tolerance.
PRECISION = 6


def _round(array: np.ndarray) -> list:
    return np.round(array, PRECISION).tolist()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    transition = np.asarray(build_letter_transition(load_brown_counts()))[0]
    if not np.allclose(transition.sum(axis=0), 1.0, atol=1e-6):
        raise SystemExit("export refused: letter transition is not column-stochastic.")

    artifact = load_word_artifact(WORD_ARTIFACT)
    lexicon = list(artifact.vocab)
    word_prior = np.asarray(build_word_prior(artifact), dtype=np.float64)
    if word_prior.shape != (len(lexicon),):
        raise SystemExit(
            "export refused: Stage 3 word prior is not aligned with the vocabulary "
            f"({word_prior.shape} versus {(len(lexicon),)})."
        )
    if not np.all(np.isfinite(word_prior)) or np.any(word_prior < 0.0):
        raise SystemExit("export refused: Stage 3 word prior is not a finite distribution.")
    if not np.isclose(word_prior.sum(dtype=np.float64), 1.0, atol=1e-12):
        raise SystemExit("export refused: Stage 3 word prior does not sum to one.")

    ledger = load_ledger(REPO_ROOT)

    def claim(label: str) -> float:
        entry = ledger.get(label)
        if entry is None:
            raise SystemExit(f"export refused: ledger has no {label!r}.")
        return float(entry["value"])

    best_gamma = claim("generation_best_gamma")
    policy_probes: list[dict[str, object]] = []
    for gamma in (best_gamma, HIGH_GAMMA):
        agent = build_generative_letter_agent(transition, gamma=gamma)
        for prefix in POLICY_PROBE_PREFIXES:
            current_letter = SPACE_INDEX if not prefix else ALPHA_TO_IDX[prefix[-1]]
            preference = lexicon_preference(prefix, artifact, word_prior)
            posterior = efe_policy_posterior(agent, current_letter, preference, transition)
            policy_probes.append(
                {
                    "prefix": prefix,
                    "current_letter": IDX_TO_ALPHA[current_letter],
                    "gamma": gamma,
                    "posterior": np.asarray(posterior, dtype=np.float64).tolist(),
                }
            )

    lexicon_prior = {"lexicon": lexicon, "word_prior": word_prior.tolist()}
    payload = {
        "schema": "ailm-web-model-v2",
        "provenance": {
            "source": "scripts/export_web_model.py",
            "letter_transition": "ailm.levels.letter.build_letter_transition(load_brown_counts())",
            "word_prior": (
                "ailm.levels.word.build_word_prior("
                "load_word_artifact(vendor/brown_word_counts.json))"
            ),
            "word_artifact": str(WORD_ARTIFACT.relative_to(REPO_ROOT)),
            "word_artifact_sha256": sha256_authenticated_regular_file(
                WORD_ARTIFACT,
                description="Stage 3 word artifact",
            ),
            "lexicon_word_prior_sha256": _canonical_sha256(lexicon_prior),
            "note": (
                "Column-stochastic B[next, current] over the 30-symbol alphabet, "
                "exactly as the shipped pymdp agent receives it. The emission is "
                "the identity, which is why it is not exported: observing a letter "
                "reveals the letter state, so A contributes nothing to store. The "
                "Stage 3 word prior is exported without rounding in the exact order "
                "of lexicon, and its digest binds the two arrays together."
            ),
        },
        "alphabet": [IDX_TO_ALPHA[i] for i in range(N_SYMBOLS)],
        "letter_transition": _round(transition),
        "held_out_text": HELD_OUT_TEXT,
        # The lexicon drives the generation preference. Words plus their aligned
        # prior are enough: the trie is rebuilt in the browser because deriving it
        # is both cheaper than transporting it and a check on the transport.
        "lexicon": lexicon,
        "word_prior": word_prior.tolist(),
        # Python-computed fixed-prefix posteriors. The page independently rebuilds
        # the weighted trie and refuses to render when any vector disagrees.
        "policy_parity_probes": policy_probes,
        # Every number the page displays as a claim, so none is typed into the
        # markup. The self-check compares its own recomputation against the first.
        "claims": {
            "stage1_letter_perplexity": claim("stage1_letter_perplexity"),
            "stage1_unigram_baseline": claim("stage1_unigram_baseline"),
            "stage1_uniform_baseline": claim("stage1_uniform_baseline"),
            "stage1_bigram_floor_perplexity": claim("stage1_bigram_floor_perplexity"),
            "stage3_vocab_size": claim("stage3_vocab_size"),
            "generation_best_gamma": best_gamma,
            "generation_best_tv": claim("generation_best_tv"),
            "generation_best_membership_pct": claim("generation_best_membership_pct"),
            "generation_baseline_membership_pct": claim("generation_baseline_membership_pct"),
            "generation_degenerate_gamma": claim("generation_degenerate_gamma"),
            "generation_shipped_information_gain_nats": claim(
                "generation_shipped_information_gain_nats"
            ),
            "generation_max_information_gain_possible_nats": claim(
                "generation_max_information_gain_possible_nats"
            ),
            "stage4_derived_margin_pts": claim("stage4_derived_margin_pts"),
            "stage4_path_deprel_margin_pts": claim("stage4_path_deprel_margin_pts"),
            "stage4_path_morphology_margin_pts": claim("stage4_path_morphology_margin_pts"),
            "stage4_path_adposition_margin_pts": claim("stage4_path_adposition_margin_pts"),
            "stage4_path_deprel_tokens": claim("stage4_path_deprel_tokens"),
            "stage4_path_morphology_tokens": claim("stage4_path_morphology_tokens"),
            "stage4_path_adposition_tokens": claim("stage4_path_adposition_tokens"),
            # The mechanism: on two paths the observation alphabet is one symbol,
            # so the mutual information is exactly zero and no emission-driven
            # margin was ever possible there (decision 67).
            "stage4_path_deprel_upos_alphabet_size": claim("stage4_path_deprel_upos_alphabet_size"),
            "stage4_path_morphology_upos_alphabet_size": claim(
                "stage4_path_morphology_upos_alphabet_size"
            ),
            "stage4_path_adposition_upos_alphabet_size": claim(
                "stage4_path_adposition_upos_alphabet_size"
            ),
            "stage4_path_deprel_mutual_information_nats": claim(
                "stage4_path_deprel_mutual_information_nats"
            ),
            "stage4_path_morphology_mutual_information_nats": claim(
                "stage4_path_morphology_mutual_information_nats"
            ),
            "stage4_path_adposition_mutual_information_nats": claim(
                "stage4_path_adposition_mutual_information_nats"
            ),
        },
    }

    atomic_write_text(OUTPUT, json.dumps(payload, separators=(",", ":")))
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} ({size_kb:.0f} KB)")
    print(f"  alphabet {N_SYMBOLS} symbols, lexicon {len(artifact.vocab)} words")
    print(f"  {len(payload['claims'])} claims carried; no claim label written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
