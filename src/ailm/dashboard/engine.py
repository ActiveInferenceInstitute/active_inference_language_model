"""Inference service engine for the AILM Multimodal Interactive Dashboard.

Runs the implemented Level 1–6 analyses over an input sentence and returns their
belief trajectories, active matrix slices, and diagnostics. The generated reply
is a separate L3→L1g policy rollout. Level 2, Level 4, Level 5, and Level 6
analyze the input but do not feed that reply, so this module must not describe
the dashboard as an end-to-end six-level generation cascade.
"""

from __future__ import annotations

import re
import zlib
from functools import lru_cache
from typing import Any

import numpy as np

from ailm.corpus.cmudict import load_cmudict, word_to_phones
from ailm.corpus.ud import build_upos_lexicon, load_ud_ewt_dev
from ailm.dashboard.config import DashboardConfig
from ailm.generation import lexicon_preference
from ailm.levels.case import (
    CASES,
    N_CASES,
    build_case_emission,
    build_case_start,
    build_case_transition,
    load_case_artifact,
)
from ailm.levels.clause import (
    CLAUSE_TYPES,
    build_type_models,
    load_clause_artifact,
    sequence_log_evidence,
)
from ailm.levels.clause import (
    N_CASES as N_CLAUSE_CASES,
)
from ailm.levels.letter import (
    ALPHA_LIST,
    ALPHA_TO_IDX,
    N_SYMBOLS,
    build_letter_transition,
    load_brown_counts,
    normalize_text,
)
from ailm.levels.narrative import (
    N_PHASES,
    NARRATIVE_PHASES,
    build_phase_models,
    build_phase_transition,
    clause_prior_with_narrative,
    load_narrative_artifact,
    tercile,
)
from ailm.levels.phoneme import (
    N_PHONES,
    PHONES,
    build_g2p_emission,
    build_g2p_start,
    build_g2p_transition,
    g2p_split,
    load_phoneme_counts,
)
from ailm.levels.widened import (
    AUX_SYMBOL,
    N_WIDENED_SYMBOLS,
    VERB_SYMBOL,
    WIDENED_SYMBOL_LABELS,
    extract_widened_sequences,
    fit_markov_chain,
)
from ailm.levels.word import (
    build_word_prior,
    load_word_artifact,
)

#: Parts of speech that occupy a slot in the widened Level 5 sequence but
#: carry no case role, so they must not advance the Level 4 case chain.
NON_CASE_UPOS = ("VERB", "AUX")

#: Compatibility views of the dataclass defaults.  The values are not declared
#: here: ``DashboardConfig`` owns them, and the server passes its live override
#: into every engine call.
_SHIPPED_LIMITS = DashboardConfig()
MAX_CONTINUATION_WORDS = _SHIPPED_LIMITS.max_continuation_words
MAX_HISTORY_TURNS = _SHIPPED_LIMITS.max_history_turns


@lru_cache(maxsize=1)
def get_dashboard_models() -> dict[str, Any]:
    """Preload and cache all level models and static matrices."""
    # 1. Level 1 (Letters)
    brown_counts = load_brown_counts()
    A_letter = np.asarray(build_letter_transition(brown_counts), dtype=np.float64)[0]
    D_letter = np.ones(N_SYMBOLS, dtype=np.float64) / N_SYMBOLS

    # 2. Level 2 (Phonemes & G2P)
    phone_counts = load_phoneme_counts()
    train_g2p, _ = g2p_split(phone_counts)
    A_g2p = np.asarray(build_g2p_emission(train_g2p), dtype=np.float64)[0]
    B_g2p = np.asarray(build_g2p_transition(train_g2p), dtype=np.float64)[0, :, :, 0]
    D_g2p = np.asarray(build_g2p_start(train_g2p), dtype=np.float64)[0]
    cmu = load_cmudict()

    # 3. Level 3 (Words)
    word_artifact = load_word_artifact()
    word_vocab_set = set(word_artifact.vocab)
    word_unigram = build_word_prior(word_artifact)

    # 4. Level 4 (Case Roles)
    case_artifact = load_case_artifact()
    A_case = np.asarray(build_case_emission(case_artifact), dtype=np.float64)
    B_case = np.asarray(build_case_transition(case_artifact), dtype=np.float64)
    D_case = np.asarray(build_case_start(case_artifact), dtype=np.float64)

    # The part-of-speech lexicon is built by the corpus layer, so the dashboard
    # and anything else asking the same question of UD get the same answer.
    ud_sentences = load_ud_ewt_dev()
    word_to_upos = build_upos_lexicon(ud_sentences)

    # 5. Level 5 (Clause / Sentence Types)
    clause_artifact = load_clause_artifact()
    clause_models = build_type_models(clause_artifact.train)

    # The widened alphabet, its extraction and its fit all live in
    # ailm.levels.widened, so this runs the same code the probe does rather
    # than a second transcription of it. The dashboard fits on ALL pairs: it
    # demonstrates the model rather than measuring held-out performance, and
    # a split here would only make the demonstrator weaker for no gain.
    train_pairs_widened = extract_widened_sequences(ud_sentences)
    dec_seqs = [sq for st, sq in train_pairs_widened if st == "declarative"]
    int_seqs = [sq for st, sq in train_pairs_widened if st == "interrogative"]
    B_dec_widened, D_dec_widened = fit_markov_chain(dec_seqs)
    B_int_widened, D_int_widened = fit_markov_chain(int_seqs)
    B_pool_widened, D_pool_widened = fit_markov_chain([sq for _, sq in train_pairs_widened])

    # 6. Level 6 (Narrative Level)
    narrative_artifact = load_narrative_artifact()
    narrative_models = build_phase_models(narrative_artifact.train)
    narrative_transition = build_phase_transition(narrative_artifact.train)

    return {
        "A_letter": A_letter,
        "D_letter": D_letter,
        "A_g2p": A_g2p,
        "B_g2p": B_g2p,
        "D_g2p": D_g2p,
        "cmu": cmu,
        "word_artifact": word_artifact,
        "word_vocab_set": word_vocab_set,
        "word_unigram": word_unigram,
        "A_case": A_case,
        "B_case": B_case,
        "D_case": D_case,
        "case_artifact": case_artifact,
        "word_to_upos": word_to_upos,
        "clause_models": clause_models,
        "B_dec_widened": B_dec_widened,
        "D_dec_widened": D_dec_widened,
        "B_int_widened": B_int_widened,
        "D_int_widened": D_int_widened,
        "B_pool_widened": B_pool_widened,
        "D_pool_widened": D_pool_widened,
        "narrative_artifact": narrative_artifact,
        "narrative_models": narrative_models,
        "narrative_transition": narrative_transition,
    }


def _entropy(prob: np.ndarray) -> float:
    """Compute Shannon entropy in nats."""
    p = np.clip(prob, 1e-300, 1.0)
    return float(-np.sum(p * np.log(p)))


def generate_topdown_continuation(
    input_text: str,
    n_words: int = 10,
    gamma: float = 1.0,
    seed: int = 42,
    *,
    max_continuation_words: int = MAX_CONTINUATION_WORDS,
) -> dict[str, Any]:
    """Resume the input fragment through L3→L1g EFE policy selection."""
    models = get_dashboard_models()
    artifact = models["word_artifact"]
    word_prior = models["word_unigram"]
    habit_matrix = models["A_letter"]  # L1 bigram transition B
    rng = np.random.default_rng(seed)
    n_words = max(0, min(int(n_words), int(max_continuation_words)))

    raw_clean = input_text.strip()
    if not raw_clean:
        raw_clean = "the"

    tokens = raw_clean.split()
    last_word = tokens[-1].lower() if tokens else ""
    last_char = raw_clean[-1].lower() if raw_clean else " "

    # If input ended with space/punctuation, start fresh word; else continue in-flight prefix
    if input_text.endswith(" ") or last_char in {".", ",", "!", "?"}:
        accum_word = ""
        curr_char = " "
        mid_word = False
    else:
        accum_word = re.sub(r"[^a-z]", "", last_word)
        curr_char = last_char if last_char in ALPHA_TO_IDX else " "
        mid_word = bool(accum_word)

    curr_idx = ALPHA_TO_IDX.get(curr_char, ALPHA_TO_IDX[" "])
    generated_chars: list[str] = []
    generated_steps: list[dict[str, Any]] = []
    words_produced = 0

    # When the rollout resumes inside a partial word, the first space closes the
    # user's own word rather than producing a new one. Counting it spent one of
    # the requested words on a suffix the caller already had, so n_words words
    # were asked for and n_words - 1 came back — and n_words = 1 returned an
    # empty string. Only whole words generated from a fresh start count.
    pending_prefix_completion = mid_word

    max_steps = max(10, (n_words + 1) * 9)

    if n_words <= 0:
        return {
            "continuation_text": "",
            "full_text": input_text.rstrip(),
            "steps": [],
            "words_generated": 0,
            "requested_words": n_words,
        }

    for step in range(max_steps):
        log_habit = np.log(np.maximum(habit_matrix[:, curr_idx], 1e-300))
        pref = lexicon_preference(accum_word, artifact, word_prior)
        scores = gamma * np.asarray(pref, dtype=np.float64) + log_habit
        scores -= scores.max()
        weights = np.exp(scores)
        q_pi = weights / weights.sum()

        next_idx = int(rng.choice(N_SYMBOLS, p=q_pi))
        next_char = ALPHA_LIST[next_idx]
        generated_chars.append(next_char)

        top_indices = np.argsort(q_pi)[-4:][::-1]
        top_candidates = [
            {
                "symbol": repr(ALPHA_LIST[k]) if ALPHA_LIST[k] != " " else "' '",
                "prob": float(q_pi[k]),
                "c_val": float(pref[k]),
                "e_val": float(habit_matrix[k, curr_idx]),
            }
            for k in top_indices
        ]

        generated_steps.append(
            {
                "step": step,
                "prefix": accum_word,
                "prev_char": curr_char,
                "emitted_char": next_char,
                "top_candidates": top_candidates,
                "entropy": _entropy(q_pi),
            }
        )

        if next_char == " ":
            accum_word = ""
            if pending_prefix_completion:
                pending_prefix_completion = False
            else:
                words_produced += 1
                if words_produced >= n_words:
                    break
        else:
            accum_word += next_char

        curr_idx = next_idx
        curr_char = next_char

    continuation_text = "".join(generated_chars).strip()

    return {
        "continuation_text": continuation_text,
        "full_text": (input_text.rstrip() + " " + continuation_text).strip(),
        "steps": generated_steps,
        "words_generated": len(continuation_text.split()),
        "requested_words": n_words,
    }


def generate_next_tokens(
    prefix_chars: list[str], max_len: int = 15, gamma: float = 1.0
) -> list[str]:
    """Perform the L3 C-channel + L1g E-habit policy rollout."""
    models = get_dashboard_models()
    artifact = models["word_artifact"]
    word_prior = models["word_unigram"]
    habit_matrix = models["A_letter"]

    current_prefix = "".join(prefix_chars).strip().lower()
    last_word = current_prefix.split()[-1] if current_prefix.split() else ""
    last_char = prefix_chars[-1] if prefix_chars else " "
    curr_idx = ALPHA_TO_IDX.get(last_char, ALPHA_TO_IDX[" "])

    generated: list[str] = []
    accum_word = last_word

    for _ in range(max_len):
        log_habit = np.log(np.maximum(habit_matrix[:, curr_idx], 1e-300))
        pref = lexicon_preference(accum_word, artifact, word_prior)
        scores = gamma * np.asarray(pref, dtype=np.float64) + log_habit
        scores -= scores.max()
        weights = np.exp(scores)
        q_pi = weights / weights.sum()

        next_idx = int(np.argmax(q_pi))
        next_char = ALPHA_LIST[next_idx]
        generated.append(next_char)

        if next_char == " ":
            accum_word = ""
        else:
            accum_word += next_char
        curr_idx = next_idx

        if len(generated) > 4 and next_char in {".", "!", "?"}:
            break

    return generated


def infer_sentence_hierarchy(
    raw_text: str,
    n_continuation_words: int = 10,
    narrative_phase: int = 0,
    narrative_alpha: float = 0.5,
    rollout_seed: int = 42,
    *,
    max_continuation_words: int = MAX_CONTINUATION_WORDS,
) -> dict[str, Any]:
    """Execute complete multi-level active inference filtering on raw input text.

    Args:
        raw_text: The sentence to filter. Any input is accepted; punctuation-only
            and out-of-lexicon text degrade to reported fallbacks, never to a
            crash.
        n_continuation_words: Words for the top-down policy rollout.
        narrative_phase: Which Level 6 phase to read the sentence as (0-2). A
            single sentence has no document position, so the phase is an input,
            not something inferred from the word count.
        narrative_alpha: Level 6 -> Level 5 channel weight in [0, 1]. ``0.0``
            is the exact identity anchor: the clause prior is returned
            unmodified.
        max_continuation_words: Active configuration ceiling. The rollout cost
            is linear in it and this runs inside a request handler.
    """
    models = get_dashboard_models()
    n_continuation_words = max(0, min(int(n_continuation_words), int(max_continuation_words)))
    narrative_alpha = float(np.clip(narrative_alpha, 0.0, 1.0))
    normalized = normalize_text(raw_text)

    # Word segmentation
    raw_words = [w.strip() for w in re.split(r"\s+", raw_text.strip()) if w.strip()]
    if not raw_words:
        raw_words = ["the", "quick", "brown", "fox"]

    # Step-by-step Level 1 Letter Filtering
    letter_chars = list(normalized)
    if not letter_chars:
        letter_chars = list("the quick brown fox")

    l1_steps = []
    l1_prior = models["D_letter"].copy()
    A_letter = models["A_letter"]  # (30, 30) - column stochastic P(next | current)

    for idx, ch in enumerate(letter_chars):
        obs_idx = ALPHA_TO_IDX.get(ch, ALPHA_TO_IDX[" "])
        qs = np.zeros(N_SYMBOLS, dtype=np.float64)
        qs[obs_idx] = 1.0
        p_obs = float(l1_prior[obs_idx])
        entropy_val = _entropy(l1_prior)

        b_col = A_letter[:, obs_idx].copy()
        top_k_indices = np.argsort(b_col)[-5:][::-1]
        top_k_predictions = [
            {
                "symbol": repr(ALPHA_LIST[k]) if ALPHA_LIST[k] != " " else "' '",
                "prob": float(b_col[k]),
            }
            for k in top_k_indices
        ]

        l1_steps.append(
            {
                "step": idx,
                "char": ch,
                "obs_idx": obs_idx,
                "prior": l1_prior.tolist(),
                "posterior": qs.tolist(),
                "p_obs": p_obs,
                "entropy": entropy_val,
                "active_transition_col": b_col.tolist(),
                "top_predictions": top_k_predictions,
            }
        )

        l1_prior = b_col.copy()

    # Step-by-step Word, Phoneme, & Widened Sequence Inference
    word_steps = []
    l4_upos_sequence = []
    l4_case_beliefs = []
    l5_widened_symbols = []

    case_belief = models["D_case"].copy()
    case_upos_to_idx = models["case_artifact"].upos_to_idx
    default_upos = "NOUN"

    for w_idx, w_raw in enumerate(raw_words):
        w_alpha = re.sub(r"[^a-zA-Z]", "", w_raw).lower()
        # A token with nothing alphabetic left is punctuation or a numeral. The
        # widened chain was fitted with UD PUNCT tokens skipped, so inference
        # must skip them too; feeding one in injects a symbol the training
        # distribution never saw at that position.
        is_punct = not w_alpha
        w_clean = w_alpha or "word"

        gold_phones = word_to_phones(w_clean, models["cmu"]) or ["AH"]
        g2p_posteriors = []
        g2p_priors = []
        g2p_prior = models["D_g2p"].copy()
        for ch in w_clean:
            c_idx = ALPHA_TO_IDX.get(ch, ALPHA_TO_IDX["a"])
            g2p_priors.append(g2p_prior.copy())
            like = models["A_g2p"][c_idx, :]
            post = g2p_prior * like
            post_sum = post.sum()
            post = post / post_sum if post_sum > 0 else np.full(N_PHONES, 1.0 / N_PHONES)
            g2p_posteriors.append(post)
            g2p_prior = models["B_g2p"] @ post

        g2p_decoded_indices = [int(np.argmax(p)) for p in g2p_posteriors]
        g2p_decoded_phones = [PHONES[i] for i in g2p_decoded_indices]

        phone_steps = []
        for ch_idx, ch in enumerate(w_clean):
            p_post = g2p_posteriors[ch_idx]
            top_p_indices = np.argsort(p_post)[-3:][::-1]
            phone_steps.append(
                {
                    "char": ch,
                    "top_phones": [
                        {"phone": PHONES[pi], "prob": float(p_post[pi])} for pi in top_p_indices
                    ],
                    "entropy": _entropy(p_post),
                }
            )

        in_vocab = w_clean in models["word_vocab_set"]
        matching_words = [
            vw
            for vw in models["word_artifact"].vocab
            if vw.startswith(w_clean[: min(3, len(w_clean))])
        ][:12]

        # The part of speech is read once, from the lexicon, and kept. Rewriting
        # an unmapped tag to the default BEFORE the widened-symbol decision is
        # what silently deleted the AUX symbol: AUX has no column in the case
        # emission, so every auxiliary was relabeled NOUN and symbol 9 became
        # unreachable — taking the strongest interrogative cue in English with
        # it. The tag is therefore resolved first and only the case-emission
        # lookup falls back.
        if is_punct:
            upos_tag = "PUNCT"
            upos_source = "punctuation"
        else:
            looked_up = models["word_to_upos"].get(w_alpha)
            upos_source = "lexicon" if looked_up is not None else "fallback"
            upos_tag = looked_up if looked_up is not None else default_upos
        l4_upos_sequence.append(upos_tag)

        # A token carries a case role only if it is neither punctuation nor a
        # non-case part of speech, and its tag has a column in the emission.
        # A tag the lexicon has but the emission does not (PART, SCONJ, CCONJ)
        # contributes nothing, which matches training: those tokens reach the
        # `case_derived is not None` branch and are dropped when the case
        # oracle declines to assign them a role.
        case_bearing = (
            not is_punct and upos_tag not in NON_CASE_UPOS and upos_tag in case_upos_to_idx
        )

        if case_bearing:
            obs_upos_idx = case_upos_to_idx[upos_tag]
            like_case = models["A_case"][obs_upos_idx, :]
            post_case = case_belief * like_case
            post_case_sum = post_case.sum()
            post_case = (
                post_case / post_case_sum if post_case_sum > 0 else np.full(N_CASES, 1.0 / N_CASES)
            )
            l4_case_beliefs.append(post_case)
            l5_widened_symbols.append(int(np.argmax(post_case)))
            case_belief = models["B_case"] @ post_case
        else:
            # Not case-bearing: report the standing belief for display, but do
            # not fold a non-observation into the chain.
            post_case = case_belief
            if upos_tag == "VERB":
                l5_widened_symbols.append(VERB_SYMBOL)
            elif upos_tag == "AUX":
                l5_widened_symbols.append(AUX_SYMBOL)

        word_steps.append(
            {
                "word_idx": w_idx,
                "raw_word": w_raw,
                "clean_word": w_clean,
                "upos": upos_tag,
                "upos_source": upos_source,
                "case_bearing": bool(case_bearing),
                "gold_phones": gold_phones,
                "decoded_phones": g2p_decoded_phones,
                "phone_steps": phone_steps,
                "in_vocab": in_vocab,
                "candidate_trie_words": matching_words,
                "case_posterior": {CASES[i].name: float(post_case[i]) for i in range(N_CASES)},
                "top_case": CASES[int(np.argmax(post_case))].name if case_bearing else "—",
            }
        )

    # Level 5: Clause / Sentence Type Posterior
    # 1. Standard 8-case Markov mixture
    # Punctuation-only input, or input whose every token is a verb or auxiliary,
    # leaves no case observation at all. sequence_log_evidence indexes
    # sequence[0] unconditionally, so guard it and fall back to the type prior:
    # no observations means no evidence, not a crash and not a confident answer.
    case_indices = [int(np.argmax(b)) for b in l4_case_beliefs]
    if case_indices:
        _evidence, clause_posteriors_8case = sequence_log_evidence(
            case_indices, models["clause_models"]
        )
    else:
        clause_posteriors_8case = np.asarray(
            models["clause_models"]["type_prior"], dtype=np.float64
        )

    # 2. Widened 10-symbol Markov likelihood calculation
    p_dec_prior = float(models["clause_models"]["type_prior"][0])
    p_int_prior = float(models["clause_models"]["type_prior"][1])

    # With no case-bearing or verbal token in the input (punctuation-only text,
    # say) the widened sequence is empty and the likelihood carries no
    # evidence. Falling back to the type prior is the honest answer; the
    # previous code took log of D[symbol] on an empty list only by luck of the
    # guard, then still reported a posterior as if evidence had been seen.
    if l5_widened_symbols:
        lp_dec = float(np.log(models["D_dec_widened"][l5_widened_symbols[0]]))
        lp_int = float(np.log(models["D_int_widened"][l5_widened_symbols[0]]))
        for t in range(len(l5_widened_symbols) - 1):
            lp_dec += float(
                np.log(models["B_dec_widened"][l5_widened_symbols[t + 1], l5_widened_symbols[t]])
            )
            lp_int += float(
                np.log(models["B_int_widened"][l5_widened_symbols[t + 1], l5_widened_symbols[t]])
            )
    else:
        lp_dec = 0.0
        lp_int = 0.0

    log_unnorm_dec = lp_dec + np.log(p_dec_prior)
    log_unnorm_int = lp_int + np.log(p_int_prior)
    max_lp = max(log_unnorm_dec, log_unnorm_int)
    norm_factor = np.exp(log_unnorm_dec - max_lp) + np.exp(log_unnorm_int - max_lp)
    post_widened_dec = float(np.exp(log_unnorm_dec - max_lp) / norm_factor)
    post_widened_int = float(np.exp(log_unnorm_int - max_lp) / norm_factor)
    clause_posteriors_widened = [post_widened_dec, post_widened_int]

    # Level 6: Narrative Phase & the top-down channel D'(alpha).
    #
    # The dashboard is handed ONE sentence, so there is no document position to
    # infer a phase from — the honest move is to let the caller say which phase
    # the sentence is being read as, and to show what the channel does at each.
    # (The previous code computed tercile(min(len(words) - 1, 0), ...), and
    # min(n - 1, 0) is 0 for every non-empty input: the panel was pinned to
    # "opening" for all input, so the level it exists to demonstrate never
    # moved. tests/test_dashboard.py now pins the phase-response.)
    narrative_phase_idx = int(np.clip(narrative_phase, 0, N_PHASES - 1))
    narrative_phase_name = NARRATIVE_PHASES[narrative_phase_idx]

    # Narrative phase emission P(type | phase)
    phase_type_emission = models["narrative_models"]["emission"][:, narrative_phase_idx]

    blended_clause_prior = clause_prior_with_narrative(
        models["clause_models"],
        narrative_phase_idx,
        models["narrative_models"],
        alpha=narrative_alpha,
    )

    # The full channel response, so the panel shows the mechanism rather than a
    # single frozen number: type mass per phase across the alpha grid, with
    # alpha = 0 as the exact identity anchor and the phase-reversed scramble as
    # the entropy-preserving control.
    alpha_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    interrogative_idx = CLAUSE_TYPES.index("interrogative")
    block = slice(interrogative_idx * N_CLAUSE_CASES, (interrogative_idx + 1) * N_CLAUSE_CASES)
    channel_sweep = [
        {
            "alpha": a,
            "phases": {
                NARRATIVE_PHASES[ph]: float(
                    clause_prior_with_narrative(
                        models["clause_models"], ph, models["narrative_models"], alpha=a
                    )[block].sum()
                )
                for ph in range(N_PHASES)
            },
            "scrambled": {
                NARRATIVE_PHASES[ph]: float(
                    clause_prior_with_narrative(
                        models["clause_models"],
                        ph,
                        models["narrative_models"],
                        alpha=a,
                        scrambled=True,
                    )[block].sum()
                )
                for ph in range(N_PHASES)
            },
        }
        for a in alpha_grid
    ]

    # Separate L3→L1g policy rollout. None of the L2/L4/L5/L6 analyses above
    # conditions this generation path.
    topdown_continuation = generate_topdown_continuation(
        raw_text,
        n_words=n_continuation_words,
        seed=rollout_seed,
        max_continuation_words=max_continuation_words,
    )

    alphabet_labels = [repr(c) if c != " " else "' '" for c in ALPHA_LIST]

    return {
        "text": raw_text,
        "normalized": normalized,
        "words": raw_words,
        "topdown_continuation": topdown_continuation,
        "levels": {
            "l1_letter": {
                "steps": l1_steps,
                "alphabet": alphabet_labels,
                "n_symbols": N_SYMBOLS,
            },
            "l2_g2p": {
                "phones": list(PHONES),
                "n_phones": N_PHONES,
            },
            "l3_word": {
                "vocab_size": len(models["word_artifact"].vocab),
                "words": word_steps,
            },
            "l4_case": {
                "cases": [c.name for c in CASES],
                "steps": [w["case_posterior"] for w in word_steps],
                "case_bearing": [w["case_bearing"] for w in word_steps],
                "n_case_bearing": len(l4_case_beliefs),
            },
            "l5_clause": {
                "types": list(CLAUSE_TYPES),
                "prior": [p_dec_prior, p_int_prior],
                "posterior": clause_posteriors_widened,
                "posterior_8case": clause_posteriors_8case.tolist(),
                "posterior_widened": clause_posteriors_widened,
                "predicted_type": CLAUSE_TYPES[int(np.argmax(clause_posteriors_widened))],
                "widened_symbols_seq": l5_widened_symbols,
                "symbol_labels": list(WIDENED_SYMBOL_LABELS),
                "n_widened_symbols": N_WIDENED_SYMBOLS,
                "has_evidence": bool(l5_widened_symbols),
            },
            "l6_narrative": {
                "phases": list(NARRATIVE_PHASES),
                "current_phase": narrative_phase_name,
                "phase_idx": narrative_phase_idx,
                "alpha": narrative_alpha,
                "phase_type_emission": phase_type_emission.tolist(),
                "blended_prior_sample": blended_clause_prior[:8].tolist(),
                "channel_sweep": channel_sweep,
            },
        },
    }


def converse(
    message: str,
    history: list[dict[str, Any]] | None = None,
    n_words: int = 8,
    narrative_phase: int | None = None,
    narrative_alpha: float = 0.5,
    *,
    max_history_turns: int = MAX_HISTORY_TURNS,
    max_continuation_words: int = MAX_CONTINUATION_WORDS,
) -> dict[str, Any]:
    """One bidirectional turn: filter the user's message, then answer from the model.

    What the reply IS, stated plainly because the interface must not overstate
    it: the separate L3→L1g policy rollout, with Level 3's lexicon preference
    acting against Level 1g's letter habit and resuming from the user's own text.
    The L2, L4, L5, and L6 results analyze that input for display; they are not
    reply inputs and do not form a six-level generation cascade. This model has no
    semantics, no retrieval and no training objective that rewards answering a
    question. It has {free parameters} in the low thousands and it emits English
    letter statistics shaped by a 2,000-word lexicon. The reply is worth showing
    because its L3→L1g policy state is inspectable, not because it is a good
    answer, and the interface labels it that way.

    Conversation is the one setting where Level 6 has something real to do. A
    narrative phase is a document-position tercile, and a single sentence has no
    position — but a turn in a transcript does. With ``narrative_phase`` left
    unset the phase is derived from where this turn falls in the conversation so
    far, so the displayed L6→L5 analysis moves as the exchange progresses
    instead of sitting on "opening" forever. It does not enter the reply policy.

    Args:
        message: The user's turn.
        history: Prior turns, each ``{"role": "user"|"model", "text": str}``.
        n_words: Length of the generated reply, clamped by the rollout.
        narrative_phase: Force a phase (0-2). ``None`` derives it from the turn
            index, which is the honest default in a conversation.
        narrative_alpha: Level 6 to Level 5 channel weight.
        max_history_turns: Active configuration ceiling for retained entries.
        max_continuation_words: Active configuration ceiling for the reply.

    Returns:
        The reply, the full hierarchy for this turn, and the turn accounting.
    """
    turns = list(history or [])[-int(max_history_turns) :]
    turn_index = sum(1 for turn in turns if turn.get("role") == "user")

    if narrative_phase is None:
        # tercile() needs a length that contains the index; a conversation has no
        # known end, so the horizon grows with it. Early turns read as "opening"
        # and later ones advance, which is the behavior a transcript should show.
        horizon = max(turn_index + 1, N_PHASES)
        derived_phase = tercile(min(turn_index, horizon - 1), horizon)
        phase_source = "derived_from_turn"
    else:
        derived_phase = int(np.clip(narrative_phase, 0, N_PHASES - 1))
        phase_source = "explicit"

    # A fixed rollout seed makes every turn emit the same words, which reads as
    # a broken interface rather than as a deterministic model. The seed is
    # derived from the turn and the message instead: identical input at the same
    # point in a conversation still reproduces exactly, and successive turns do
    # not repeat themselves.
    # zlib.crc32, not hash(): Python salts string hashing per process, so hash()
    # would give a different reply for the same turn on every server restart —
    # reproducibility is a stated property of this project, and an interface
    # that quietly violates it is worse than one that repeats itself.
    seed = (zlib.crc32(f"{turn_index}\x00{message}".encode()) & 0x7FFFFFFF) or 1

    hierarchy = infer_sentence_hierarchy(
        message,
        n_continuation_words=n_words,
        narrative_phase=derived_phase,
        narrative_alpha=narrative_alpha,
        rollout_seed=seed,
        max_continuation_words=max_continuation_words,
    )
    continuation = hierarchy["topdown_continuation"]
    reply = continuation["continuation_text"].strip()

    l5 = hierarchy["levels"]["l5_clause"]
    l6 = hierarchy["levels"]["l6_narrative"]
    words = hierarchy["levels"]["l3_word"]["words"]

    return {
        "reply": reply or "(the rollout produced no words at this length)",
        "turn_index": turn_index,
        "rollout_seed": seed,
        "phase_source": phase_source,
        "hierarchy": hierarchy,
        # A compact per-turn summary, so the transcript can show what the model
        # inferred about the user's turn without re-walking the whole payload.
        "read_as": {
            "clause_type": l5["predicted_type"],
            "clause_confidence": max(l5["posterior"]),
            "has_evidence": l5["has_evidence"],
            "narrative_phase": l6["current_phase"],
            "surface": [l5["symbol_labels"][s] for s in l5["widened_symbols_seq"]],
            "in_lexicon": sum(1 for w in words if w["in_vocab"]),
            "tokens": len(words),
            "defaulted_tags": sum(1 for w in words if w["upos_source"] == "fallback"),
        },
    }
