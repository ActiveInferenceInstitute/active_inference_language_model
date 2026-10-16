"""This project's own decoders, scored on the same positions as the external models."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ailm.benchmarks.constants import DEFAULT_SEED
from ailm.benchmarks.ollama import ask_ollama
from ailm.benchmarks.prompts import (
    ALPHABETIC_SYMBOLS,
    build_prompt,
    parse_letter_answer,
    parse_phoneme_answer,
)
from ailm.benchmarks.scoring import (
    paired_difference_interval,
    paired_discordance,
    paired_power,
    wilson_interval,
)
from ailm.levels.letter import ALPHA_LIST, ALPHA_TO_IDX, build_letter_transition, normalize_text
from ailm.levels.phoneme import (
    PHONES,
    build_g2p_emission,
    build_g2p_start,
    build_g2p_transition,
    build_phone_transition,
)


def _letter_comparator(counts: dict, positions: list[tuple[str, str]]) -> list[str]:
    matrix = np.asarray(build_letter_transition(counts))[0]
    return [
        ALPHA_LIST[int(np.argmax(matrix[:, ALPHA_TO_IDX[prefix[-1]]]))]
        if prefix
        else ALPHA_LIST[int(np.argmax(matrix.mean(axis=1)))]
        for prefix, _ in positions
    ]


def _phoneme_comparator(
    unigrams: dict[str, int], bigrams: dict[str, int], positions: list[tuple[str, str]]
) -> list[str]:
    matrix = np.asarray(build_phone_transition(bigrams))[0, :, :, 0]
    priors = np.asarray([unigrams.get(phone, 0) for phone in PHONES])
    return [
        PHONES[int(np.argmax(matrix[:, PHONES.index(prefix.split()[-1])]))]
        if prefix
        else PHONES[int(np.argmax(priors))]
        for prefix, _ in positions
    ]


def _ailm_g2p_decode(
    emission: np.ndarray, transition: np.ndarray, start: np.ndarray, word: str
) -> list[str]:
    """Independent numpy filtering decoder using the live G2P matrices."""
    letters = [c for c in normalize_text(word) if c in ALPHA_TO_IDX]
    prior = start.copy()
    decoded: list[str] = []
    for letter in letters:
        likelihood = emission[ALPHA_TO_IDX[letter], :]
        posterior = likelihood * prior
        total = posterior.sum()
        posterior = posterior / total if total else prior
        decoded.append(PHONES[int(np.argmax(posterior))])
        prior = transition @ posterior
    return decoded


def _ailm_g2p_metrics(
    train_pairs: list[tuple[str, list[str]]], words: list[tuple[str, list[str]]]
) -> tuple[float, float, list[list[str]]]:
    """Comparator G2P metrics *and* the per-word decodes they were computed from.

    Returning the decodes lets the caller build the paired exact-match flags
    without decoding a second time. Discarding them costs a full rebuild of the
    emission, transition and start matrices per scored word, which is a training
    pass over every train pair for each word decoded.
    """
    emission = np.asarray(build_g2p_emission(train_pairs))[0]
    transition = np.asarray(build_g2p_transition(train_pairs))[0, :, :, 0]
    start = np.asarray(build_g2p_start(train_pairs)).ravel()
    predictions = [_ailm_g2p_decode(emission, transition, start, word) for word, _ in words]
    total = sum(len(gold) for _, gold in words)
    position = (
        # The inner loop must not reuse ``gold``: rebinding the word's gold
        # sequence to one of its own phones inside the comprehension that
        # iterates over it reads as a bug even where it happens to work.
        sum(
            predicted_phone == gold_phone
            for prediction, (_, gold) in zip(predictions, words, strict=True)
            for predicted_phone, gold_phone in zip(prediction, gold, strict=False)
        )
        / total
        if total
        else 0.0
    )
    exact = (
        sum(prediction == gold for prediction, (_, gold) in zip(predictions, words, strict=True))
        / len(words)
        if words
        else 0.0
    )
    return position, exact, predictions


def score_symbol_task(
    model: str,
    task: str,
    positions: list[tuple[str, str]],
    comparator: list[str],
    *,
    seed: int = DEFAULT_SEED,
    prompt_builder: Callable[[str], str] | None = None,
    answerer: Callable[[str, str, str, int, int], str] | None = None,
) -> dict[str, Any]:
    """Score one model and its comparator on the same held-out positions.

    Runs the model answerer (or a live ``ask_ollama`` call) over every
    position, scores exact-match against gold, and returns Wilson intervals
    plus the paired discordance table against the comparator answers.
    """
    answer_call = ask_ollama if answerer is None else answerer
    correct = unparsable = 0
    model_correct: list[bool] = []
    for prefix, gold in positions:
        prompt = prompt_builder(prefix) if prompt_builder else build_prompt(task, prefix)
        answer = answer_call(model, prompt, task, 32, seed)
        parsed, bad = (
            parse_letter_answer(answer) if task == "letter" else parse_phoneme_answer(answer)
        )
        unparsable += int(bad)
        is_correct = parsed == gold
        model_correct.append(is_correct)
        correct += int(is_correct)
    n = len(positions)
    comp = (
        sum(pred == gold for pred, (_, gold) in zip(comparator, positions, strict=True)) / n
        if n
        else 0.0
    )
    interval = wilson_interval(correct, n)
    comparator_correct = [
        pred == gold for pred, (_, gold) in zip(comparator, positions, strict=True)
    ]
    paired = paired_discordance(model_correct, comparator_correct)
    return {
        "n": n,
        "top1_accuracy": correct / n if n else 0.0,
        "wilson_low": interval[0],
        "wilson_high": interval[1],
        "comparator_top1_accuracy": comp,
        "comparator_wilson_low": wilson_interval(sum(comparator_correct), n)[0],
        "comparator_wilson_high": wilson_interval(sum(comparator_correct), n)[1],
        "unparsable_count": unparsable,
        "paired": paired,
        "delta": (paired["b"] - paired["c"]) / n if n else 0.0,
        "delta_low": paired_difference_interval(paired["b"], paired["c"], n)[0] if n else 0.0,
        "delta_high": paired_difference_interval(paired["b"], paired["c"], n)[1] if n else 0.0,
        "power": paired_power(paired["b"], paired["c"]),
        "model_correct": model_correct,
        "comparator_correct": comparator_correct,
    }


def restrict_to_alphabetic(
    positions: list[tuple[str, str]], scored: dict[str, Any]
) -> dict[str, Any]:
    """Rescore an already-measured letter task on alphabetic-gold positions only.

    "Predict the next letter" admits two honest readings — the thirty-symbol
    stream the corpus actually contains, or the twenty-six alphabetic symbols
    the phrase most naturally suggests — and a probe's handling of space,
    apostrophe, hyphen and period can move the comparison either way. Reporting
    the arm that favors one system and calling it the result would be a choice
    disguised as a measurement, so both arms are reported from the same calls.
    No additional requests are issued: this is a restriction of the recorded
    per-position outcomes, not a second run.
    """
    keep = [index for index, (_prefix, gold) in enumerate(positions) if gold in ALPHABETIC_SYMBOLS]
    model_correct = [scored["model_correct"][index] for index in keep]
    comparator_correct = [scored["comparator_correct"][index] for index in keep]
    total = len(keep)
    paired = paired_discordance(model_correct, comparator_correct)
    # Wilson bounds on the restricted denominator, so the arm carries the same
    # interval the headline letter task does rather than a bare point.
    interval = wilson_interval(sum(model_correct), total)
    comparator_interval = wilson_interval(sum(comparator_correct), total)
    return {
        "n": total,
        "top1_accuracy": sum(model_correct) / total if total else 0.0,
        "wilson_low": interval[0],
        "wilson_high": interval[1],
        "comparator_top1_accuracy": sum(comparator_correct) / total if total else 0.0,
        "comparator_wilson_low": comparator_interval[0],
        "comparator_wilson_high": comparator_interval[1],
        "paired": paired,
    }


def restrict_to_non_alphabetic(
    positions: list[tuple[str, str]], scored: dict[str, Any]
) -> dict[str, Any]:
    """Rescore a letter task on the non-alphabetic-gold positions only.

    The exact complement of :func:`restrict_to_alphabetic`, and the reason the
    two letter arms disagree. Almost every non-alphabetic target is a space, and
    a probe that never emits one loses all of them regardless of how well it
    predicts letters — so the full-inventory arm measures inventory compliance
    at these positions and predictive skill at the others. Reporting the two
    arms without this split shows *that* they disagree while hiding *why*.

    Like the alphabetic restriction this issues no additional requests: it is a
    partition of the recorded per-position outcomes. Together the two arms
    cover every scored position exactly once.
    """
    keep = [
        index for index, (_prefix, gold) in enumerate(positions) if gold not in ALPHABETIC_SYMBOLS
    ]
    model_correct = [scored["model_correct"][index] for index in keep]
    comparator_correct = [scored["comparator_correct"][index] for index in keep]
    total = len(keep)
    return {
        "n": total,
        "top1_accuracy": sum(model_correct) / total if total else 0.0,
        "comparator_top1_accuracy": sum(comparator_correct) / total if total else 0.0,
        "paired": paired_discordance(model_correct, comparator_correct),
    }
