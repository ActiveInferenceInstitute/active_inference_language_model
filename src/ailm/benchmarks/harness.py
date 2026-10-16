"""Running a whole benchmark: replicates, agreement, and per-bucket metrics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ailm.benchmarks.calibration import reconstructed_calibration_metrics
from ailm.benchmarks.comparators import (
    _ailm_g2p_metrics,
    _letter_comparator,
    _phoneme_comparator,
    restrict_to_alphabetic,
    restrict_to_non_alphabetic,
    score_symbol_task,
)
from ailm.benchmarks.constants import (
    DEFAULT_CALIBRATION_N,
    DEFAULT_ENSEMBLE_N,
    DEFAULT_REPEAT_N,
    DEFAULT_SEED,
    DEFAULT_TASK_N,
    G2P_BUCKETS,
    LETTER_CENSUS,
    LETTER_TEXT,
    LETTER_WINDOW_LONG,
    LETTER_WINDOW_SHORT,
    OLLAMA_URL,
    PROTOCOL_VERSION,
    REPEAT_TRIAL_PROTOCOLS,
    SEED_OFFSETS,
)
from ailm.benchmarks.corpora import (
    letter_positions,
    phone_statistics,
    phoneme_positions,
    unique_letter_positions,
)
from ailm.benchmarks.ollama import (
    _answer_token_index,
    _generate_payload,
    _post_generate,
    ask_ollama,
)
from ailm.benchmarks.prompts import (
    build_prompt,
    build_undelimited_letter_prompt,
    parse_g2p_answer,
    parse_letter_answer,
    subsample_positions,
)
from ailm.benchmarks.scoring import (
    g2p_length_bucket,
    levenshtein_align,
    paired_difference_interval,
    paired_discordance,
    paired_power,
    wilson_interval,
)
from ailm.levels.letter import load_brown_counts
from ailm.levels.phoneme import (
    g2p_split,
    load_phoneme_counts,
)

ResponseLookup = Callable[[dict[str, Any], str], dict[str, Any]]
TextAnswerer = Callable[[str, str, str, int, int], str]


def seed_replicates(
    model: str,
    positions: list[tuple[str, str]],
    *,
    seed: int = DEFAULT_SEED,
    offsets: tuple[int, ...] = SEED_OFFSETS,
    answerer: TextAnswerer | None = None,
) -> dict[str, Any]:
    """Letter accuracy at the canonical seed's neighbors, at temperature zero.

    The canonical protocol decodes greedily, so the sampling seed cannot reach
    the output and the expected spread is exactly zero. That is a prediction,
    not a license to skip the measurement: the run below is what turns "seed
    does not matter here" from an assumption into a recorded observation. It is
    a determinism replication, not an estimate of run-to-run variance — there is
    no variance to estimate at temperature zero.
    """
    answer_call = ask_ollama if answerer is None else answerer
    accuracies = []
    for offset in offsets:
        correct = 0
        for prefix, gold in positions:
            answer = answer_call(model, build_prompt("letter", prefix), "letter", 32, seed + offset)
            parsed, _bad = parse_letter_answer(answer)
            correct += int(parsed == gold)
        accuracies.append(correct / len(positions) if positions else 0.0)
    return {
        "n": len(positions),
        "offsets": list(offsets),
        "accuracies": accuracies,
        "spread": max(accuracies) - min(accuracies) if accuracies else 0.0,
    }


def validated_repeat_protocols(protocols: Sequence[str]) -> tuple[str, ...]:
    """Refuse repeat-trial tags that are not one run's scoping of the prefixes.

    The check is on the general shape rather than on any particular run: one
    distinct tag per trial, each under its own prefix, and all of them carrying
    the same single-segment scope. An unscoped prefix, a mismatched pair, or two
    trials from different runs are all rejected here rather than becoming cache
    keys whose provenance no receipt can state.
    """

    tags = tuple(protocols)
    if len(tags) != len(REPEAT_TRIAL_PROTOCOLS):
        raise ValueError(
            f"repeat-trial protocols must name {len(REPEAT_TRIAL_PROTOCOLS)} trials, got {len(tags)}"
        )
    if len(set(tags)) != len(tags):
        raise ValueError("repeat-trial protocols must be distinct")
    scopes = set()
    for prefix, tag in zip(REPEAT_TRIAL_PROTOCOLS, tags, strict=True):
        if not isinstance(tag, str) or not tag.startswith(f"{prefix}/"):
            raise ValueError(f"repeat-trial protocol {tag!r} is not scoped under {prefix!r}")
        scope = tag[len(prefix) + 1 :]
        if not scope or "/" in scope or scope != scope.strip():
            raise ValueError(f"repeat-trial protocol {tag!r} carries an invalid run scope")
        scopes.add(scope)
    if len(scopes) != 1:
        raise ValueError("repeat-trial protocols must share one run scope")
    return tags


def repeat_agreement(
    model: str,
    positions: list[tuple[str, str]],
    *,
    repeat_trial_protocols: Sequence[str],
    seed: int = DEFAULT_SEED,
    endpoint: str = OLLAMA_URL,
    response_lookup: ResponseLookup | None = None,
) -> dict[str, Any]:
    """Fraction of positions where the same seed twice returns the same string.

    Both live calls bypass cache reads, but each reply is durably stored under a
    distinct cache-purpose protocol.  The two server payloads remain identical;
    the purpose tag prevents the second trial from replaying the first while
    making both observations independently recoverable by the publication
    receipt validator.

    ``repeat_trial_protocols`` is required and has no default. The two tags are
    run-scoped (see :func:`ailm.benchmarks.constants.repeat_trial_protocols`),
    and the run they name is the caller's to choose: a default here would either
    reinstate the unscoped key two runs collide on, or read a clock inside the
    measurement layer and make a call's cache identity depend on when the
    function ran.

    A measurement whose value is forced by memoization is worse than no
    measurement.  Live acquisition therefore sets ``cache_read=False`` for both
    trials.  Offline semantic replay supplies ``response_lookup`` and reads the
    two purpose-bound records without contacting the server.

    The premise this probe tests is also narrower than "the server is
    deterministic": it holds under *matched server state*, and a model load or an
    intervening workload changes that state. What is measured is reproducibility
    within one run, which is what the claims recorded from that run rest on.
    That is also why the tags are run-scoped rather than shared: two runs are two
    server states, so their trials are two measurements, not one repeated.
    """
    trial_protocols = validated_repeat_protocols(repeat_trial_protocols)
    agree = 0
    for prefix, _gold in positions:
        payload = _generate_payload(model, build_prompt("letter", prefix), "letter", 32, seed)
        bodies = [
            (
                response_lookup(payload, protocol)
                if response_lookup is not None
                else _post_generate(
                    payload,
                    endpoint=endpoint,
                    cache_protocol=protocol,
                    cache_read=False,
                )
            )
            for protocol in trial_protocols
        ]
        first, second = (str(body.get("response", "")) for body in bodies)
        agree += int(first == second)
    return {"n": len(positions), "agreement": agree / len(positions) if positions else 0.0}


def _g2p_bucket_metrics(
    records: list[dict[str, Any]], comparator_predictions: list[list[str]]
) -> dict[str, dict[str, float]]:
    """Per-length-bucket position accuracy for the probe and the comparator.

    Bucketing by gold phone count separates "the comparator is competitive"
    from "the comparator is competitive on short transparent words", which the
    pooled number cannot distinguish. Each system keeps the alignment rule its
    pooled number uses, so a bucket is comparable to the headline it decomposes.
    """
    buckets: dict[str, dict[str, float]] = {
        name: {"words": 0, "gold_phones": 0, "model_correct": 0, "comparator_correct": 0}
        for name, _low, _high in G2P_BUCKETS
    }
    for record, prediction in zip(records, comparator_predictions, strict=True):
        gold = record["gold"]
        bucket = buckets[g2p_length_bucket(len(gold))]
        bucket["words"] += 1
        bucket["gold_phones"] += len(gold)
        bucket["model_correct"] += record["aligned"]["correct_positions"]
        bucket["comparator_correct"] += sum(
            predicted == expected for predicted, expected in zip(prediction, gold, strict=False)
        )
    metrics: dict[str, dict[str, float]] = {}
    for name, bucket in buckets.items():
        phones = int(bucket["gold_phones"])
        # Wilson bounds on the bucket's own denominator (its gold phone positions),
        # so a stratified bar carries the same interval kind the pooled bar does.
        model_interval = wilson_interval(int(bucket["model_correct"]), phones)
        comparator_interval = wilson_interval(int(bucket["comparator_correct"]), phones)
        metrics[name] = {
            "words": bucket["words"],
            "gold_phones": bucket["gold_phones"],
            "model_position_accuracy": (bucket["model_correct"] / phones if phones else 0.0),
            "model_position_wilson_low": model_interval[0],
            "model_position_wilson_high": model_interval[1],
            "comparator_position_accuracy": (
                bucket["comparator_correct"] / phones if phones else 0.0
            ),
            "comparator_position_wilson_low": comparator_interval[0],
            "comparator_position_wilson_high": comparator_interval[1],
        }
    return metrics


def _load_benchmark_artifacts() -> tuple[dict, dict]:
    """Load the two offline comparator artifacts from package resources."""
    return load_brown_counts(), load_phoneme_counts()


def benchmark_model(
    model: str,
    *,
    repeat_trial_protocols: Sequence[str],
    letter_n: int = DEFAULT_TASK_N,
    phoneme_n: int = DEFAULT_TASK_N,
    g2p_n: int = DEFAULT_TASK_N,
    calibration_n: int = DEFAULT_CALIBRATION_N,
    ensemble_n: int = DEFAULT_ENSEMBLE_N,
    repeat_n: int = DEFAULT_REPEAT_N,
    long_window: int = LETTER_WINDOW_LONG,
    seed: int = DEFAULT_SEED,
    include_prompt_sensitivity: bool = False,
    response_lookup: ResponseLookup | None = None,
) -> dict[str, Any]:
    """Measure one model, or replay its exact identity-bound cached replies.

    ``repeat_trial_protocols`` names the run whose reliability trials this
    measurement writes or replays, and is required for the same reason it is
    required of :func:`repeat_agreement`: only the caller knows which run the
    resulting cache records belong to.
    """

    trial_protocols = validated_repeat_protocols(repeat_trial_protocols)

    def text_answer(
        selected_model: str,
        prompt: str,
        task: str,
        num_predict: int,
        selected_seed: int,
    ) -> str:
        """One model reply: a live call, or exact replay from the lookup."""
        if response_lookup is None:
            return ask_ollama(
                selected_model,
                prompt,
                task,
                num_predict,
                seed=selected_seed,
            )
        payload = _generate_payload(
            selected_model,
            prompt,
            task,
            num_predict,
            selected_seed,
        )
        return str(response_lookup(payload, PROTOCOL_VERSION).get("response", ""))

    def candidate_answer(
        selected_model: str,
        prompt: str,
        task: str,
        num_predict: int,
        selected_seed: int,
        top_logprobs: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Reply plus top-token candidates: live call, or exact replay."""
        if response_lookup is None:
            from ailm.benchmarks.ollama import ask_ollama_answer_candidates

            return ask_ollama_answer_candidates(
                selected_model,
                prompt,
                task,
                num_predict,
                selected_seed,
                top_logprobs,
            )
        payload = {
            **_generate_payload(
                selected_model,
                prompt,
                task,
                num_predict,
                selected_seed,
            ),
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }
        body = response_lookup(payload, PROTOCOL_VERSION)
        text = str(body.get("response", ""))
        raw_entries = body.get("logprobs") or []
        if not isinstance(raw_entries, list):
            raise ValueError("cached benchmark logprobs must be a list")
        entries = [entry for entry in raw_entries if isinstance(entry, dict)]
        tokens = [str(entry.get("token", "")) for entry in entries]
        index = _answer_token_index(tokens)
        if index is None or index >= len(entries):
            return text, []
        raw_candidates = entries[index].get("top_logprobs") or []
        if not isinstance(raw_candidates, list) or any(
            not isinstance(candidate, dict) for candidate in raw_candidates
        ):
            raise ValueError("cached benchmark answer candidates must be objects")
        return text, list(raw_candidates)

    brown, counts = _load_benchmark_artifacts()
    pool_short, pool_long = unique_letter_positions(
        LETTER_TEXT, window=LETTER_WINDOW_SHORT, long_window=long_window
    )
    # LETTER_CENSUS scores the whole pool. Resolving it here, against the pool
    # actually built above, is what keeps "every position" from silently
    # becoming "the first 7952 positions" if the fixture ever changes size.
    resolved_letter_n = len(pool_short) if letter_n == LETTER_CENSUS else letter_n
    letters = subsample_positions(pool_short, resolved_letter_n, seed)
    letters_long = subsample_positions(pool_long, resolved_letter_n, seed)
    if [gold for _prefix, gold in letters] != [gold for _prefix, gold in letters_long]:
        raise ValueError("context sweep lost position alignment; the two windows must pair")
    train_pairs, test_pairs = g2p_split(counts)
    train_unigrams, train_bigrams = phone_statistics(train_pairs)
    phones = subsample_positions(phoneme_positions(counts, test_pairs), phoneme_n, seed)
    words = subsample_positions(test_pairs, g2p_n, seed)

    # One discarded call before anything is scored. Greedy decoding reproduces
    # exactly under matched server state but not across a change in it, and the
    # first request after a load or a different workload is where that change
    # lands. Warming up moves it off the measured positions.
    text_answer(model, build_prompt("letter", "the"), "letter", 32, seed)

    letter_comparator = _letter_comparator(brown, letters)
    letter = score_symbol_task(
        model, "letter", letters, letter_comparator, seed=seed, answerer=text_answer
    )
    long_comparator = _letter_comparator(brown, letters_long)
    if long_comparator != letter_comparator:
        raise ValueError("the order-1 comparator must be context-invariant across windows")
    letter_long = score_symbol_task(
        model, "letter", letters_long, long_comparator, seed=seed, answerer=text_answer
    )
    # Same positions and inventory, different delimitation wording: a paired
    # prompt-sensitivity diagnostic, not a second headline accuracy.
    prompt_sensitivity = None
    if include_prompt_sensitivity:
        letter_undelimited = score_symbol_task(
            model,
            "letter",
            letters,
            letter_comparator,
            seed=seed,
            prompt_builder=build_undelimited_letter_prompt,
            answerer=text_answer,
        )
        prompt_paired = paired_discordance(
            letter_undelimited["model_correct"], letter["model_correct"]
        )
        prompt_interval = paired_difference_interval(
            prompt_paired["b"], prompt_paired["c"], len(letters)
        )
        prompt_sensitivity = {
            "undelimited": letter_undelimited,
            "b": prompt_paired["b"],
            "c": prompt_paired["c"],
            "p_value": prompt_paired["p_value"],
            "delta": (prompt_paired["b"] - prompt_paired["c"]) / len(letters) if letters else 0.0,
            "delta_low": prompt_interval[0] if letters else 0.0,
            "delta_high": prompt_interval[1] if letters else 0.0,
        }
    phoneme = score_symbol_task(
        model,
        "phoneme",
        phones,
        _phoneme_comparator(train_unigrams, train_bigrams, phones),
        seed=seed,
        answerer=text_answer,
    )

    records: list[dict[str, Any]] = []
    for word, gold in words:
        parsed, bad = parse_g2p_answer(
            text_answer(
                model,
                build_prompt("g2p", word),
                "g2p",
                max(16, len(gold) * 8 + 4),
                seed,
            )
        )
        records.append(
            {"word": word, "gold": gold, "bad": bad, "aligned": levenshtein_align(gold, parsed)}
        )
    pos = sum(record["aligned"]["correct_positions"] for record in records)
    exact = sum(int(record["aligned"]["exact"]) for record in records)
    unparsable = sum(int(record["bad"]) for record in records)
    total = sum(len(record["gold"]) for record in records)
    model_exact = [bool(record["aligned"]["exact"]) for record in records]
    comparator_position, comparator_exact, comparator_predictions = _ailm_g2p_metrics(
        train_pairs, words
    )
    comparator_exact_flags = [
        prediction == gold
        for prediction, (_word, gold) in zip(comparator_predictions, words, strict=True)
    ]
    comparator_levenshtein = (
        sum(
            levenshtein_align(gold, prediction)["correct_positions"]
            for prediction, (_word, gold) in zip(comparator_predictions, words, strict=True)
        )
        / total
        if total
        else 0.0
    )
    pos_interval = wilson_interval(pos, total)
    exact_interval = wilson_interval(exact, len(words))
    paired_exact = paired_discordance(model_exact, comparator_exact_flags)
    g2p = {
        "n": len(words),
        "position_accuracy": pos / total if total else 0.0,
        "position_wilson_low": pos_interval[0],
        "position_wilson_high": pos_interval[1],
        "exact_match_rate": exact / len(words) if words else 0.0,
        "exact_wilson_low": exact_interval[0],
        "exact_wilson_high": exact_interval[1],
        "unparsable_count": unparsable,
        "comparator_position_accuracy": comparator_position,
        "comparator_position_wilson_low": wilson_interval(
            round(comparator_position * total), total
        )[0],
        "comparator_position_wilson_high": wilson_interval(
            round(comparator_position * total), total
        )[1],
        "comparator_levenshtein_position_accuracy": comparator_levenshtein,
        "comparator_exact_match_rate": comparator_exact,
        "comparator_exact_wilson_low": wilson_interval(
            round(comparator_exact * len(words)), len(words)
        )[0],
        "comparator_exact_wilson_high": wilson_interval(
            round(comparator_exact * len(words)), len(words)
        )[1],
        "paired_exact": paired_exact,
        "delta": (paired_exact["b"] - paired_exact["c"]) / len(words) if words else 0.0,
        "delta_low": paired_difference_interval(paired_exact["b"], paired_exact["c"], len(words))[0]
        if words
        else 0.0,
        "delta_high": paired_difference_interval(paired_exact["b"], paired_exact["c"], len(words))[
            1
        ]
        if words
        else 0.0,
        "power": paired_power(paired_exact["b"], paired_exact["c"]),
        "buckets": _g2p_bucket_metrics(records, comparator_predictions),
    }

    alphabetic = restrict_to_alphabetic(letters, letter)
    non_alphabetic = restrict_to_non_alphabetic(letters, letter)
    context = {
        "n": len(letters),
        "window_short": LETTER_WINDOW_SHORT,
        "window_long": long_window,
        "short_top1_accuracy": letter["top1_accuracy"],
        "short_wilson_low": letter["wilson_low"],
        "short_wilson_high": letter["wilson_high"],
        "long_top1_accuracy": letter_long["top1_accuracy"],
        "long_wilson_low": letter_long["wilson_low"],
        "long_wilson_high": letter_long["wilson_high"],
        "comparator_short_top1_accuracy": letter["comparator_top1_accuracy"],
        "comparator_short_wilson_low": letter["comparator_wilson_low"],
        "comparator_short_wilson_high": letter["comparator_wilson_high"],
        "comparator_long_top1_accuracy": letter_long["comparator_top1_accuracy"],
        "comparator_long_wilson_low": letter_long["comparator_wilson_low"],
        "comparator_long_wilson_high": letter_long["comparator_wilson_high"],
        "unparsable_count": letter_long["unparsable_count"],
        "paired": paired_discordance(letter_long["model_correct"], letter["model_correct"]),
    }
    return {
        "model": model,
        "seed": seed,
        "protocol_version": PROTOCOL_VERSION,
        "letter_pool_total": len(letter_positions(LETTER_TEXT)),
        "letter_pool_unique": len(pool_short),
        # The count actually scored, after LETTER_CENSUS resolves. Reported by
        # the measurement rather than re-derived by the recorder, because the
        # recorder only has the requested value and the sentinel is not a count.
        "letter_positions_scored": resolved_letter_n,
        "letter": letter,
        "letter_alphabetic": alphabetic,
        "prompt_sensitivity": prompt_sensitivity,
        "letter_non_alphabetic": non_alphabetic,
        "phoneme": phoneme,
        "g2p": g2p,
        "context": context,
        "calibration": reconstructed_calibration_metrics(
            model,
            brown,
            subsample_positions(pool_short, calibration_n, seed),
            seed=seed,
            answer_candidates=candidate_answer,
        ),
        "ensemble": seed_replicates(
            model,
            subsample_positions(pool_short, ensemble_n, seed),
            seed=seed,
            answerer=text_answer,
        ),
        "repeat": repeat_agreement(
            model,
            subsample_positions(pool_short, repeat_n, seed),
            repeat_trial_protocols=trial_protocols,
            seed=seed,
            response_lookup=response_lookup,
        ),
    }
