"""Position sampling and the prompt strings handed to an external model."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from typing import Any

from ailm.benchmarks.constants import DEFAULT_SEED
from ailm.levels.letter import ALPHA_TO_IDX
from ailm.levels.phoneme import (
    PHONES,
)
from ailm.levels.synthetic import SYNTHETIC_LETTERS


def subsample_positions(items: Iterable[Any], n: int, seed: int = DEFAULT_SEED) -> list[Any]:
    """Deterministic stride sample matching the project's non-prefix arithmetic."""
    values = list(items)
    if n < 0 or n > len(values):
        raise ValueError(f"requested {n} positions from {len(values)}")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return [] if n == 0 else [values[i * len(values) // n] for i in range(n)]


#: Prose statement of the letter inventory, and the symbol set it claims. The
#: prompt, the response schema, the parser and the gold stream must name the
#: same thirty symbols: a prompt that asks for a narrower set than the scorer
#: accepts penalizes the probe for obeying it, and space is the most frequent
#: target in the stream. The pair below is checked against ``ALPHA_LIST`` in
#: ``tests/test_benchmarks.py`` so the sentence cannot drift from the inventory
#: it describes.
LETTER_INVENTORY_PHRASE = "a lowercase letter a-z, a space, an apostrophe, a hyphen, or a period"
LETTER_INVENTORY_SYMBOLS = frozenset("abcdefghijklmnopqrstuvwxyz '-.")

#: The alphabetic subset of the inventory. The scored alphabet has thirty
#: symbols, but "next letter" can reasonably mean either that or the
#: twenty-six alphabetic ones, and the choice is not neutral: a probe that
#: under- or over-produces space, apostrophe, hyphen and period is scored
#: differently under each. The benchmark reports both arms from one run rather
#: than picking the inventory that flatters either system.
ALPHABETIC_SYMBOLS = frozenset("abcdefghijklmnopqrstuvwxyz")

#: The stipulated-language response contract is deliberately narrower than the
#: English letter contract. Its gold stream contains only these symbols, so
#: offering space and punctuation would make the displayed 1/26 floor disagree
#: with the response opportunity the external probe actually received.
SYNTHETIC_LETTER_INVENTORY_PHRASE = "a lowercase letter a-z"
SYNTHETIC_LETTER_INVENTORY_SYMBOLS = frozenset(SYNTHETIC_LETTERS)


def build_prompt(task: str, value: str) -> str:
    """Build the fixed instruction sent for one independent position."""
    if task == "letter":
        return f'Predict the next character in this sequence. The answer is exactly one character from this inventory: {LETTER_INVENTORY_PHRASE}. The sequence is delimited by angle brackets and a trailing space inside them is part of it. Reply with exactly one JSON object, {{"answer":"<one character>"}}, and no other text. Sequence: <{value}>'
    if task == "synthetic_letter":
        return f'Predict the next character in this stipulated sequence. The answer is exactly one character from this inventory: {SYNTHETIC_LETTER_INVENTORY_PHRASE}. The sequence is delimited by angle brackets. Reply with exactly one JSON object, {{"answer":"<one character>"}}, and no other text. Sequence: <{value}>'
    if task == "phoneme":
        return f'Predict the next ARPAbet phoneme in this sequence. Reply with exactly one JSON object, {{"answer":"<one ARPAbet symbol>"}}, and no other text. Sequence: {value}'
    if task == "g2p":
        return f'Return the ARPAbet pronunciation for this word. Reply with exactly one JSON object, {{"answer":["PHONE1","PHONE2",...]}}, use only canonical ARPAbet symbols, and no other text. Word: {value}'
    raise ValueError(f"unknown task {task!r}")


def build_in_context_synthetic_letter_prompt(value: str, examples: list[tuple[str, str]]) -> str:
    """Synthetic-letter prompt with labeled observations and the exact inventory."""
    if not examples:
        raise ValueError(
            "in-context prompt requires at least one example; k=0 is the zero-shot "
            "synthetic arm, which build_prompt already produces"
        )
    shown = "\n".join(f"<{prefix}> -> {symbol!r}" for prefix, symbol in examples)
    return (
        f"Here are {len(examples)} observed continuations from the same stipulated "
        f"source, each a sequence in angle brackets followed by the character that "
        f"came next:\n{shown}\n\nPredict the next character in this stipulated "
        f"sequence. The answer is exactly one character from this inventory: "
        f"{SYNTHETIC_LETTER_INVENTORY_PHRASE}. The sequence is delimited by angle "
        f'brackets. Reply with exactly one JSON object, {{"answer":"<one character>"}}, '
        f"and no other text. Sequence: <{value}>"
    )


def derange_examples(
    examples: list[tuple[str, str]],
) -> list[tuple[str, str]] | None:
    """Permute symbols completely while preserving prefixes and symbol counts.

    A multiset admits a derangement exactly when no symbol occupies more than
    half its slots. Grouping equal symbols and rotating by the largest group
    size constructs one whenever that bound holds.
    """
    if not examples:
        return None
    size = len(examples)
    groups: dict[str, list[int]] = {}
    for index, (_prefix, symbol) in enumerate(examples):
        groups.setdefault(symbol, []).append(index)
    ordered_groups = sorted(groups.values(), key=len, reverse=True)
    largest = len(ordered_groups[0])
    if largest * 2 > size:
        return None

    slots = [index for group in ordered_groups for index in group]
    symbols = [examples[index][1] for index in slots]
    shifted = symbols[largest:] + symbols[:largest]

    deranged = list(examples)
    for slot, symbol in zip(slots, shifted, strict=True):
        deranged[slot] = (examples[slot][0], symbol)
    if any(new == old for (_p, new), (_q, old) in zip(deranged, examples, strict=True)):
        return None
    return deranged


def build_in_context_synthetic_letter_prompts(
    positions: list[tuple[str, str]],
    pool: list[tuple[str, str]],
    *,
    k: int,
    deranged: bool,
    example_seed: int,
    draw_attempts: int,
    arm: str,
) -> list[str]:
    """Build one synthetic k-shot arm under the exact zero-shot inventory contract."""
    example_sets = draw_in_context_example_sets(
        positions,
        pool,
        k=k,
        example_seed=example_seed,
        draw_attempts=draw_attempts,
        arm=arm,
    )
    prompts: list[str] = []
    for (prefix, _gold), (examples, shuffled) in zip(positions, example_sets, strict=True):
        if k == 0:
            prompts.append(build_prompt("synthetic_letter", prefix))
            continue
        prompts.append(
            build_in_context_synthetic_letter_prompt(prefix, shuffled if deranged else examples)
        )
    return prompts


def draw_in_context_example_sets(
    positions: list[tuple[str, str]],
    pool: list[tuple[str, str]],
    *,
    k: int,
    example_seed: int,
    draw_attempts: int,
    arm: str,
) -> list[tuple[list[tuple[str, str]], list[tuple[str, str]]]]:
    """Return the exact treatment/control example pairs shown at each position.

    Each scored position owns a derived RNG seed.  Adding or removing another
    position therefore cannot shift every later prompt's example draw, while
    treatment and deranged arms still receive the same prefixes and label
    multiset.  The second list is the complete label derangement of the first.
    """

    pairs, _attempts = draw_in_context_example_sets_with_diagnostics(
        positions,
        pool,
        k=k,
        example_seed=example_seed,
        draw_attempts=draw_attempts,
        arm=arm,
    )
    return pairs


def draw_in_context_example_sets_with_diagnostics(
    positions: list[tuple[str, str]],
    pool: list[tuple[str, str]],
    *,
    k: int,
    example_seed: int,
    draw_attempts: int,
    arm: str,
) -> tuple[
    list[tuple[list[tuple[str, str]], list[tuple[str, str]]]],
    list[int],
]:
    """Return example pairs and the accepted draw-attempt number per position."""

    if k < 0:
        raise ValueError("k must be non-negative")
    if draw_attempts <= 0:
        raise ValueError("draw_attempts must be positive")
    if k > len(pool):
        raise ValueError(f"cannot draw {k} examples from a pool of {len(pool)}")
    if k == 0:
        return [([], []) for _position in positions], [1 for _position in positions]

    result: list[tuple[list[tuple[str, str]], list[tuple[str, str]]]] = []
    attempts_used: list[int] = []
    for prefix, _gold in positions:
        rng = random.Random(f"{example_seed}:{k}:{prefix}")
        examples, shuffled, accepted_attempt = _draw_derangeable_examples_with_attempts(
            rng,
            pool,
            k,
            prefix=prefix,
            arm=arm,
            draw_attempts=draw_attempts,
        )
        result.append((examples, shuffled))
        attempts_used.append(accepted_attempt)
    return result, attempts_used


def _draw_derangeable_examples_with_attempts(
    rng: random.Random,
    pool: list[tuple[str, str]],
    k: int,
    *,
    prefix: str,
    arm: str,
    draw_attempts: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], int]:
    """Draw a derangeable set and report its one-based accepted attempt."""

    for attempt in range(1, draw_attempts + 1):
        examples = rng.sample(pool, k)
        shuffled = derange_examples(examples)
        if shuffled is not None:
            return examples, shuffled, attempt
    raise ValueError(
        f"in-context probe refused: no derangeable draw of {k} examples appeared in "
        f"{draw_attempts} attempts at prefix {prefix!r} in arm {arm}. No claims written."
    )


def build_undelimited_letter_prompt(value: str) -> str:
    """Letter prompt without delimiters or a trailing-space notice."""
    return (
        "Predict the next character in this sequence. The answer is exactly one "
        f"character from this inventory: {LETTER_INVENTORY_PHRASE}. Reply with "
        'exactly one JSON object, {"answer":"<one character>"}, and no other '
        f"text. Sequence: {value}"
    )


def _contract_answer(answer: str) -> Any:
    """Extract the value from the strict Ollama JSON contract."""
    try:
        value = json.loads(answer.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != {"answer"}:
        return None
    return value["answer"]


def parse_letter_answer(answer: str) -> tuple[str | None, bool]:
    """Parse one letter answer, treating a lone space as the symbol it is.

    Space is a scored inventory symbol and the most frequent target in the
    stream, so it is matched before any whitespace trimming: stripping first and
    asking for length one afterwards would turn a correct space answer into an
    empty string and score it invalid *and* wrong. Padding around a non-space
    symbol is still tolerated.
    """
    contracted = _contract_answer(answer)
    if not isinstance(contracted, str):
        return (None, True)
    token = contracted.lower()
    if token == " ":
        return (" ", False)
    stripped = token.strip()
    return (stripped, False) if len(stripped) == 1 and stripped in ALPHA_TO_IDX else (None, True)


def parse_synthetic_letter_answer(answer: str) -> tuple[str | None, bool]:
    """Parse against the stipulated language's exact 26-symbol opportunity set."""
    if not isinstance(answer, str) or answer != answer.strip():
        return (None, True)
    contracted = _contract_answer(answer)
    if not isinstance(contracted, str) or contracted not in SYNTHETIC_LETTER_INVENTORY_SYMBOLS:
        return (None, True)
    return (contracted, False)


def parse_phoneme_answer(answer: str) -> tuple[str | None, bool]:
    """Parse a single-phoneme answer; ``(None, True)`` when invalid/absent."""
    contracted = _contract_answer(answer)
    tokens = [contracted.upper()] if isinstance(contracted, str) else []
    return (tokens[0], False) if len(tokens) == 1 and tokens[0] in PHONES else (None, True)


def parse_g2p_answer(answer: str) -> tuple[list[str], bool]:
    """Parse a full G2P sequence; ``([], True)`` when any token is off-inventory."""
    contracted = _contract_answer(answer)
    tokens = [str(token).upper() for token in contracted] if isinstance(contracted, list) else []
    return (tokens, False) if tokens and all(token in PHONES for token in tokens) else ([], True)
