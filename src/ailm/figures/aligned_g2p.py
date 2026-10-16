"""The stipulated aligned spelling-to-pronunciation comparator arm (protocol P2, design B).

One panel of position accuracies: the native stipulated-fit engine, the
trivial closed-form ``T`` lookup, the context-free pronunciation-bigram
comparator, and the two attribution controls (the English-transfer fit and the
broken-correspondence fit), each as a position accuracy with a Wilson interval
derived at render time from the recorded accuracy and the recorded test
denominator (decision 101). The uniform floor is a reference rule, not a bar:
nothing was measured on it, so it is drawn as a line and carries no interval.

Every plotted value is read from the token map. No verdict word appears unless
it is derived from the ledger — the one verdict the panel states is the
recorder's own boolean on whether the native engine ties the trivial ``T``
lookup, rendered through the derived tie-word token (decision 116).
"""

from __future__ import annotations

from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ailm.figures._common import (
    MissingVariableError,
    load_manuscript_variables,
    token_float,
    token_int,
)
from ailm.statistics import wilson_interval
from ailm.viz_config import FONTS, PALETTE, apply_style

__all__ = ["aligned_g2p_data", "generate_aligned_g2p"]


#: The five measured arms, in draw order, with their reader-facing names. The
#: arm keys are the recorder's own label stems; adding an arm is a recording
#: change first, and this table is where the figure meets it.
_ARM_PRESENTATION: tuple[tuple[str, str], ...] = (
    ("native", "Native refit"),
    ("t_lookup", "T lookup"),
    ("context_free", "Context-free"),
    ("a1_english", "A1 English fit"),
    ("a2_broken", "A2 broken fit"),
)

_ARM_KEYS = tuple(arm for arm, _ in _ARM_PRESENTATION)


class AlignedG2pData(TypedDict):
    """Everything the panel draws, assembled exclusively from manuscript tokens."""

    arms: list[str]
    accuracy_pct: list[float]
    wilson_low_pct: list[float]
    wilson_high_pct: list[float]
    floor_pct: float
    inventory_size: int
    positions: int
    tie_word: str


def _token_bool(variables: dict[str, str], token: str) -> bool:
    """Read one recorded boolean out of the token map, failing closed."""
    value = variables.get(token)
    if value is None:
        raise MissingVariableError(f"missing token {token}; regenerate the token map")
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise MissingVariableError(f"token {token} is not a recorded boolean: {value!r}")


def aligned_g2p_data(variables: dict[str, str] | None = None) -> AlignedG2pData:
    """The five arms' position accuracies, render-time Wilson bounds, and floor.

    The recorder records accuracies and the two test counts and never
    intervals, so the bounds here are derived from the recorded accuracy times
    the recorded denominator — the decision-101 contract for a stratified
    benchmark bar. The tie verdict comes from the recorded boolean via the
    derived tie-word token; the panel states no other verdict.

    Raises:
        MissingVariableError: If a plotted arm's accuracy or the shared
            denominator is absent, or the two disagree about the arm's success
            count — a bar drawn from a denominator its accuracy does not
            determine would be exactly the drift this figure exists to catch.
    """
    tokens = variables if variables is not None else load_manuscript_variables()
    positions = token_int(tokens, "BENCH_SYNTHG2P_COMPARATOR_TEST_POSITION_COUNT")
    tie_word = tokens.get("BENCH_SYNTHG2P_COMPARATOR_NATIVE_TIE_WORD")
    if not isinstance(tie_word, str) or not tie_word.strip():
        raise MissingVariableError(
            "BENCH_SYNTHG2P_COMPARATOR_NATIVE_TIE_WORD is absent; regenerate the "
            "token map so the tie verdict can come from its recorded boolean"
        )
    _token_bool(tokens, "BENCH_SYNTHG2P_COMPARATOR_NATIVE_TIES_T_LOOKUP")

    arms: list[str] = []
    accuracy_pct: list[float] = []
    wilson_low_pct: list[float] = []
    wilson_high_pct: list[float] = []
    for arm, display in _ARM_PRESENTATION:
        label = f"BENCH_SYNTHG2P_COMPARATOR_{arm.upper()}_POSITION_ACCURACY"
        accuracy = token_float(tokens, label)
        scaled = accuracy * positions
        successes = round(scaled)
        if abs(scaled - successes) > 1e-6:
            raise MissingVariableError(
                f"{label} cannot come from an integer success count at n={positions}: {accuracy!r}"
            )
        low, high = wilson_interval(successes, positions)
        arms.append(display)
        accuracy_pct.append(accuracy * 100.0)
        wilson_low_pct.append(low * 100.0)
        wilson_high_pct.append(high * 100.0)

    return {
        "arms": arms,
        "accuracy_pct": accuracy_pct,
        "wilson_low_pct": wilson_low_pct,
        "wilson_high_pct": wilson_high_pct,
        "floor_pct": token_float(tokens, "BENCH_SYNTHG2P_COMPARATOR_UNIFORM_POSITION_FLOOR")
        * 100.0,
        "inventory_size": token_int(tokens, "PHONEME_COUNT_39"),
        "positions": positions,
        "tie_word": tie_word.strip(),
    }


def build_aligned_g2p_figure(variables: dict[str, str] | None = None) -> Figure:
    apply_style()
    data = aligned_g2p_data(variables)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    x = range(len(data["arms"]))
    lower = [
        max(accuracy - low, 0.0)
        for accuracy, low in zip(data["accuracy_pct"], data["wilson_low_pct"], strict=True)
    ]
    upper = [
        high - accuracy
        for accuracy, high in zip(data["accuracy_pct"], data["wilson_high_pct"], strict=True)
    ]
    ax.bar(
        list(x),
        data["accuracy_pct"],
        color=PALETTE["MODEL"],
        yerr=[lower, upper],
        error_kw={"ecolor": PALETTE["INK"], "elinewidth": 1.2, "capsize": 3},
    )
    floor = data["floor_pct"]
    ax.axhline(
        floor,
        color=PALETTE["INK"],
        linewidth=1.2,
        linestyle="--",
        label=f"uniform floor, 1 of {data['inventory_size']} ({floor:.2f}%)",
    )
    ax.set_xticks(list(x), data["arms"], rotation=20, ha="right")
    ax.set_ylabel("Position accuracy", fontsize=FONTS["axis_label"])
    ax.set_ylim(0, 100)
    ax.set_title(
        "Stipulated spelling-to-pronunciation arms:\n"
        f"the native fit {data['tie_word']} the trivial T lookup",
        fontsize=FONTS["title"],
    )
    ax.legend(fontsize=FONTS["legend"], loc="center right")
    fig.tight_layout()
    return fig


def generate_aligned_g2p(variables: dict[str, str] | None = None) -> Figure:
    """The aligned-G2P comparator panel; the pipeline owns saving."""
    return build_aligned_g2p_figure(variables)
