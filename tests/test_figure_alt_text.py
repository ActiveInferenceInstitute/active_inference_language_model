"""Every figure has a short, token-free image alternative.

A caption in this paper is not an image alternative. It carries the population,
the provenance and the falsification condition, and several run past three
hundred words; read aloud in place of a picture it buries what a screen-reader
user needs. ``data/figure_alt_text.yaml`` holds the one-sentence description
instead, and the registry publishes it under ``metadata.alt_text``.

The template's renderer fails closed when a registry record exists without one,
which is how the gap was found: the HTML stage refused fourteen files at once
while every project test stayed green. That failure mode is the reason for this
file — the rule now has a check on this side of the boundary rather than only in
the renderer.

Guards on the guard: the map must be non-empty, a planted token must be caught,
and a planted over-long alternative must be caught.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALT_SOURCE = PROJECT_ROOT / "data" / "figure_alt_text.yaml"
REGISTRY = PROJECT_ROOT / "output" / "figures" / "figure_registry.json"

#: A manuscript token, a citation key, or inline math. None belongs in speech.
_UNSPEAKABLE = re.compile(r"\{\{|\}\}|\[@|\\\(|\$")

#: An alternative longer than this is a caption wearing the wrong hat. The
#: longest authored entry sits well below it; the bound exists to catch a future
#: paste of a full caption, not to police wording.
_MAX_WORDS = 60


def _alt_texts() -> dict[str, str]:
    payload = yaml.safe_load(ALT_SOURCE.read_text(encoding="utf-8"))
    return {label: " ".join(str(text).split()) for label, text in payload["figures"].items()}


def _registry_labels() -> set[str]:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {record["label"] for record in payload["figures"]}


def find_violations(alt_texts: dict[str, str]) -> list[str]:
    """Report every alternative that is unspeakable, empty, or caption-length."""
    violations: list[str] = []
    for label, text in sorted(alt_texts.items()):
        if not text:
            violations.append(f"{label}: empty alternative")
            continue
        if _UNSPEAKABLE.search(text):
            violations.append(f"{label}: contains a token, citation key, or math: {text[:60]}")
        if len(text.split()) > _MAX_WORDS:
            violations.append(
                f"{label}: {len(text.split())} words, over the {_MAX_WORDS}-word bound"
            )
    return violations


def test_the_map_is_not_empty() -> None:
    """An empty map would pass every rule below."""
    assert len(_alt_texts()) > 20


def test_every_alternative_is_speakable_and_short() -> None:
    assert not find_violations(_alt_texts()), "\n".join(find_violations(_alt_texts()))


def test_the_map_and_the_registry_name_the_same_figures() -> None:
    """No figure without an alternative, no alternative without a figure."""
    authored = set(_alt_texts())
    published = _registry_labels()
    assert authored - published == set(), f"alternatives for absent figures: {authored - published}"
    assert published - authored == set(), f"figures with no alternative: {published - authored}"


def test_the_registry_publishes_every_alternative() -> None:
    """The published record carries the authored text, not a derived one."""
    authored = _alt_texts()
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    for record in payload["figures"]:
        published = (record.get("metadata") or {}).get("alt_text")
        assert published == authored[record["label"]], record["label"]


def test_an_alternative_is_not_its_caption() -> None:
    """The two are different objects, and the shorter one is the alternative."""
    authored = _alt_texts()
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    for record in payload["figures"]:
        alt = authored[record["label"]]
        assert alt != record["caption"], record["label"]
        assert len(alt) < len(record["caption"]), record["label"]


def test_a_planted_token_is_caught() -> None:
    """Negative control: a number that could drift must not reach a reader here."""
    assert find_violations(
        {"fig:planted": "A bar chart of {{STAGE1_LETTER_PERPLEXITY}} perplexity."}
    )


def test_a_planted_caption_length_alternative_is_caught() -> None:
    """Negative control: pasting a caption in must fail."""
    assert find_violations({"fig:planted": "word " * (_MAX_WORDS + 1)})


def test_conforming_text_passes() -> None:
    """Positive control: the rule does not fire on a correct entry."""
    assert not find_violations({"fig:planted": "A bar chart of three comparators against one bar."})


# ── An alternative may only describe marks the builder actually draws ────────

#: A word that names a mark, and the matplotlib calls that could put that mark
#: on a canvas. An alternative that says "interval" for a figure whose builder
#: only calls ``ax.bar`` describes something the reader cannot see: three
#: entries shipped that way while every text-only check stayed green.
_MARK_CALLS: dict[str, tuple[str, ...]] = {
    "interval": ("errorbar", "yerr", "xerr", "fill_between", "fill_betweenx", "hlines", "vlines"),
    "errorbar": ("errorbar", "yerr", "xerr"),
    "heatmap": ("imshow", "pcolormesh", "matshow"),
}


def _builder_source(generated_by: str) -> str:
    """Source text of the figure module the registry names as the generator."""
    module_path = PROJECT_ROOT / "src" / Path(*generated_by.split(".")).with_suffix(".py")
    return module_path.read_text(encoding="utf-8")


def find_undrawn_marks(alt_texts: dict[str, str], sources: dict[str, str]) -> list[str]:
    """Report alternatives naming a mark whose builder source never draws one."""
    violations: list[str] = []
    for label, text in sorted(alt_texts.items()):
        source = sources.get(label)
        if source is None:
            continue
        lowered = text.lower()
        for word, calls in _MARK_CALLS.items():
            # A plotting call is matched at its call site; an error-bar keyword is
            # matched wherever it is spelled, since builders assemble kwargs.
            drawn = any(
                call in source if call.endswith("err") else f"{call}(" in source for call in calls
            )
            if word in lowered and not drawn:
                violations.append(f"{label}: says {word!r} but its builder draws none")
    return violations


def test_every_named_mark_is_drawn_by_its_builder() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    sources = {
        record["label"]: _builder_source(record["generated_by"]) for record in payload["figures"]
    }
    assert len(sources) > 20
    violations = find_undrawn_marks(_alt_texts(), sources)
    assert not violations, "\n".join(violations)


def test_an_interval_claimed_over_plain_bars_is_caught() -> None:
    """Negative control: the exact defect three entries shipped with."""
    alt = {"fig:planted": "The effect drawn as an interval against its threshold."}
    assert find_undrawn_marks(alt, {"fig:planted": "ax.bar(labels, values)\n"})


def test_an_interval_drawn_with_errorbars_passes() -> None:
    """Positive control: a builder that draws the mark is not flagged."""
    alt = {"fig:planted": "The effect drawn as an interval against its threshold."}
    assert not find_undrawn_marks(alt, {"fig:planted": "ax.errorbar(x, y, yerr=err)\n"})
