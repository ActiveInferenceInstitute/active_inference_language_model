"""Discovering the shipped specs and addressing them by model id.

`gnn/models/` is the source of truth for which models exist; this module is how
the rest of the codebase asks it."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ailm.gnn.parsing import parse_gnn_spec
from ailm.gnn.types import GnnSpec, GnnSpecError

# ── MODEL REGISTRY ────────────────────────────────────────────────────────────
#
# A flat directory of prose-named specs stops scaling once mutations and
# compositions are first-class artifacts rather than Python fixtures. The naming
# grammar below makes a spec's identity — where it sits in the hierarchy, which
# canonical model it derives from, what was mutated — readable from its filename
# and parseable by code, so specs can be enumerated, switched, composed and
# mutated without a hand-maintained index that would drift.


#: Suffix marking a file in ``gnn/`` as a spec rather than documentation.
SPEC_SUFFIX = ".gnn.md"

#: Subdirectories of ``gnn/``, keyed by the kind of spec each holds.
SPEC_KINDS: tuple[str, ...] = ("models", "mutations", "compositions")

_MODEL_ID_RE = re.compile(
    r"""^
    (?P<kind>L|X)                 # L = level model, X = composition
    (?P<level>\d+)                # hierarchy position
    (?P<part>[a-z]?)              # optional sub-model within a level
    _(?P<slug>[a-z0-9]+(?:_[a-z0-9]+)*)
    (?:@(?P<mutation>[a-z0-9]+(?:_[a-z0-9]+)*))?
    $""",
    re.VERBOSE,
)


@dataclass(frozen=True)
class GnnModelId:
    """The parsed identity of a GNN spec, derived from its filename.

    The grammar is ``<kind><level><part>_<slug>[@<mutation>]``:

    ==================================  ==========================================
    ``L1_letter``                       Level 1's canonical model
    ``L2a_phoneme_sequence``            Level 2, sub-model A
    ``L2b_grapheme_to_phoneme``         Level 2, sub-model B
    ``L3_word@uniform_prior``           a *mutation* of Level 3's canonical model
    ``X1_letter_to_phoneme``            a composition spanning levels
    ==================================  ==========================================

    Sorting by ``(level, part, mutation)`` therefore yields hierarchy order, and
    a mutation always sorts immediately after the model it derives from.

    Attributes:
        kind: ``"L"`` for a level model, ``"X"`` for a composition.
        level: Hierarchy position. For a composition, the lowest level spanned.
        part: Sub-model discriminator within a level, or ``""``.
        slug: What the model is *of*, in snake_case.
        mutation: The named defect this spec injects, or ``None`` if canonical.
    """

    kind: str
    level: int
    part: str
    slug: str
    mutation: str | None = None

    def __str__(self) -> str:
        base = f"{self.kind}{self.level}{self.part}_{self.slug}"
        return f"{base}@{self.mutation}" if self.mutation else base

    @property
    def is_canonical(self) -> bool:
        """True when this is a model rather than a deliberate mis-wiring."""
        return self.mutation is None

    @property
    def base(self) -> GnnModelId:
        """The canonical model a mutation derives from; itself when canonical."""
        return (
            self if self.is_canonical else GnnModelId(self.kind, self.level, self.part, self.slug)
        )

    @property
    def sort_key(self) -> tuple:
        """Hierarchy order, with each mutation directly after its base model."""
        return (self.kind != "L", self.level, self.part, self.slug, self.mutation or "")

    @property
    def filename(self) -> str:
        """The spec's filename, without a directory."""
        return f"{self}{SPEC_SUFFIX}"

    @property
    def kind_dir(self) -> str:
        """Which of :data:`SPEC_KINDS` this spec belongs in.

        The canonical/mutant split is tested FIRST, for every kind. An earlier
        version short-circuited on kind ``X`` before reaching that branch, which
        would have filed a composition mutant beside the canonical compositions
        and left the directory unable to discriminate must-pass from must-fail —
        the property the whole mutation layer rests on. No ``X`` spec exists yet,
        so the defect was latent rather than live; it is fixed here because a
        latent defect in a discriminator is worth less than nothing once someone
        trusts the directory.
        """
        if not self.is_canonical:
            return "mutations"
        return "compositions" if self.kind == "X" else "models"


def parse_model_id(name: str) -> GnnModelId:
    """Parse a spec identifier or filename into a :class:`GnnModelId`.

    Args:
        name: ``"L3_word"``, ``"L3_word@uniform_prior"``, or either with the
            ``.gnn.md`` suffix attached.

    Returns:
        The parsed identity.

    Raises:
        GnnSpecError: If the name does not match the grammar. A name that does
            not parse is refused rather than defaulted, for the same reason a
            symbolic dimension is: an identity guessed from an unparseable name
            would bind a conformance check to the wrong model.
    """
    stem = name[: -len(SPEC_SUFFIX)] if name.endswith(SPEC_SUFFIX) else name
    match = _MODEL_ID_RE.match(stem)
    if match is None:
        raise GnnSpecError(
            f"{name!r} does not match the GNN model-id grammar "
            f"'<kind><level><part>_<slug>[@<mutation>]', e.g. 'L2b_grapheme_to_phoneme' "
            f"or 'L3_word@uniform_prior'. See gnn/README.md."
        )
    return GnnModelId(
        kind=match["kind"],
        level=int(match["level"]),
        part=match["part"],
        slug=match["slug"],
        mutation=match["mutation"],
    )


def spec_path(name: str | GnnModelId) -> Path:
    """Resolve a spec identifier to its path under :func:`spec_dir`.

    Args:
        name: A model id, or a :class:`GnnModelId`.

    Returns:
        The path the spec is expected at, whether or not it exists.

    Raises:
        GnnSpecError: If ``name`` does not parse.
    """
    model_id = name if isinstance(name, GnnModelId) else parse_model_id(name)
    return spec_dir() / model_id.kind_dir / model_id.filename


def discover_models() -> dict[GnnModelId, Path]:
    """Enumerate every spec in ``gnn/``, keyed by parsed identity.

    Returns:
        Mapping from identity to path, in hierarchy order.

    Raises:
        GnnSpecError: If a spec file's name does not parse, or if a spec sits in
            the wrong subdirectory for its kind. Both are refused rather than
            skipped: a silently ignored spec is one nothing checks.
    """
    found: dict[GnnModelId, Path] = {}
    root = spec_dir()
    for kind_dir in SPEC_KINDS:
        directory = root / kind_dir
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(f"*{SPEC_SUFFIX}")):
            model_id = parse_model_id(path.name)
            if model_id.kind_dir != kind_dir:
                raise GnnSpecError(
                    f"{path} is a {model_id.kind_dir!r} spec but sits in "
                    f"{kind_dir!r}. Move it, or fix the name."
                )
            found[model_id] = path
    return dict(sorted(found.items(), key=lambda kv: kv[0].sort_key))


def load_spec(name: str | GnnModelId) -> GnnSpec:
    """Parse a spec by model id.

    Args:
        name: A model id such as ``"L3_word"`` or ``"L3_word@uniform_prior"``,
            or a :class:`GnnModelId`.

    Returns:
        The parsed spec.

    Raises:
        GnnSpecError: If the name does not parse, or the file is absent or
            malformed.
    """
    return parse_gnn_spec(spec_path(name))


def spec_dir() -> Path:
    """Return installed canonical specs or the source checkout's ``gnn/`` directory.

    Found by walking up from this file until a directory containing ``gnn/`` and
    ``src/`` appears, rather than by counting ``.parent`` hops. The count was
    wrong the moment this module moved one level deeper during the package
    split, and it failed in the least useful way: ``src/gnn/models/...`` is a
    plausible-looking path, so every caller reported "no such file" for a spec
    that was sitting exactly where it always had been.

    Raises:
        RuntimeError: If installed specs are absent and no ancestor holds both
            source directories.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "gnn").is_dir() and (
            candidate / "src/ailm/gnn/registry.py"
        ).resolve() == Path(__file__).resolve():
            return candidate / "gnn"
    packaged = Path(__file__).resolve().parents[1] / "_gnn"
    if packaged.is_dir():
        return packaged
    raise RuntimeError(
        f"no project root above {__file__} contains both gnn/ and src/; "
        f"ailm.gnn.registry cannot locate the shipped specs"
    )
