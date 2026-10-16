"""The GNN model registry — naming grammar, layout, and coverage.

Why a grammar rather than a hand-kept index
-------------------------------------------
``gnn/`` began as three prose-named files. Once mutations became first-class
artifacts — a negative control is a *spec* that must fail, not a Python fixture —
a flat directory of prose names stopped carrying enough information to enumerate,
switch, compose or mutate models programmatically. An index file would have to be
maintained by hand alongside the specs, which makes it a second source of truth
and therefore the one that drifts.

The grammar instead makes identity derivable from the filename:

    <kind><level><part>_<slug>[@<mutation>].gnn.md

so ``discover_models()`` is a directory scan, hierarchy order is a sort, and a
mutation is bound to the model it derives from by construction.

What this file enforces
-----------------------
The checks are **default-deny**. Any spec file that does not parse, sits in the
wrong subdirectory, or derives from a model that does not exist is a failure —
the alternative is a spec nothing checks, which is worse than an absent one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ailm.gnn import (
    SPEC_KINDS,
    SPEC_SUFFIX,
    GnnModelId,
    GnnSpecError,
    discover_models,
    load_spec,
    parse_model_id,
    spec_dir,
    spec_path,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_spec_directory_prefers_real_source_over_stale_packaged_specs(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "src/ailm/gnn/registry.py"
    for installed in (False, True):
        root = tmp_path / ("installed" if installed else "checkout")
        package = root / "ailm" if installed else root / "src/ailm"
        module = package / "gnn/registry.py"
        module.parent.mkdir(parents=True)
        module.write_bytes(source.read_bytes())
        packaged = package / "_gnn"
        packaged.mkdir()
        canonical = root / "gnn"
        canonical.mkdir()
        expected = packaged if installed else canonical
        program = (
            "import runpy,sys; "
            "namespace=runpy.run_path(sys.argv[1]); "
            "assert str(namespace['spec_dir']()) == sys.argv[2]"
        )
        result = subprocess.run(
            [sys.executable, "-c", program, str(module), str(expected)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


#: Every level with shipped code must have a canonical spec. This is the
#: coverage obligation: a level whose model is not written down is a level whose
#: wiring nothing pins by name.
REQUIRED_MODELS: tuple[str, ...] = (
    "L1_letter",
    "L1g_letter_generation",
    "L2a_phoneme_sequence",
    "L2b_grapheme_to_phoneme",
    "L3_word",
    "L4_case",
    "L5_clause",
    "L6_narrative",
)

#: Public GNN prose whose evidence vocabulary must move with the canonical
#: specs. The L5/L6 mutation annotations repeat the governing evidence summary,
#: so they share this regression obligation even though their matrices are
#: deliberately broken negative controls.
GNN_EVIDENCE_PROSE: tuple[Path, ...] = (
    PROJECT_ROOT / "docs" / "gnn_composition.md",
    PROJECT_ROOT / "gnn" / "README.md",
    PROJECT_ROOT / "gnn" / "AGENTS.md",
    spec_path("L5_clause"),
    spec_path("L6_narrative"),
    spec_path("L5_clause@identity_joint_transition"),
    spec_path("L5_clause@single_type"),
    spec_path("L6_narrative@uniform_phase_prior"),
)

#: Exact retired assertions, rather than broad words such as ``cascade`` or
#: ``confirmed`` that remain legitimate inside an explicit denial. A helper
#: positive control below proves this list can reject the historical prose.
RETIRED_GNN_ASSERTIONS: tuple[str, ...] = (
    "remaining five specs are declared but not yet checked",
    "confirmed within its own stage",
    "confirmed pre-registered finding",
    "type-oracle ceiling clears",
    "document-position cascade",
    "top-down c cascade",
    "the narrative level's c over sentence types is the top-down channel",
    "the clause level's posterior moves the word level's prior",
    "no hierarchy above level 2",
    "recorded and resolved",
    "resolved null",
)


def _retired_gnn_assertions(text: str) -> list[str]:
    lowered = " ".join(text.lower().split())
    return [phrase for phrase in RETIRED_GNN_ASSERTIONS if phrase in lowered]


# ── The grammar ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("L1_letter", GnnModelId("L", 1, "", "letter", None)),
        ("L2a_phoneme_sequence", GnnModelId("L", 2, "a", "phoneme_sequence", None)),
        ("L3_word", GnnModelId("L", 3, "", "word", None)),
        (
            "L3_word@uniform_prior",
            GnnModelId("L", 3, "", "word", "uniform_prior"),
        ),
        ("X1_letter_to_phoneme", GnnModelId("X", 1, "", "letter_to_phoneme", None)),
    ],
)
def test_grammar_parses_each_shape(name, expected):
    assert parse_model_id(name) == expected
    assert str(expected) == name


def test_the_suffix_is_optional_when_parsing():
    assert parse_model_id(f"L3_word{SPEC_SUFFIX}") == parse_model_id("L3_word")


@pytest.mark.parametrize(
    "name",
    [
        "letter",  # no kind or level
        "L_letter",  # no level
        "L1letter",  # no separator
        "L1_Letter",  # not snake_case
        "L1_letter@",  # empty mutation
        "Q1_letter",  # unknown kind
        "L1_letter@Bad_Mutation",  # not snake_case
    ],
)
def test_the_grammar_refuses_what_it_cannot_bind(name):
    """A name that does not parse is refused, never defaulted.

    An identity guessed from an unparseable name would bind a conformance check
    to the wrong model — the same reason a symbolic dimension is rejected rather
    than defaulted to 1.
    """
    with pytest.raises(GnnSpecError, match="grammar"):
        parse_model_id(name)


def test_mutations_carry_their_base_model():
    mutation = parse_model_id("L3_word@uniform_prior")
    assert not mutation.is_canonical
    assert mutation.base == parse_model_id("L3_word")
    assert mutation.base.is_canonical


def test_sort_order_is_hierarchy_order_with_mutations_after_their_base():
    ids = [
        parse_model_id(n)
        for n in (
            "X1_letter_to_phoneme",
            "L3_word@uniform_prior",
            "L1_letter",
            "L3_word",
            "L2b_grapheme_to_phoneme",
            "L2a_phoneme_sequence",
        )
    ]
    assert [str(i) for i in sorted(ids, key=lambda i: i.sort_key)] == [
        "L1_letter",
        "L2a_phoneme_sequence",
        "L2b_grapheme_to_phoneme",
        "L3_word",
        "L3_word@uniform_prior",
        "X1_letter_to_phoneme",
    ]


def test_kind_dir_routes_each_kind_to_its_own_directory():
    assert parse_model_id("L3_word").kind_dir == "models"
    assert parse_model_id("L3_word@x").kind_dir == "mutations"
    assert parse_model_id("X1_a_b").kind_dir == "compositions"
    assert set(SPEC_KINDS) == {"models", "mutations", "compositions"}


# ── The registry ──────────────────────────────────────────────────────────────


def test_registry_is_not_empty():
    """NON-VACUITY: every check below would pass over an empty registry."""
    assert len(discover_models()) >= len(REQUIRED_MODELS)


@pytest.mark.parametrize("name", REQUIRED_MODELS)
def test_every_shipped_level_has_a_canonical_spec(name):
    assert spec_path(name).is_file(), (
        f"{name} has shipped code but no spec at {spec_path(name)}. A level whose "
        f"model is not written down has no declared matrix roles to pin."
    )


def test_clause_spec_reserves_t_for_time_not_sentence_type():
    text = spec_path("L5_clause").read_text(encoding="utf-8")
    assert "sum_z P(z) * P(o_{1:T} | z)" in text
    assert "q_T[(z, c)]" in text
    assert "sum_t P(t) * P(o_{1:T} | t)" not in text


def test_gnn_public_prose_names_every_canonical_spec():
    """The README's model inventory must cover the same default-deny set as code."""
    readme = (PROJECT_ROOT / "gnn" / "README.md").read_text(encoding="utf-8")
    missing = [name for name in REQUIRED_MODELS if f"models/{name}.gnn.md" not in readme]
    assert missing == [], f"canonical specs absent from gnn/README.md: {missing}"
    assert "all nine" in readme.lower(), "README no longer states the checked set's size"


def test_gnn_public_prose_has_no_retired_evidence_assertion():
    """Old status language must not survive beside corrected governing prose."""
    offenders = {
        str(path.relative_to(PROJECT_ROOT)): _retired_gnn_assertions(
            path.read_text(encoding="utf-8")
        )
        for path in GNN_EVIDENCE_PROSE
    }
    offenders = {path: hits for path, hits in offenders.items() if hits}
    assert offenders == {}, f"retired GNN evidence assertions: {offenders}"


def test_retired_gnn_assertion_detector_is_live():
    """POSITIVE CONTROL: the public-prose guard must reject the old wording."""
    planted = (
        "The remaining five specs are declared but not yet checked. "
        "Stage 6 was confirmed within its own stage by a document-position cascade. "
        "Stage 5 was a resolved null."
    )
    hits = _retired_gnn_assertions(planted)
    assert {
        "remaining five specs are declared but not yet checked",
        "confirmed within its own stage",
        "document-position cascade",
        "resolved null",
    } <= set(hits)


def test_l5_and_l6_specs_state_the_current_evidence_boundaries():
    """Pin the two qualifications most likely to regress into a passing stage."""
    l5 = " ".join(spec_path("L5_clause").read_text(encoding="utf-8").lower().split())
    assert "non-passing" in l5
    assert "gain remains unresolved" in l5
    assert "no equivalence margin was registered" in l5
    assert "receiver-sensitivity reference" in l5
    assert "not a ceiling or hard information bound" in l5

    l6 = " ".join(spec_path("L6_narrative").read_text(encoding="utf-8").lower().split())
    for required in (
        "observed-position",
        "imprecise and non-confirmatory",
        "no target-effect power analysis is recorded",
        "reversing a's tercile columns",
        "separate descriptive sequential score",
        "different estimand",
        "receiving clause agent's joint d prior",
        "does not modify or transmit l6's c",
        "not connected to the separate privileged punctuation-type-oracle→l3 or l3→l1g runs",
    ):
        assert required in l6, f"L6 spec omits evidence boundary {required!r}"


@pytest.mark.parametrize(
    "name",
    ("L5_clause@identity_joint_transition", "L5_clause@single_type"),
)
def test_l5_mutation_annotations_preserve_current_evidence_boundaries(name):
    """A negative-control matrix must not carry a superseded public claim."""
    text = " ".join(spec_path(name).read_text(encoding="utf-8").lower().split())
    for required in (
        "non-passing",
        "gain remains unresolved",
        "no equivalence margin was registered",
        "receiver-sensitivity reference",
        "not a ceiling or hard information bound",
        "not live-pymdp equivalence",
        "negative control, not an estimator",
    ):
        assert required in text, f"{name} omits evidence boundary {required!r}"


def test_l6_mutation_annotation_preserves_current_evidence_boundaries():
    """The uniform-D control must describe the same estimand limits as L6."""
    name = "L6_narrative@uniform_phase_prior"
    text = " ".join(spec_path(name).read_text(encoding="utf-8").lower().split())
    for required in (
        "observed-position",
        "imprecise and non-confirmatory",
        "no target-effect power analysis is recorded",
        "reversing a's tercile columns",
        "separate descriptive sequential score",
        "different estimand",
        "receiving clause agent's joint d prior",
        "does not modify or transmit l6's c",
        "separate privileged punctuation-type-oracle→l3 or l3→l1g experiments",
        "former consumes no l5 posterior",
    ):
        assert required in text, f"{name} omits evidence boundary {required!r}"


def test_composition_doc_separates_historical_interventions_from_connected_execution():
    raw = (PROJECT_ROOT / "docs" / "gnn_composition.md").read_text(encoding="utf-8")
    for seam in (
        "L1 → L2B",
        "L3 → L2B",
        "L3 → L1g",
        "L6 → L5",
        "punctuation-type oracle → L3",
    ):
        assert seam in raw, f"composition guide omits measured seam {seam}"
    text = " ".join(raw.split())
    assert "Their results do not establish a cascade effect" in text
    assert "connected runtime now executes its own declared messages" in text
    assert "does not send L6 `C`" in text
    assert "does not pass a Clause Agent posterior" in text
    assert "privileged oracle rather than a model level" in text


def test_every_discovered_spec_parses():
    """A spec that does not parse is a spec nothing checks."""
    for model_id, path in discover_models().items():
        try:
            load_spec(model_id)
        except GnnSpecError as exc:  # pragma: no cover - failure path
            pytest.fail(f"{path} does not parse: {exc}")


def test_no_spec_file_escapes_the_grammar():
    """DEFAULT-DENY over the whole tree, including files nothing references.

    Enumerating the specs that must conform would only constrain the ones
    somebody remembered to enumerate. This walks ``gnn/`` instead and requires
    every ``.gnn.md`` to parse and to sit in the directory its kind demands.
    """
    offenders: list[str] = []
    for path in sorted(spec_dir().rglob(f"*{SPEC_SUFFIX}")):
        relative = path.relative_to(spec_dir())
        if len(relative.parts) != 2 or relative.parts[0] not in SPEC_KINDS:
            offenders.append(f"{relative}: not directly inside one of {SPEC_KINDS}")
            continue
        try:
            model_id = parse_model_id(path.name)
        except GnnSpecError as exc:
            offenders.append(f"{relative}: {exc}")
            continue
        if model_id.kind_dir != relative.parts[0]:
            offenders.append(
                f"{relative}: is a {model_id.kind_dir!r} spec in {relative.parts[0]!r}"
            )
    assert offenders == [], "spec files outside the grammar:\n  " + "\n  ".join(offenders)


def test_no_stray_markdown_masquerades_as_a_spec():
    """A ``.md`` inside a spec directory is either a spec or a mistake.

    Documentation lives at ``gnn/``'s top level. A plain ``.md`` next to the
    specs would be skipped silently by ``discover_models``, which is how a spec
    stops being checked without anything failing.
    """
    strays = [
        str(path.relative_to(spec_dir()))
        for kind in SPEC_KINDS
        for path in sorted((spec_dir() / kind).glob("*.md"))
        if not path.name.endswith(SPEC_SUFFIX)
    ]
    assert strays == [], (
        f"Markdown files inside spec directories that are not specs: {strays}. "
        f"Move documentation to gnn/, or give them the {SPEC_SUFFIX} suffix."
    )


def test_every_mutation_derives_from_a_model_that_exists():
    """An orphan mutation tests nothing, and its base rename went unnoticed."""
    models = discover_models()
    canonical = {mid for mid in models if mid.is_canonical}
    orphans = [str(mid) for mid in models if not mid.is_canonical and mid.base not in canonical]
    assert orphans == [], (
        f"Mutations whose base model is missing: {orphans}. Either the base was "
        f"renamed without its mutations, or the mutation names a model that never "
        f"existed."
    )


def test_every_canonical_model_declares_its_level_consistently():
    """The filename's level must match what the spec says about itself.

    The grammar puts the level in the filename, which makes it renameable
    independently of the content — so the two are checked against each other
    rather than trusted to agree.
    """
    import re

    mismatches: list[str] = []
    for model_id in discover_models():
        if not model_id.is_canonical or model_id.kind != "L":
            continue
        annotation = load_spec(model_id).sections.get("ModelAnnotation", "")
        # Any phrasing is fine as long as the annotation names its own level and
        # no other. Pinning one sentence shape would make the check a style rule.
        levels = {int(n) for n in re.findall(r"[Ll]evel (\d+)\b", annotation)}
        if model_id.level not in levels:
            mismatches.append(
                f"{model_id}: ModelAnnotation never says 'level {model_id.level}' "
                f"(it mentions levels {sorted(levels) or 'none'})"
            )
    assert mismatches == [], (
        "Specs whose filename level disagrees with their own annotation:\n  "
        + "\n  ".join(mismatches)
    )


def test_spec_path_and_discovery_agree():
    """Two ways of locating a spec must not disagree."""
    for model_id, path in discover_models().items():
        assert spec_path(model_id) == path
        assert spec_path(str(model_id)) == path


def test_a_composition_mutant_files_under_mutations_not_beside_the_canonical_ones() -> None:
    """The directory has to discriminate must-pass from must-fail, for every kind.

    `kind_dir` once tested the composition kind before the canonical/mutant
    branch, so an ``X``-kind mutant would have been filed in ``compositions/``
    beside the specs that must conform. The whole mutation layer rests on that
    directory being a verdict, and a mutant sitting in the must-pass directory
    inverts the verdict for that file.

    No composition spec exists yet, so this was latent rather than live. It is
    asserted for every kind rather than for ``X`` alone, because a rule checked
    on one member is satisfied by any implementation that special-cases that
    member — which is exactly how the defect arose.
    """
    for kind in ("L", "X"):
        canonical = parse_model_id(f"{kind}1_alpha")
        mutant = parse_model_id(f"{kind}1_alpha@planted")
        assert mutant.kind_dir == "mutations", (
            f"a {kind!r}-kind mutant files under {mutant.kind_dir!r}; every mutant "
            f"belongs in mutations/ or the directory stops being a verdict"
        )
        assert canonical.kind_dir != "mutations", (
            f"a canonical {kind!r}-kind spec files under mutations/"
        )
