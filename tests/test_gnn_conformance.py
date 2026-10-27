"""GNN conformance — the third independent guard on AILM's wiring orientation.

What this file proves
---------------------
``gnn/*.md`` declares each AILM level's structure and, crucially, the *semantic
role* of every matrix (``A=RecognitionMatrix``, ``B=TransitionMatrix``,
``D=Prior``). ``ailm.gnn`` checks a LIVE pymdp agent against that declaration.

The properties below have to hold for that to be worth anything; they are
tested here rather than asserted in prose:

1. **The specs describe the shipped models.** Every spec is checked against an
   agent built from the real vendored artifacts, not against hand-typed tuples.
2. **The checker can fail.** The positive controls below construct genuinely
   mis-wired pymdp agents — including the historical anti-informative A/B swap —
   and require the checker to reject them. Each control asserts on the reported
   violation TEXT, so an unrelated exception cannot satisfy it.
3. **Coverage is default-deny, with no exception list.**
   ``TestEveryShippedSpecIsCovered`` reads ``gnn/models/`` off disk and requires
   the covered set to equal what is there, exactly. A spec added later cannot go
   unchecked by being forgotten, and there is nowhere to record that one is
   allowed to: a spec that does not conform is a finding to fix or to record in
   ``docs/decisions.md``.

4. **Every declaration KIND has a reader.** A spec can be bound to an agent and
   still declare things nothing compares — which is how L1g's policy prior
   ``E`` stayed unverified while every coverage test above stayed green.
   ``TestEveryDeclarationHasAReader`` enumerates the sections, ontology terms,
   ``ModelParameters`` keys, structural forms and ``Learned`` arguments actually
   present under ``gnn/models/`` and requires each to be either named in an
   inventory of what reads it, or named in an inventory of what genuinely
   cannot be read, with the reason. Both inventories are default-deny: a ninth
   ontology term or a new ``Learned`` argument fails the suite rather than
   arriving unread.

Property 2 is the point of the file, and property 4 is what keeps property 2
honest. A conformance checker that cannot reject anything is a comment with a
test-shaped wrapper; one that rejects only what it happens to read is a comment
about the rest. Note in particular that the letter-level A/B swap has
*identical matrix shapes* to the correct wiring — the control passes only
because the checker also enforces the structural forms declared in
``InitialParameterization``.

Every spec under ``gnn/models/`` is bound here
------------------------------------------------------------------------------
``L1_letter``, ``L1g_letter_generation``, ``L2a_phoneme_sequence``,
``L2b_grapheme_to_phoneme``, ``L3_word``, ``L4_case``, ``L5_clause``,
``L5w_clause_widened`` and ``L6_narrative``. ``L5w_clause_widened`` is the
widened-observation arm registered under decision 112 and promoted to the
canonical shipped clause model by decision 114. ``L5_clause`` remains the
case-only ablation. Both are bound because coverage is default-deny and
there is nowhere to record that a spec may go unchecked.
Several are also conformance-checked inside their own stage
gate (``tests/test_stage3_gate.py`` and friends), where the check sits next to
the measurement it protects. That duplication is deliberate: the gate files bind
one spec each and skip when their artifact is absent, so only a suite that
enumerates the directory can notice a spec nobody bound at all.

``L3_word`` is the one multi-factor model, and the single-factor assertions stay
exact rather than being loosened to accommodate it: it has its own parallel
assertions keyed on ``num_hidden_states_f<N>``, ``B_f0``/``B_f1`` and
``D_f0``/``D_f1``.

Scope discipline
----------------
Conformance pins structure and declared roles. It does NOT verify the learned
probabilities: ``tests/test_oracle.py`` and the stage gate tests do that. No
assertion in this file should ever be cited as evidence that a number is right.

Upstream interoperability
-------------------------
``TestUpstreamInteroperability`` parses the same nine files with the real
GeneralizedNotationNotation Markdown parser. It skips cleanly when no upstream
checkout is present (set ``AILM_GNN_UPSTREAM`` to point at one), because AILM
must not take a 25-step pipeline as a test dependency — but where a checkout
exists it runs, so "GNN compatible" is demonstrated rather than claimed.

The two readers agree on every variable of every spec except one: upstream
normalizes the ASCII identifier ``pi`` in ``L1g_letter_generation`` to the Greek
letter ``π``. That single rename is recorded in :data:`UPSTREAM_NAME_ALIASES`
and asserted to be the *only* disagreement, so it cannot become a hiding place
for a real one.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ailm.gnn import (
    COMBINATION_RULES,
    GNN_SECTIONS,
    MATRIX_ROLES,
    SPEC_SUFFIX,
    GnnConformanceError,
    GnnSpec,
    GnnSpecError,
    LiveConformanceInputs,
    _check_combination_rule,
    _check_inference_configuration,
    assert_conforms,
    check_conformance,
    describe_shapes,
    load_spec,
    parse_gnn_spec,
    parse_gnn_string,
    parse_learned_arguments,
)
from ailm.levels.word import N_OBS as WORD_N_OBS
from ailm.levels.word import load_word_artifact
from ailm.verification.upstream_gnn import (
    default_upstream_root as _default_upstream_root,
)
from ailm.verification.upstream_gnn import (
    load_release_pin,
    selected_upstream_root,
    upstream_candidates,
    verify_release_checkout,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GNN_DIR = PROJECT_ROOT / "gnn"

#: Every shipped level, each with the spec that must describe it, in hierarchy
#: order. :class:`TestEveryShippedSpecIsCovered` requires this to equal what is
#: on disk under ``gnn/models/``, so extending the directory without extending
#: this tuple is a failure rather than a silent gap.
SPEC_NAMES: tuple[str, ...] = (
    "L1_letter",
    "L1g_letter_generation",
    "L2a_phoneme_sequence",
    "L2b_grapheme_to_phoneme",
    "L3_word",
    "L4_case",
    "L5_clause",
    "L5w_clause_widened",
    "L6_narrative",
)

#: Levels whose spec declares more than one hidden factor, and which therefore
#: size their factors with ``num_hidden_states_f<N>`` rather than a single
#: ``num_hidden_states``. Kept separate so the single-factor assertions below
#: stay exact instead of being loosened to accommodate them.
MULTIFACTOR_SPEC_NAMES: tuple[str, ...] = ("L3_word",)

#: The complement: everything sized by a scalar ``num_hidden_states``.
SINGLE_FACTOR_SPEC_NAMES: tuple[str, ...] = tuple(
    name for name in SPEC_NAMES if name not in MULTIFACTOR_SPEC_NAMES
)

#: Non-vacuity floor for the on-disk scan. Nine levels ship today; a discovery
#: that returned nothing would otherwise satisfy every set comparison below.
MIN_SHIPPED_SPECS = 9

#: Declared (num_obs, num_hidden_states) per single-factor level, pinned against
#: the code's real alphabet sizes. The G2P row is the crux of Stage 2: two
#: DIFFERENT alphabets, so the recognition matrix is not square. L4, L5 and L6
#: are asymmetric too — each observes the level below's state space.
EXPECTED_PARAMETERS: dict[str, tuple[int, int]] = {
    "L1_letter": (30, 30),
    "L1g_letter_generation": (30, 30),
    "L2a_phoneme_sequence": (39, 39),
    "L2b_grapheme_to_phoneme": (30, 39),
    "L4_case": (12, 8),
    "L5_clause": (8, 16),
    "L5w_clause_widened": (10, 20),
    "L6_narrative": (2, 3),
}

#: The authenticated Stage 3 artifact owns both structural factor sizes.  In
#: particular, the position factor follows the configured vocabulary horizon,
#: not the longest word that happens to survive a frequency bound.
STAGE3_WORD_ARTIFACT = load_word_artifact()

#: Declared (num_obs, per-factor hidden sizes) for each multi-factor level.
EXPECTED_FACTOR_PARAMETERS: dict[str, tuple[int, tuple[int, ...]]] = {
    "L3_word": (
        WORD_N_OBS,
        (len(STAGE3_WORD_ARTIFACT.vocab), STAGE3_WORD_ARTIFACT.n_positions),
    ),
}

#: The variable each spec's first directed connection must flow FROM: the prior
#: seeding the hidden state. Pinned per spec rather than matched by role, so a
#: spec that reordered its connections is a failure and not a shrug.
EXPECTED_PRIOR_SEED: dict[str, tuple[str, ...]] = {
    name: ("D_f0",) if name in MULTIFACTOR_SPEC_NAMES else ("D",) for name in SPEC_NAMES
}

#: Variable-name renames the upstream GNN parser performs, upstream name to
#: ours. Upstream normalizes the ASCII identifier ``pi`` to the Greek letter, so
#: the two readers disagree on that one key in ``L1g_letter_generation``.
#: Recorded rather than skipped, and asserted below to be the only difference
#: anywhere: an undocumented rename is a place a real disagreement could hide.
UPSTREAM_NAME_ALIASES: dict[str, str] = {"π": "pi"}

# ── DECLARATION INVENTORY ────────────────────────────────────────────────────
#
# The audit these tables encode started from a single finding: ``describe_shapes``
# read A/B/C/D and nothing else, so ``E=PolicyPrior`` / ``E=Uniform(30)`` was
# parsed, stored on the GnnSpec, and compared to nothing — while every coverage
# test passed, because coverage was over SPECS and not over what a spec DECLARES.
# The tables below close that class of gap by enumerating declaration kinds and
# refusing any that neither inventory names.

#: Ontology term -> what compares a variable carrying it against something.
#: Default-deny: :class:`TestEveryDeclarationHasAReader` requires the terms on
#: disk to equal these keys exactly, so a ninth term cannot arrive unread.
ONTOLOGY_TERM_READERS: dict[str, str] = {
    "RecognitionMatrix": "check_conformance: dims, num_obs x num_states, factor count",
    "TransitionMatrix": "check_conformance: dims, squareness, control axis",
    "Prior": "check_conformance: dims; _check_connection_graph: seeds a HiddenState",
    "LogPreferenceVector": "check_conformance: dims == (num_obs,)",
    "PolicyPrior": "check_conformance via MATRIX_ROLES, against the live agent.E",
    "HiddenState": "_check_vector_declarations; _check_emission_factor_count",
    "NextHiddenState": "_check_vector_declarations",
    "Observation": "_check_vector_declarations: dims == (num_obs,)",
    "PolicyVector": "_check_vector_declarations; _check_connection_graph",
    "Time": "_check_time_index: the Time block must name it",
}

#: ``ModelParameters`` key family -> what reads it. ``num_hidden_states_f0`` and
#: ``num_hidden_states_f1`` collapse to one family, since they differ only in
#: which factor they size.
MODEL_PARAMETER_READERS: dict[str, str] = {
    "num_hidden_states": "check_conformance: A/B/D sizing, live state dimension",
    "num_hidden_states_f<N>": "_factor_sizes -> _check_multifactor_conformance",
    "num_obs": "check_conformance: A's observation axis, C, o_t",
    "num_factors": "check_conformance: the single/multi dispatch and len(agent.B)",
    "num_modalities": "check_conformance: len(agent.A)",
    "num_controls": "check_conformance: B's control axis, E's size, the graph",
    "gamma": "_check_precision, against the live agent.gamma",
    # Deliberately NOT read off a live agent: the same emission, transition and
    # prior serve both combination rules and only the code combining them
    # differs, so the rule is invisible in the matrices. That is precisely why it
    # belongs in the spec — decision 47 showed the rule, not the insertion site,
    # decides whether an inter-level message can help at all.
    "combination_rule": "_check_combination_rule: membership in COMBINATION_RULES",
}

#: ``Learned(...)`` argument -> what resolves or checks it.
LEARNED_ARGUMENT_READERS: dict[str, str] = {
    "generator": "provenance_violations: must exist in the repository",
    "artifact": "provenance_violations: a repository path, or a builder symbol",
    "key": "artifact_key_violations: a top-level key of the vendored JSON",
    "normalization": "_check_learned_arguments: an axis claim on the live array",
    "estimator": "_check_learned_arguments: positivity, for unqualified Laplace",
    "n_symbols": "spec_agent._resolve_learned: exact role/provenance key selects the alphabet",
}

#: ``Learned(...)`` arguments that resist a check rather than merely lacking
#: one. Each reason is stated here because this is where a reader looking for
#: the missing check will arrive.
UNCHECKABLE_LEARNED_ARGUMENTS: dict[str, str] = {
    "split": (
        "names which split of the vendored artifact the counts came from. The "
        "live matrix is the OUTPUT of that selection and retains no trace of "
        "which rows produced it; recovering the split would mean re-running the "
        "estimator, which is tests/test_provenance_checksums.py's job."
    ),
    "alignment": (
        "names how letters were aligned to phones before counting. Alignment is "
        "consumed by the generator; the emission it produces is a distribution "
        "that any alignment could in principle have produced."
    ),
    "statistic": (
        "names which statistic of the artifact was taken (first phone of each "
        "word). Same reason as 'split': the live vector is the result, and "
        "several statistics can yield the same distribution."
    ),
}

#: Section -> what interprets its contents beyond retaining the raw text.
SECTION_READERS: dict[str, str] = {
    "ModelName": "parse_gnn_string: non-empty; asserted to start with 'AILM'",
    "StateSpaceBlock": "check_conformance and _check_declaration_coherence",
    "Connections": "_check_connection_graph and _check_emission_factor_count",
    "InitialParameterization": "_check_structural_form and _check_learned_arguments",
    "ActInfOntologyAnnotation": "MATRIX_ROLES, _VECTOR_ROLE_SIZES, _check_time_index",
    "ModelParameters": "check_conformance and _check_precision",
    "Time": "_check_time_index reads the 'Time=<variable>' line",
}

#: Sections retained verbatim and interpreted nowhere, with the reason.
UNCHECKABLE_SECTIONS: dict[str, str] = {
    "GNNSection": (
        "an identifier for the block. There is no corresponding attribute on a "
        "pymdp agent, and comparing it to the filename would pin a convention "
        "rather than the model."
    ),
    "GNNVersionAndFlags": (
        "the notation version the file is written in. It constrains the PARSER, "
        "not the model; the upstream interoperability class below is what "
        "actually demonstrates the version claim."
    ),
    "ModelAnnotation": (
        "prose. Its 'claim:' labels are the one machine-readable part, and they "
        "are backed by data/claims.json and enforced by "
        "tests/test_claims_discipline.py rather than here."
    ),
    "Equations": (
        "the recursion in mathematical notation. Interpreting it would mean "
        "writing a second implementation of the model, and when the two "
        "disagreed there would be no way to say which was wrong."
    ),
    "Footer": (
        "prose naming the implementing modules. The one checkable part is those "
        "paths, resolved against the repository by footer_path_violations below."
    ),
    "Signature": (
        "'Unsigned' on every AILM spec. Provenance is git history plus the "
        "checksums in docs/provenance/, so there is no signature to verify."
    ),
}

#: Structural form -> the property :func:`_check_structural_form` asserts.
STRUCTURAL_FORM_READERS: dict[str, str] = {
    "Identity": "max |M - I| <= 1e-5",
    "Uniform": "every entry within 1e-5 of 1/n",
    "Zeros": "every entry within 1e-5 of 0",
    "Learned": "NOT the identity when square, plus the normalization and estimator claims",
}

#: ``normalization=`` prefixes the checker understands. Any other prefix is a
#: violation rather than a skip, so this set cannot silently fall behind.
KNOWN_NORMALIZATION_PREFIXES: tuple[str, ...] = ("column-stochastic", "sums-to-one")

#: The exact ``(spec, variable)`` pairs whose ``Learned`` form declares NO
#: ``normalization=``, so nothing checks which axis they sum over. A RESIDUAL
#: GAP, pinned rather than hidden: it cannot grow without failing here, and
#: closing it means adding the argument to those two specs, not loosening this.
LEARNED_FORMS_WITHOUT_NORMALIZATION: frozenset[tuple[str, str]] = frozenset(
    {("L2a_phoneme_sequence", "D"), ("L2b_grapheme_to_phoneme", "D")}
)

#: Paths cited in prose are recognized by their leading directory.
_FOOTER_PATH_RE = re.compile(r"\b(?:src|scripts|tests|docs|vendor)/[\w./-]*\.\w+")

#: Non-vacuity floors for the repository-resolution scans below. A scan that
#: resolves nothing passes every "no violations" assertion.
MIN_ARTIFACT_KEYS = 5
MIN_FOOTER_PATHS = 16

#: An upstream GNN checkout, if one is available.
#: Where to look for a GeneralizedNotationNotation checkout: the checkout's
#: sibling namespace first, then a plain sibling directory. Both share the
#: checkout's lifetime. No temporary directory is searched: a scratch probe was
#: once the only default, and cleanup reaped it — the parser sources vanished
#: while the package directory and its ``__pycache__`` stayed, so thirty-three
#: interoperability tests stopped running and reported "no checkout" at a path
#: where a checkout plainly was. Evidence for a manuscript claim should not live
#: where the operating system tidies up.
_UPSTREAM_CANDIDATES = upstream_candidates(PROJECT_ROOT)
UPSTREAM_ROOT = selected_upstream_root(PROJECT_ROOT)
UPSTREAM_SRC = UPSTREAM_ROOT / "src"


def spec_path(name: str) -> Path:
    """Absolute path of a shipped spec, resolved through the model registry."""
    from ailm.gnn import spec_path as registry_spec_path

    return registry_spec_path(name)


def spec_text(name: str) -> str:
    """Raw text of a shipped spec."""
    return spec_path(name).read_text(encoding="utf-8")


def mutate(text: str, old: str, new: str) -> str:
    """Replace ``old`` with ``new`` once, asserting the anchor actually matched.

    ``str.replace`` on a missing anchor returns the input unchanged, which would
    turn every mutation-based positive control below into a test of the pristine
    spec — a silent fake pass. This helper makes that impossible.
    """
    assert old in text, (
        f"mutation anchor {old!r} not present in the spec — the replacement would "
        f"be a silent no-op and the control would test the unmodified spec."
    )
    return text.replace(old, new, 1)


def discovered_model_specs() -> tuple[str, ...]:
    """Every canonical level spec sitting under ``gnn/models/`` right now.

    Read off the filesystem rather than from any registry constant, because the
    thing being guarded against is a spec that exists on disk and is named
    nowhere in Python.
    """
    directory = GNN_DIR / "models"
    return tuple(
        sorted(path.name[: -len(SPEC_SUFFIX)] for path in directory.glob(f"*{SPEC_SUFFIX}"))
    )


def parameter_family(key: str) -> str:
    """Collapse ``num_hidden_states_f3`` to the family ``num_hidden_states_f<N>``.

    Per-factor keys differ only in which factor they size, so they are one
    declaration kind. Without this the inventory would have to list a key per
    factor and would go stale the moment a model gained one.
    """
    return re.sub(r"_f\d+$", "_f<N>", key)


def provenance_entries(args: str) -> dict[str, str]:
    """The ``generator=``/``artifact=`` pair out of a ``Learned(...)`` argument list."""
    return {
        key: value
        for key, value in parse_learned_arguments(args).items()
        if key in {"generator", "artifact"}
    }


def artifact_key_violations(spec: GnnSpec) -> tuple[list[str], int]:
    """Resolve every ``Learned(key=...)`` inside the artifact that entry names.

    The defect this catches: a spec naming a slice of a vendored artifact that
    the artifact does not contain. ``key=phone_unigrams`` is a load-bearing
    declaration — it says WHICH of the ten payloads in ``cmudict_counts.json``
    parameterizes the slot — and it was parsed into
    ``GnnSpec.parameterization`` and read by nothing, so a renamed payload would
    leave the spec citing a key that resolves to nothing while still reading as
    reproducible.

    Only artifacts that name a repository path to a JSON file can be resolved;
    a ``key=`` alongside a builder symbol has no document to look in and is
    counted as unresolved rather than waved through.

    Returns:
        ``(violations, keys_resolved)``. The count lets the caller assert the
        scan was not vacuous, since a spec with no ``key=`` resolves nothing and
        would otherwise report clean.
    """
    violations: list[str] = []
    resolved = 0
    for variable, (form, args) in spec.parameterization.items():
        if form != "Learned":
            continue
        entries = parse_learned_arguments(args)
        named = entries.get("key")
        if named is None:
            continue
        artifact = entries.get("artifact", "")
        target = PROJECT_ROOT / artifact
        if "/" not in artifact or target.suffix != ".json" or not target.is_file():
            violations.append(
                f"{variable} cites key={named} but artifact={artifact!r} is not a "
                f"readable JSON path, so the key names a slice of nothing."
            )
            continue
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or named not in payload:
            violations.append(
                f"{variable} cites key={named}, which is not a top-level key of "
                f"{artifact}. Its keys are {sorted(payload)[:12] if isinstance(payload, dict) else type(payload)}."
            )
            continue
        resolved += 1
    return violations, resolved


def footer_path_violations(spec: GnnSpec) -> tuple[list[str], int]:
    """Resolve every repository path the ``Footer`` cites.

    The defect this catches: a Footer that names the module implementing the
    level, the module having since moved. ``Footer`` is otherwise prose and
    interpreted nowhere — this is the only part of it a checker can hold to
    account, and it is worth holding, because a spec whose Footer points at a
    deleted file reads as a map of the code while being one.

    Returns:
        ``(violations, paths_resolved)``; the count guards against a regex that
        silently stops matching.
    """
    violations: list[str] = []
    resolved = 0
    for cited in _FOOTER_PATH_RE.findall(spec.sections.get("Footer", "")):
        if (PROJECT_ROOT / cited).is_file():
            resolved += 1
            continue
        violations.append(f"Footer cites {cited}, which does not exist at {PROJECT_ROOT / cited}.")
    return violations, resolved


def provenance_violations(spec: GnnSpec) -> tuple[list[str], int]:
    """Resolve every ``Learned`` form's cited provenance against the repository.

    The defect this catches: a spec that reads as reproducible while citing a
    generator or builder that no longer exists.

    Two shapes of provenance ship, and both are resolved rather than one being
    waved through. An ``artifact=`` naming a repository path (it contains a
    ``/``) must be a file. An ``artifact=`` naming a bare symbol — how the
    structural forms cite their builder, e.g. ``build_position_transition`` —
    must be defined in the generator the same entry names, which is the stronger
    check of the two: renaming that function breaks it.

    Existence only. Whether the generator rebuilds the artifact byte-identically
    is ``tests/test_provenance_checksums.py``'s job.

    Returns:
        ``(violations, targets_resolved)``. The count lets the caller assert the
        scan was not vacuous, since a spec with no ``Learned`` form resolves
        nothing and would otherwise report clean.
    """
    violations: list[str] = []
    resolved = 0
    for variable, (form, args) in spec.parameterization.items():
        if form != "Learned":
            continue
        cited = provenance_entries(args)
        if set(cited) != {"generator", "artifact"}:
            violations.append(
                f"{variable} cites {sorted(cited)}; a Learned form must name both "
                f"a generator and an artifact."
            )
            continue

        generator = PROJECT_ROOT / cited["generator"]
        if not generator.is_file():
            violations.append(
                f"{variable} cites generator={cited['generator']} which does not "
                f"exist at {generator}."
            )
            continue
        resolved += 1

        artifact = cited["artifact"]
        if "/" in artifact:
            target = PROJECT_ROOT / artifact
            if not target.is_file():
                violations.append(
                    f"{variable} cites artifact={artifact} which does not exist at {target}."
                )
                continue
        elif f"def {artifact}(" not in generator.read_text(encoding="utf-8"):
            violations.append(
                f"{variable} cites artifact={artifact}, which names no path and is "
                f"not defined in {cited['generator']}. A builder renamed out from "
                f"under the spec must fail here."
            )
            continue
        resolved += 1
    return violations, resolved


def policy_prior_violations(spec: GnnSpec, agent: Any) -> list[str]:
    """Check a declared ``E`` against the live agent's policy prior.

    The defect this catches: a policy prior whose declaration nothing reads.
    ``describe_shapes`` once extracted A, B, C and D only, so ``check_conformance``
    left a declared ``E`` unverified — and ``L1g_letter_generation`` is the one
    shipped model that has one. Its whole argument for annotating the Brown
    bigram ``PolicyPrior`` instead of ``TransitionMatrix`` is that the bigram is
    installed into ``E`` per generation step while the construction-time ``E`` is
    uniform. Unchecked, that distinction rests on prose.

    ``describe_shapes`` now reports ``E`` and ``MATRIX_ROLES`` binds
    ``PolicyPrior`` to it, so the checker itself sizes the slot and enforces its
    structural form. This function is kept as a SECOND, independent reader: it
    reaches the live ``.E`` directly rather than through the description, so a
    regression in ``describe_shapes`` — the exact failure that opened the gap —
    still fails here.

    Returns:
        Every violation found; empty when the spec declares no ``E``.
    """
    if "E" not in spec.parameterization:
        return []
    form, args = spec.parameterization["E"]
    if form != "Uniform":
        # Refuse rather than pass: an unrecognized form must be loud, because
        # silently returning [] would mark the slot verified when it is not.
        return [
            f"E is declared {form}({args}); this check only knows Uniform, so the "
            f"slot would go unverified. Teach it the new form."
        ]
    live = getattr(agent, "E", None)
    if live is None:
        return [f"the spec declares E={form}({args}) but the live agent has no E."]
    vector = np.asarray(live, dtype=np.float64).ravel()
    size = int(args)
    if vector.size != size:
        return [f"E: declared Uniform({size}) but the live E has {vector.size} entries."]
    if not np.allclose(vector, 1.0 / size, atol=1e-6):
        return [
            f"E: declared Uniform({size}) but the live policy prior is not uniform "
            f"(min {vector.min():.6g}, max {vector.max():.6g}). The learned habit "
            f"belongs in E per generation step, not baked in at construction."
        ]
    return []


# ── LIVE AGENT FIXTURES (real pymdp, real vendored artifacts, no mocks) ───────


@pytest.fixture(scope="module")
def specs() -> dict[str, GnnSpec]:
    """Every shipped canonical spec, parsed, keyed by name.

    Parsed once: the declaration inventory below reads all nine repeatedly, and
    a per-test reparse would make the audit the slowest thing in the file for no
    added coverage.
    """
    return {name: parse_gnn_spec(spec_path(name)) for name in SPEC_NAMES}


@pytest.fixture(scope="module")
def brown_bigram() -> np.ndarray:
    """The real Brown letter-bigram matrix, shape (1, 30, 30)."""
    from ailm.levels.letter import build_letter_transition, load_brown_counts

    return build_letter_transition(load_brown_counts())


@pytest.fixture(scope="module")
def letter_agent(brown_bigram):
    """The shipped L1 letter agent, built from vendor/brown_counts.json."""
    from ailm.levels.letter import build_letter_agent

    return build_letter_agent(brown_bigram)


@pytest.fixture(scope="module")
def phoneme_counts() -> dict:
    """The vendored CMUdict statistics payload."""
    from ailm.levels.phoneme import load_phoneme_counts

    return load_phoneme_counts()


@pytest.fixture(scope="module")
def phoneme_agent(phoneme_counts):
    """The shipped L2A phoneme sequence agent."""
    from ailm.levels.phoneme import (
        build_phone_transition,
        build_phone_unigram,
        build_phoneme_agent,
    )

    return build_phoneme_agent(
        build_phone_transition(phoneme_counts["phone_bigrams"]),
        build_phone_unigram(phoneme_counts["phone_unigrams"]),
    )


@pytest.fixture(scope="module")
def g2p_train(phoneme_counts) -> list:
    """The TRAIN split of the vendored G2P pairs."""
    from ailm.levels.phoneme import g2p_split

    train, _test = g2p_split(phoneme_counts)
    return train


@pytest.fixture(scope="module")
def g2p_agent(g2p_train):
    """The shipped L2B G2P agent: 30 letters observed, 39 phones hidden."""
    from ailm.levels.phoneme import (
        build_g2p_agent,
        build_g2p_emission,
        build_g2p_start,
        build_g2p_transition,
    )

    return build_g2p_agent(
        build_g2p_emission(g2p_train),
        build_g2p_transition(g2p_train),
        build_g2p_start(g2p_train),
    )


@pytest.fixture(scope="module")
def generative_letter_agent(brown_bigram):
    """The shipped L1g agent: same 30 letters, but a CONTROLLABLE transition.

    Built exactly as ``src/ailm/generation.py`` builds it, from the same Brown
    bigram the recognition level puts in ``B``. Here that table is the policy
    prior instead, which is the difference the L1g spec exists to pin.
    """
    from ailm.generation import build_generative_letter_agent

    return build_generative_letter_agent(np.asarray(brown_bigram)[0])


@pytest.fixture(scope="module")
def word_agent():
    """The shipped L3 agent: two hidden factors, word identity and position."""
    from ailm.levels.word import build_word_agent, load_word_artifact

    return build_word_agent(load_word_artifact())


@pytest.fixture(scope="module")
def case_agent():
    """The shipped L4 agent, on the DERIVED population.

    The population matters: the L4 spec's ``num_obs`` is the UPOS alphabet of
    the derived population, and the gold population's alphabet is a different
    size. Building from ``"gold"`` here would test a different model than the
    one the spec describes.
    """
    from ailm.levels.case import build_case_agent, load_case_artifact

    return build_case_agent(load_case_artifact("derived"))


@pytest.fixture(scope="module")
def clause_agent():
    """The shipped L5 agent over the joint (sentence type x case role) space."""
    from ailm.levels.clause import build_clause_agent, load_clause_artifact

    return build_clause_agent(load_clause_artifact())


@pytest.fixture(scope="module")
def widened_clause_agent():
    """The L5w agent over the joint (sentence type x widened symbol) space.

    The separately recorded widened-observation arm of decision 112. It is
    bound here for the same reason every other spec is: ``gnn/models/`` is the
    source of truth and this suite's coverage is default-deny, so a spec on
    disk with no live agent fails rather than going unchecked.
    """
    from ailm.levels.clause import build_type_models
    from ailm.levels.widened import (
        N_WIDENED_SYMBOLS,
        build_widened_agent,
        load_widened_clause_artifact,
    )

    artifact = load_widened_clause_artifact()
    return build_widened_agent(build_type_models(artifact.train, n_symbols=N_WIDENED_SYMBOLS))


@pytest.fixture(scope="module")
def narrative_agent():
    """The shipped L6 agent: three document phases over two sentence types."""
    from ailm.levels.narrative import build_narrative_agent, load_narrative_artifact

    return build_narrative_agent(load_narrative_artifact())


@pytest.fixture(scope="module")
def live_agents(
    letter_agent,
    generative_letter_agent,
    phoneme_agent,
    g2p_agent,
    word_agent,
    case_agent,
    clause_agent,
    widened_clause_agent,
    narrative_agent,
) -> dict:
    """Spec name -> the live agent that spec must describe.

    One entry per shipped spec. ``TestEveryShippedSpecIsCovered`` asserts the
    keys equal :data:`SPEC_NAMES`, so an agent added without a spec — or a spec
    added without an agent — fails rather than quietly narrowing the suite.
    """
    return {
        "L1_letter": letter_agent,
        "L1g_letter_generation": generative_letter_agent,
        "L2a_phoneme_sequence": phoneme_agent,
        "L2b_grapheme_to_phoneme": g2p_agent,
        "L3_word": word_agent,
        "L4_case": case_agent,
        "L5_clause": clause_agent,
        "L5w_clause_widened": widened_clause_agent,
        "L6_narrative": narrative_agent,
    }


# ── DEFAULT-DENY COVERAGE ────────────────────────────────────────────────────


class TestEveryShippedSpecIsCovered:
    """The covered set must equal the directory, minus named exclusions.

    The defect this catches: a ninth spec lands in ``gnn/models/``, nobody adds
    it to ``SPEC_NAMES``, and the suite goes on passing while the new model is
    described by a document no test ever compares to an agent. That is exactly
    how the five-spec gap this class closes came about.
    """

    def test_discovery_finds_the_shipped_specs(self):
        """Non-vacuity. A scan that finds nothing satisfies every set test below."""
        discovered = discovered_model_specs()
        assert (GNN_DIR / "models").is_dir(), f"{GNN_DIR / 'models'} is missing"
        assert len(discovered) >= MIN_SHIPPED_SPECS, (
            f"only {len(discovered)} spec(s) found under {GNN_DIR / 'models'}: "
            f"{list(discovered)}. At least {MIN_SHIPPED_SPECS} ship, so the scan "
            f"is looking in the wrong place and every comparison below is vacuous."
        )

    def test_coverage_equals_the_shipped_directory(self):
        """Every spec on disk is bound to a live agent, with no exception list.

        This carried an ``UNCHECKED_SPECS`` escape hatch mapping a spec to the
        slot it failed on. It reached empty once all nine specs conformed and
        was removed: a dormant exclusion mechanism is an invitation to use it,
        and "every shipped spec is checked" is a stronger statement than "every
        shipped spec is checked except the ones listed here".
        """
        discovered = set(discovered_model_specs())
        covered = set(SPEC_NAMES)

        uncovered = discovered - covered
        assert uncovered == set(), (
            f"spec(s) {sorted(uncovered)} ship under gnn/models/ but are bound to "
            f"no live agent here. Add them to SPEC_NAMES. If one genuinely does "
            f"not conform, that is a finding about the repository to fix or to "
            f"record in docs/decisions.md, not an entry in an exception list."
        )
        phantom = covered - discovered
        assert phantom == set(), (
            f"SPEC_NAMES lists {sorted(phantom)}, which is not on disk. A name "
            f"that resolves to nothing would parametrize an empty check."
        )

    def test_the_registry_and_the_directory_agree(self):
        """A spec filed under the wrong kind would escape the scan above.

        ``discover_models`` parses every filename and rejects one sitting in a
        directory its name does not match, so running both and comparing turns a
        misfiled spec into a failure here rather than a silent omission.
        """
        from ailm.gnn import discover_models

        registry = {
            str(model_id)
            for model_id, path in discover_models().items()
            if path.parent.name == "models"
        }
        assert registry == set(discovered_model_specs())

    def test_every_covered_spec_has_a_live_agent(self, live_agents):
        """The fixture map and the covered set must be the same set.

        A dict cannot hold a duplicate key but a tuple can, so a name repeated
        in ``SPEC_NAMES`` would still compare equal as a set while inflating the
        parametrized counts. That is checked on the tuple itself.
        """
        assert set(live_agents) == set(SPEC_NAMES)
        assert len(SPEC_NAMES) == len(set(SPEC_NAMES)), f"duplicate in SPEC_NAMES: {SPEC_NAMES}"
        assert len(live_agents) == len(SPEC_NAMES)

    @pytest.mark.parametrize("name", SPEC_NAMES)
    def test_each_spec_is_single_or_multifactor_but_not_both(self, name):
        """The two assertion families must partition the specs, not overlap.

        A spec in neither tuple would be parametrized by no structural test at
        all while still counting as covered.
        """
        single = name in SINGLE_FACTOR_SPEC_NAMES
        multi = name in MULTIFACTOR_SPEC_NAMES
        assert single != multi, (
            f"{name} is in {'both' if single and multi else 'neither'} of "
            f"SINGLE_FACTOR_SPEC_NAMES and MULTIFACTOR_SPEC_NAMES."
        )
        # The listing must match what the spec itself declares, or a multi-factor
        # model could be checked by the single-factor assertions and pass them by
        # accident of naming its first factor 'B'.
        spec = parse_gnn_spec(spec_path(name))
        declared = spec.parameters.get("num_factors")
        if single:
            assert declared == 1, f"{name} is listed single-factor but declares {declared}"
        else:
            assert isinstance(declared, int) and declared > 1, (
                f"{name} is listed multi-factor but declares num_factors={declared}"
            )


class TestEveryDeclarationHasAReader:
    """Default-deny over declaration KINDS, not only over spec files.

    ``TestEveryShippedSpecIsCovered`` guarantees every spec is compared to a live
    agent. That is necessary and not sufficient: a spec can be bound to an agent
    and still declare things the comparison never looks at. It is exactly how
    L1g's policy prior went unverified — ``describe_shapes`` read A/B/C/D, so a
    whole declaration kind was parsed, stored, and compared to nothing while
    every set-level coverage test stayed green.

    Each test below reads a declaration kind off disk and requires the observed
    set to equal the inventory of what reads it, union the inventory of what
    genuinely cannot be read. Both directions matter: an unnamed declaration is
    a silent gap, and a named one that no longer appears is an inventory entry
    describing nothing.
    """

    def test_the_inventory_scan_is_not_vacuous(self, specs):
        """A scan over nothing satisfies every set comparison in this class."""
        assert len(specs) >= MIN_SHIPPED_SPECS, sorted(specs)
        assert all(spec.variables and spec.ontology for spec in specs.values())
        assert sum(len(spec.parameterization) for spec in specs.values()) >= 4 * len(specs)

    def test_every_section_on_disk_is_read_or_named_unreadable(self, specs):
        observed = {section for spec in specs.values() for section in spec.sections}
        inventoried = set(SECTION_READERS) | set(UNCHECKABLE_SECTIONS)
        assert observed - inventoried == set(), (
            f"section(s) {sorted(observed - inventoried)} appear in a shipped spec "
            f"and are named in neither SECTION_READERS nor UNCHECKABLE_SECTIONS. "
            f"Either something reads them — say what — or nothing can, and the "
            f"reason belongs in UNCHECKABLE_SECTIONS where a reader will meet it."
        )
        assert inventoried - observed == set(), (
            f"{sorted(inventoried - observed)} is inventoried but appears in no "
            f"shipped spec, so the entry describes nothing."
        )
        assert set(SECTION_READERS) & set(UNCHECKABLE_SECTIONS) == set(), (
            "a section cannot be both read and unreadable"
        )
        assert observed <= set(GNN_SECTIONS)

    def test_every_ontology_term_on_disk_has_a_reader(self, specs):
        """No ontology term may be merely descriptive.

        A term with no reader is a role annotation that pins nothing, which is
        the whole thing these specs exist to do.
        """
        observed = {term for spec in specs.values() for term in spec.ontology.values()}
        assert observed == set(ONTOLOGY_TERM_READERS), (
            f"ontology terms on disk: {sorted(observed)}; inventoried: "
            f"{sorted(ONTOLOGY_TERM_READERS)}. A term appearing in a spec and not "
            f"here is annotated onto a variable nothing compares to a live model."
        )

    def test_the_role_tables_cover_every_ontology_term(self, specs):
        """The inventory must match the code, not merely be self-consistent.

        Naming a reader in a docstring proves nothing; this asserts that every
        term is in one of the two tables ``ailm.gnn`` actually dispatches on, so
        the inventory above cannot drift into fiction.
        """
        from ailm.gnn import _VECTOR_ROLE_SIZES

        observed = {term for spec in specs.values() for term in spec.ontology.values()}
        dispatched = set(MATRIX_ROLES) | set(_VECTOR_ROLE_SIZES) | {"Time"}
        assert observed <= dispatched, (
            f"term(s) {sorted(observed - dispatched)} are declared in a spec but "
            f"appear in neither MATRIX_ROLES nor _VECTOR_ROLE_SIZES, so nothing "
            f"in ailm.gnn dispatches on them."
        )
        assert "PolicyPrior" in MATRIX_ROLES, (
            "PolicyPrior must bind to a live slot; it is the gap this audit began "
            "from, and demoting it to descriptive would reopen it."
        )

    def test_every_model_parameter_on_disk_has_a_reader(self, specs):
        observed = {parameter_family(key) for spec in specs.values() for key in spec.parameters}
        assert observed == set(MODEL_PARAMETER_READERS), (
            f"ModelParameters keys on disk: {sorted(observed)}; inventoried: "
            f"{sorted(MODEL_PARAMETER_READERS)}. A key nothing reads is a number "
            f"in the spec that the model is free to contradict."
        )

    def test_every_learned_argument_on_disk_is_read_or_named_unreadable(self, specs):
        observed: set[str] = set()
        for spec in specs.values():
            for form, args in spec.parameterization.values():
                if form == "Learned":
                    observed |= set(parse_learned_arguments(args))
        inventoried = set(LEARNED_ARGUMENT_READERS) | set(UNCHECKABLE_LEARNED_ARGUMENTS)
        assert observed == inventoried, (
            f"Learned arguments on disk: {sorted(observed)}; inventoried: "
            f"{sorted(inventoried)}. Add a reader, or add the reason it resists "
            f"one to UNCHECKABLE_LEARNED_ARGUMENTS."
        )
        assert set(LEARNED_ARGUMENT_READERS) & set(UNCHECKABLE_LEARNED_ARGUMENTS) == set()
        for reason in UNCHECKABLE_LEARNED_ARGUMENTS.values():
            assert len(reason) > 60, f"a one-word excuse is not a reason: {reason!r}"

    def test_every_structural_form_on_disk_has_a_reader(self, specs):
        observed = {
            form for spec in specs.values() for form, _args in spec.parameterization.values()
        }
        assert observed == set(STRUCTURAL_FORM_READERS), (
            f"structural forms on disk: {sorted(observed)}; inventoried: "
            f"{sorted(STRUCTURAL_FORM_READERS)}."
        )

    def test_every_declared_normalization_prefix_is_understood(self, specs):
        """An axis claim the checker does not know is a slot left unverified.

        ``_check_learned_arguments`` reports an unknown prefix rather than
        skipping it, so this cannot fail silently — but failing here names the
        spec and the axis, which is the more useful place to find out.
        """
        seen = 0
        for name, spec in specs.items():
            for variable, (form, args) in spec.parameterization.items():
                if form != "Learned":
                    continue
                declared = parse_learned_arguments(args).get("normalization")
                if declared is None:
                    continue
                seen += 1
                assert declared.startswith(KNOWN_NORMALIZATION_PREFIXES), (
                    f"{name} {variable}: normalization={declared} starts with none "
                    f"of {KNOWN_NORMALIZATION_PREFIXES}, so its axis is unverified."
                )
        assert seen >= 15, f"only {seen} normalization claim(s) scanned; near-vacuous"

    def test_the_learned_forms_declaring_no_normalization_are_exactly_the_known_two(self, specs):
        """Pin a RESIDUAL GAP so it cannot spread.

        Two shipped ``Learned`` forms declare no ``normalization=``, so nothing
        checks which axis they sum over. That is a real hole, recorded rather
        than hidden: a third one fails here. Closing it means adding the
        argument to those two specs — which live outside this file — not
        widening this set.
        """
        missing = {
            (name, variable)
            for name, spec in specs.items()
            for variable, (form, args) in spec.parameterization.items()
            if form == "Learned" and "normalization" not in parse_learned_arguments(args)
        }
        assert missing == LEARNED_FORMS_WITHOUT_NORMALIZATION, (
            f"Learned forms without a normalization claim: {sorted(missing)}; "
            f"expected exactly {sorted(LEARNED_FORMS_WITHOUT_NORMALIZATION)}. A new "
            f"one is a new unverified axis; a missing one means the gap closed and "
            f"the entry should go."
        )

    def test_every_artifact_key_resolves_in_the_artifact(self, specs):
        """``key=`` names a slice of a vendored payload; it must be there."""
        total = 0
        for name, spec in specs.items():
            violations, resolved = artifact_key_violations(spec)
            assert violations == [], f"{name}:\n" + "\n".join(violations)
            total += resolved
        assert total >= MIN_ARTIFACT_KEYS, (
            f"only {total} artifact key(s) resolved across {len(specs)} specs; the "
            f"scan is looking in the wrong place and reports clean for free."
        )

    def test_every_footer_path_exists(self, specs):
        """The one checkable part of an otherwise-prose section."""
        total = 0
        for name, spec in specs.items():
            violations, resolved = footer_path_violations(spec)
            assert violations == [], f"{name}:\n" + "\n".join(violations)
            assert resolved >= 2, (
                f"{name}: only {resolved} path(s) matched in the Footer. Every "
                f"shipped Footer names at least an implementation and a test, so "
                f"the regex has stopped matching."
            )
            total += resolved
        assert total >= MIN_FOOTER_PATHS, total


# ── PARSING ───────────────────────────────────────────────────────────────────


class TestSpecFilesParse:
    """Every shipped spec must parse into a fully-populated GnnSpec."""

    def test_every_named_spec_exists(self):
        """Guard against a vacuous parametrized suite over a missing directory."""
        assert GNN_DIR.is_dir(), f"{GNN_DIR} is missing"
        assert SPEC_NAMES, "SPEC_NAMES is empty; every parametrized test below is a no-op"
        missing = [name for name in SPEC_NAMES if not spec_path(name).is_file()]
        assert missing == [], f"missing spec file(s): {missing}"

    @pytest.mark.parametrize("name", SPEC_NAMES)
    def test_spec_parses(self, name):
        spec = parse_gnn_spec(spec_path(name))
        assert isinstance(spec, GnnSpec)
        assert spec.model_name.startswith("AILM"), spec.model_name
        assert spec.origin.endswith(f"{name}{SPEC_SUFFIX}")

    @pytest.mark.parametrize("name", SINGLE_FACTOR_SPEC_NAMES)
    def test_variables_are_populated_with_dims_and_dtypes(self, name):
        spec = parse_gnn_spec(spec_path(name))
        for required in ("A", "B", "C", "D", "s_t", "o_t", "t"):
            assert required in spec.variables, f"{name}: {required} not declared"
        dims, dtype = spec.variables["A"]
        assert all(isinstance(d, int) and d > 0 for d in dims), dims
        assert dtype == "float"
        assert spec.variables["o_t"][1] == "int"

    @pytest.mark.parametrize("name", MULTIFACTOR_SPEC_NAMES)
    def test_multifactor_variables_are_populated_per_factor(self, name):
        """The same requirement, stated per factor rather than relaxed.

        A multi-factor model has no bare ``B`` or ``D``: it carries one of each
        per factor, bound by an ``_f<N>`` suffix. Accepting a bare ``B`` here
        would let a two-factor model pass while its second factor went unpinned.
        """
        spec = parse_gnn_spec(spec_path(name))
        for required in ("A", "B_f0", "B_f1", "C", "D_f0", "D_f1", "s_word", "s_pos", "o_t", "t"):
            assert required in spec.variables, f"{name}: {required} not declared"
        assert "B" not in spec.variables and "D" not in spec.variables, (
            f"{name}: an unsuffixed B or D in a multi-factor model binds to no "
            f"factor, so nothing would notice it sizing the wrong one."
        )
        dims, dtype = spec.variables["A"]
        assert all(isinstance(d, int) and d > 0 for d in dims), dims
        assert dtype == "float"
        assert spec.variables["o_t"][1] == "int"

    @pytest.mark.parametrize("name", SINGLE_FACTOR_SPEC_NAMES)
    def test_ontology_names_every_matrix_role(self, name):
        """Without these three annotations the spec pins nothing."""
        spec = parse_gnn_spec(spec_path(name))
        assert spec.ontology["A"] == "RecognitionMatrix"
        assert spec.ontology["B"] == "TransitionMatrix"
        assert spec.ontology["D"] == "Prior"
        # Bijection: nothing declared is left un-annotated.
        assert set(spec.ontology) == set(spec.variables)

    @pytest.mark.parametrize("name", MULTIFACTOR_SPEC_NAMES)
    def test_multifactor_ontology_names_a_role_per_factor(self, name):
        spec = parse_gnn_spec(spec_path(name))
        assert spec.variables_with_role("RecognitionMatrix") == ["A"]
        assert spec.variables_with_role("TransitionMatrix") == ["B_f0", "B_f1"]
        assert spec.variables_with_role("Prior") == ["D_f0", "D_f1"]
        assert set(spec.ontology) == set(spec.variables)

    @pytest.mark.parametrize("name", SINGLE_FACTOR_SPEC_NAMES)
    def test_model_parameters_match_the_shipped_alphabet_sizes(self, name):
        spec = parse_gnn_spec(spec_path(name))
        expected_obs, expected_states = EXPECTED_PARAMETERS[name]
        assert spec.parameters["num_obs"] == expected_obs
        assert spec.parameters["num_hidden_states"] == expected_states
        assert spec.parameters["num_factors"] == 1
        assert spec.parameters["num_modalities"] == 1

    @pytest.mark.parametrize("name", MULTIFACTOR_SPEC_NAMES)
    def test_multifactor_parameters_size_every_factor(self, name):
        spec = parse_gnn_spec(spec_path(name))
        expected_obs, expected_sizes = EXPECTED_FACTOR_PARAMETERS[name]
        assert spec.parameters["num_obs"] == expected_obs
        assert spec.parameters["num_factors"] == len(expected_sizes)
        assert spec.parameters["num_modalities"] == 1
        for index, size in enumerate(expected_sizes):
            assert spec.parameters[f"num_hidden_states_f{index}"] == size
        assert "num_hidden_states" not in spec.parameters, (
            f"{name}: a scalar num_hidden_states alongside the per-factor keys "
            f"is a second source of truth for the state space."
        )

    @pytest.mark.parametrize("name", SPEC_NAMES)
    def test_connections_and_raw_sections_are_captured(self, name):
        spec = parse_gnn_spec(spec_path(name))
        assert spec.connections, f"{name}: no connections parsed"
        directed = [c for c in spec.connections if c.directed]
        undirected = [c for c in spec.connections if not c.directed]
        assert directed and undirected, "expected both '>' and '-' connections"
        assert next(c.source for c in spec.connections if c.directed) == EXPECTED_PRIOR_SEED[name]
        # Nothing the checker declines to interpret is discarded.
        for section in ("ModelAnnotation", "Time", "Equations", "Footer"):
            assert section in spec.sections
            assert spec.sections[section].strip()

    @pytest.mark.parametrize("name", SPEC_NAMES)
    def test_matrices_are_not_inlined(self, name):
        """A 30x30 table in the spec would be a second source of truth.

        Every parameterized slot is checked, not only A/B/D, so a per-factor
        transition or a policy prior cannot slip in as a literal.
        """
        spec = parse_gnn_spec(spec_path(name))
        block = spec.sections["InitialParameterization"]
        assert "{" not in block, (
            f"{name}: InitialParameterization inlines a matrix literal. Learned "
            f"entries are a derived artifact — name the generator instead."
        )
        assert spec.parameterization, f"{name}: nothing parameterized"
        for matrix, (form, args) in spec.parameterization.items():
            assert form in {"Identity", "Uniform", "Zeros", "Learned"}, f"{matrix}: {form}"
            if form == "Learned":
                assert "generator=" in args and "artifact=" in args, f"{matrix}: {args}"
        assert len(spec.parameterization) >= 4, (
            f"{name}: only {len(spec.parameterization)} slot(s) parameterized; "
            f"A, B, C and D are the minimum, so this loop is nearly vacuous."
        )

    @pytest.mark.parametrize("name", SPEC_NAMES)
    def test_learned_provenance_targets_exist(self, name):
        spec = parse_gnn_spec(spec_path(name))
        violations, resolved = provenance_violations(spec)
        assert violations == [], f"{name}:\n" + "\n".join(violations)
        assert resolved >= 2, (
            f"{name}: only {resolved} provenance target(s) resolved — every "
            f"Learned form must name both a generator and an artifact, or this "
            f"assertion is vacuous."
        )

    def test_g2p_recognition_matrix_spans_two_alphabets(self):
        """The crux of L2B: A is a real emission over a DIFFERENT alphabet."""
        spec = parse_gnn_spec(spec_path("L2b_grapheme_to_phoneme"))
        assert spec.parameters["num_obs"] != spec.parameters["num_hidden_states"]
        assert spec.variable_dims("A") == (30, 39)
        assert spec.parameterization["A"][0] == "Learned", (
            "the G2P emission is learned, not the identity — that is what makes "
            "this level the first genuine emission in AILM"
        )
        assert spec.role_of("A") == "RecognitionMatrix"
        assert spec.variables_with_role("RecognitionMatrix") == ["A"]

    @pytest.mark.parametrize("name", ("L1_letter", "L1g_letter_generation", "L2a_phoneme_sequence"))
    def test_directly_observed_levels_declare_an_identity_emission(self, name):
        spec = parse_gnn_spec(spec_path(name))
        form, args = spec.parameterization["A"]
        assert form == "Identity"
        assert int(args) == spec.parameters["num_hidden_states"]

    def test_the_generative_level_puts_the_learned_table_in_the_policy_prior(self):
        """L1g's whole reason to exist as a separate spec, asserted.

        Same 30-state space as L1_letter, but a controllable B. Under a
        controllable transition the Brown bigram is what the agent would do
        absent a goal — a habit — so it belongs in E, and B is the structural
        "action k emits letter k". A spec annotating the bigram TransitionMatrix
        here would describe a model that cannot choose what to emit.
        """
        spec = parse_gnn_spec(spec_path("L1g_letter_generation"))
        assert spec.parameters["num_controls"] == 30, (
            "a single control would leave the policy machinery unexercised, "
            "which is exactly what L1_letter already does"
        )
        assert spec.variable_dims("B") == (30, 30, 30)
        assert spec.role_of("E") == "PolicyPrior"
        assert spec.parameterization["E"] == ("Uniform", "30"), (
            "E is uniform AT CONSTRUCTION; the generation loop installs the "
            "learned habit column per step (docs/decisions.md entry 20)."
        )
        assert spec.variables_with_role("TransitionMatrix") == ["B"]

    def test_the_word_level_puts_the_learned_table_in_the_prior(self):
        """L3's relocation of the mis-wiring risk, asserted rather than narrated.

        Below this level the learned sequence statistics parameterize B. Here
        both transitions are structural and the single learned probability is
        the word unigram prior D_f0 — so a wrong D_f0 at L3 is what a wrong B
        was at L1.
        """
        spec = parse_gnn_spec(spec_path("L3_word"))
        assert spec.parameterization["B_f0"] == ("Identity", "2000")
        assert spec.parameterization["D_f0"][0] == "Learned"
        assert "vendor/brown_word_counts.json" in spec.parameterization["D_f0"][1]
        assert spec.variable_dims("A") == (
            WORD_N_OBS,
            len(STAGE3_WORD_ARTIFACT.vocab),
            STAGE3_WORD_ARTIFACT.n_positions,
        ), (
            "the emission depends on BOTH factors; a two-axis A would be a "
            "different model in which position does not select the letter"
        )


class TestMalformedSpecsRaise:
    """Malformed input must raise GnnSpecError, never yield a partial spec."""

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(GnnSpecError, match="cannot read GNN spec"):
            parse_gnn_spec(tmp_path / "nope.md")

    def test_content_without_sections_raises(self):
        with pytest.raises(GnnSpecError, match="not a GNN spec"):
            parse_gnn_string("# Just a title\n\nsome prose\n", origin="fake.md")

    def test_bad_dims_bracket_raises_and_names_the_line(self):
        text = mutate(
            spec_text("L1_letter"),
            "A[30,30,type=float]",
            "A[30,30,type=float",
        )
        with pytest.raises(GnnSpecError) as excinfo:
            parse_gnn_string(text, origin="mutated.md")
        message = str(excinfo.value)
        assert "malformed variable declaration" in message
        assert "A[30,30,type=float" in message, "error must quote the bad line"
        assert "mutated.md:" in message, "error must name file and line number"

    def test_non_integer_dimension_raises(self):
        text = mutate(
            spec_text("L1_letter"),
            "B[30,30,1,type=float]",
            "B[30,n_states,1,type=float]",
        )
        with pytest.raises(GnnSpecError, match="not a positive integer"):
            parse_gnn_string(text, origin="mutated.md")

    def test_zero_dimension_raises(self):
        text = mutate(spec_text("L1_letter"), "C[30,1,type=float]", "C[0,1,type=float]")
        with pytest.raises(GnnSpecError, match="not a positive integer"):
            parse_gnn_string(text, origin="mutated.md")

    def test_unknown_section_raises(self):
        text = mutate(
            spec_text("L1_letter"),
            "## ModelParameters",
            "## ModelParamaters",
        )
        with pytest.raises(GnnSpecError) as excinfo:
            parse_gnn_string(text, origin="mutated.md")
        assert "unknown GNN section 'ModelParamaters'" in str(excinfo.value)

    def test_duplicate_section_raises(self):
        text = spec_text("L1_letter") + "\n## Footer\n\nsecond footer\n"
        with pytest.raises(GnnSpecError, match="duplicate GNN section 'Footer'"):
            parse_gnn_string(text, origin="mutated.md")

    def test_duplicate_variable_raises(self):
        text = mutate(
            spec_text("L1_letter"),
            "A[30,30,type=float]",
            "A[30,30,type=float]\nA[30,30,type=float]",
        )
        with pytest.raises(GnnSpecError, match="duplicate variable 'A'"):
            parse_gnn_string(text, origin="mutated.md")

    def test_ontology_naming_an_undeclared_variable_raises(self):
        text = mutate(
            spec_text("L1_letter"),
            "s_t=HiddenState",
            "s_t=HiddenState\nE=Habit",
        )
        with pytest.raises(GnnSpecError) as excinfo:
            parse_gnn_string(text, origin="mutated.md")
        assert "annotates undeclared variable 'E'" in str(excinfo.value)

    def test_declared_but_unannotated_variable_raises(self):
        """An un-annotated slot is a slot whose role nothing pins."""
        text = mutate(spec_text("L1_letter"), "\nC=LogPreferenceVector", "")
        with pytest.raises(GnnSpecError) as excinfo:
            parse_gnn_string(text, origin="mutated.md")
        assert "un-annotated variable(s) ['C']" in str(excinfo.value)

    def test_missing_ontology_section_raises(self):
        """Without ActInfOntologyAnnotation there is nothing to pin."""
        raw = spec_text("L1_letter")
        head, marker, _tail = raw.partition("## ActInfOntologyAnnotation")
        assert marker, "anchor section absent — the truncation would be a no-op"
        with pytest.raises(GnnSpecError) as excinfo:
            parse_gnn_string(head, origin="mutated.md")
        message = str(excinfo.value)
        assert "['ActInfOntologyAnnotation']" in message
        assert "nothing to" in message

    def test_inlined_matrix_literal_raises(self):
        """Rejecting an inlined matrix is what keeps one source of truth."""
        text = mutate(
            spec_text("L1_letter"),
            "A=Identity(30)",
            "A={\n  (1.0, 0.0),\n  (0.0, 1.0)\n}",
        )
        with pytest.raises(GnnSpecError) as excinfo:
            parse_gnn_string(text, origin="mutated.md")
        assert "only structural forms" in str(excinfo.value)

    def test_unknown_structural_form_raises(self):
        text = mutate(spec_text("L1_letter"), "A=Identity(30)", "A=Diagonalish(30)")
        with pytest.raises(GnnSpecError, match="unknown structural form"):
            parse_gnn_string(text, origin="mutated.md")

    def test_sized_form_with_non_integer_argument_raises(self):
        text = mutate(spec_text("L1_letter"), "A=Identity(30)", "A=Identity(thirty)")
        with pytest.raises(GnnSpecError, match="single positive integer size"):
            parse_gnn_string(text, origin="mutated.md")

    def test_learned_form_without_provenance_raises(self):
        text = mutate(spec_text("L1_letter"), "A=Identity(30)", "A=Learned()")
        with pytest.raises(GnnSpecError, match="must name its provenance"):
            parse_gnn_string(text, origin="mutated.md")

    def test_parameterization_for_undeclared_variable_raises(self):
        text = mutate(
            spec_text("L1_letter"),
            "A=Identity(30)",
            "A=Identity(30)\nE=Uniform(30)",
        )
        with pytest.raises(GnnSpecError, match="undeclared variable 'E'"):
            parse_gnn_string(text, origin="mutated.md")

    def test_connection_to_undeclared_variable_raises(self):
        text = mutate(spec_text("L1_letter"), "D>s_t", "D>s_unknown")
        with pytest.raises(GnnSpecError, match="undeclared variable 's_unknown'"):
            parse_gnn_string(text, origin="mutated.md")

    def test_connection_without_operator_raises(self):
        text = mutate(spec_text("L1_letter"), "D>s_t", "D s_t")
        with pytest.raises(GnnSpecError, match="no connection operator"):
            parse_gnn_string(text, origin="mutated.md")

    def test_malformed_model_parameter_raises(self):
        text = mutate(
            spec_text("L1_letter"),
            "num_hidden_states: 30",
            "num_hidden_states 30",
        )
        with pytest.raises(GnnSpecError, match="malformed ModelParameters entry"):
            parse_gnn_string(text, origin="mutated.md")

    def test_empty_model_name_raises(self):
        text = mutate(
            spec_text("L1_letter"),
            "## ModelName\n\nAILM Letter Level\n",
            "## ModelName\n\n",
        )
        with pytest.raises(GnnSpecError, match="ModelName section is empty"):
            parse_gnn_string(text, origin="mutated.md")

    def test_known_section_set_is_the_canonical_thirteen(self):
        """Pins the accepted vocabulary so a silent widening is visible."""
        assert GNN_SECTIONS[0] == "GNNSection"
        assert GNN_SECTIONS[-1] == "Signature"
        assert "ActInfOntologyAnnotation" in GNN_SECTIONS
        assert len(GNN_SECTIONS) == len(set(GNN_SECTIONS))


# ── CONFORMANCE AGAINST LIVE AGENTS ──────────────────────────────────────────


class TestLiveAgentsConform:
    """The shipped specs must describe the agents pymdp actually builds."""

    @pytest.mark.parametrize("name", SPEC_NAMES)
    def test_live_agent_conforms_to_its_spec(self, name, live_agents):
        spec = parse_gnn_spec(spec_path(name))
        described = describe_shapes(live_agents[name])
        violations = check_conformance(spec, described)
        assert violations == [], f"{name}:\n" + "\n".join(violations)
        assert_conforms(spec, described)

    @pytest.mark.parametrize("name", SINGLE_FACTOR_SPEC_NAMES)
    def test_describe_shapes_reads_the_real_object(self, name, live_agents):
        agent = live_agents[name]
        described = describe_shapes(agent)
        assert described.a_shape == tuple(agent.A[0].shape)
        assert described.b_shape == tuple(agent.B[0].shape)
        assert described.c_shape == tuple(agent.C[0].shape)
        assert described.d_shape == tuple(agent.D[0].shape)
        assert described.n_modalities == 1
        assert described.n_factors == 1
        for key in ("a_matrix", "b_matrix", "c_vector", "d_vector"):
            assert isinstance(getattr(described, key), np.ndarray)

    @pytest.mark.parametrize("name", SPEC_NAMES)
    def test_describe_shapes_reports_the_policy_slots(self, name, live_agents):
        """E and gamma must travel with the description, for every agent.

        The defect this catches is the audit's origin: ``describe_shapes``
        reported A/B/C/D and stopped, so a declared ``E`` or ``gamma`` reached
        ``check_conformance`` as None and the slot went unverified. pymdp always
        constructs both — a uniform E over the policy space and a broadcast
        gamma — so there is no agent for which reporting None is correct.
        """
        agent = live_agents[name]
        described = describe_shapes(agent)
        assert described.e_shape == tuple(np.asarray(agent.E).shape)
        assert isinstance(described.e_vector, np.ndarray)
        assert described.gamma is not None
        assert np.asarray(described.gamma).size >= 1

    @pytest.mark.parametrize("name", MULTIFACTOR_SPEC_NAMES)
    def test_describe_shapes_reports_every_factor(self, name, live_agents):
        """Reporting only slot 0 would leave the second factor unverified.

        A two-factor agent carries one B and one D per factor. If
        ``describe_shapes`` collapsed those to the first entry, a spec could
        pass conformance with its second factor entirely unchecked.
        """
        agent = live_agents[name]
        described = describe_shapes(agent)
        expected_obs, expected_sizes = EXPECTED_FACTOR_PARAMETERS[name]
        assert described.n_modalities == 1
        assert described.n_factors == len(expected_sizes)
        assert len(described.b_matrices) == len(expected_sizes)
        assert len(described.d_vectors) == len(expected_sizes)
        for index, size in enumerate(expected_sizes):
            assert tuple(agent.B[index].shape) == (1, size, size, 1)
            assert tuple(agent.D[index].shape) == (1, size)
        assert tuple(agent.A[0].shape) == (1, expected_obs, *expected_sizes), (
            "the emission's factor order is load-bearing: transposing two "
            "factors of different sizes is a different model, not a reshape"
        )

    def test_g2p_live_emission_is_obs_by_states(self, g2p_agent):
        """The wiring the L2B spec exists to pin, read off the live agent."""
        described = describe_shapes(g2p_agent)
        assert described.a_shape == (1, 30, 39)
        assert described.b_shape == (1, 39, 39, 1)
        assert described.d_shape == (1, 39)

    def test_the_generative_transition_really_carries_thirty_actions(self, generative_letter_agent):
        """L1g is the only level with a control axis worth comparing.

        ``_canonical_dims`` drops trailing size-1 axes, which is what lets every
        recognition level's single trivial control compare equal to a spec that
        writes ``B[n,n,1]``. That reduction would also hide a controllable model
        collapsing to one action, so the axis is read here explicitly.
        """
        described = describe_shapes(generative_letter_agent)
        assert described.b_shape == (1, 30, 30, 30)
        spec = parse_gnn_spec(spec_path("L1g_letter_generation"))
        assert described.b_shape[-1] == spec.parameters["num_controls"]

    @pytest.mark.parametrize("name", SPEC_NAMES)
    def test_declared_policy_prior_matches_the_live_agent(self, name, live_agents):
        """Close the one slot ``describe_shapes`` does not reach.

        Only L1g declares an ``E``; for every other spec this returns empty,
        which is why ``test_the_policy_prior_check_can_fail`` exists to prove
        the check is not vacuous where it does apply.
        """
        spec = parse_gnn_spec(spec_path(name))
        violations = policy_prior_violations(spec, live_agents[name])
        assert violations == [], f"{name}:\n" + "\n".join(violations)

    def test_exactly_one_shipped_spec_declares_a_policy_prior(self):
        """Non-vacuity for the check above: it must apply somewhere."""
        declaring = [
            name for name in SPEC_NAMES if "E" in parse_gnn_spec(spec_path(name)).parameterization
        ]
        assert declaring == ["L1g_letter_generation"], declaring

    @pytest.mark.parametrize("name", SPEC_NAMES)
    def test_load_spec_resolves_from_the_package(self, name, live_agents):
        """The documented one-liner must work, not merely read well.

        gnn/README.md advertises
        ``assert_conforms(load_spec(name), describe_shapes(agent))``; this runs
        it, including the path resolution, which is independent of the working
        directory.
        """
        from ailm.gnn import load_spec, spec_dir

        assert spec_dir() == GNN_DIR
        assert_conforms(load_spec(name), describe_shapes(live_agents[name]))
        assert load_spec(f"{name}{SPEC_SUFFIX}").model_name == load_spec(name).model_name

    def test_describe_shapes_refuses_a_model_without_a(self):
        class Headless:
            A: list = []
            B: list = [np.eye(2)]
            D: list = [np.ones(2) / 2]

        with pytest.raises(ValueError, match="agent has no A to describe"):
            describe_shapes(Headless())


class TestTypedLiveRecord:
    """The typed record is the call surface; its edges must fail loudly.

    ``check_conformance`` once took roughly twenty-five untyped positional
    parameters that every caller splatted from ``describe_shapes``. The record
    replaces that surface. Three properties keep the replacement honest: the
    record form and the keyword form agree on every shipped verdict, a MISSING
    field defaults to None (unverified) exactly as an absent keyword did, and
    an EXTRA field is a TypeError rather than a silently dropped value. The
    keyword adapter is retained only while the connected execution inventory
    binds this module's bytes; every suite constructs the record directly.
    """

    def test_the_record_form_and_the_keyword_form_agree(self, letter_agent):
        """Same verdict either way, on the real shipped agent."""
        spec = parse_gnn_spec(spec_path("L1_letter"))
        described = describe_shapes(letter_agent)
        assert check_conformance(spec, described) == check_conformance(spec, **described.as_dict())

    def test_a_missing_field_defaults_to_none_not_an_error(self):
        """Partial splats (test_gnn_edge_paths relies on this) stay legal."""
        spec = parse_gnn_spec(spec_path("L1_letter"))
        partial = LiveConformanceInputs(a_shape=(1, 30, 30), b_shape=(1, 30, 30, 1))
        violations = check_conformance(spec, partial)
        assert any("no live array was supplied" in v or "unverified" in v for v in violations)

    def test_an_extra_field_is_rejected_not_dropped(self):
        """A typo'd or renamed field fails when the record is built, never silently."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            LiveConformanceInputs(a_shap=(1, 30, 30))  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            check_conformance(parse_gnn_spec(spec_path("L1_letter")), a_shap=(1, 30, 30))

    def test_the_record_and_the_keyword_form_cannot_be_mixed(self, letter_agent):
        described = describe_shapes(letter_agent)
        with pytest.raises(TypeError, match="one form, not both"):
            check_conformance(parse_gnn_spec(spec_path("L1_letter")), described, gamma=16.0)

    def test_describe_shapes_returns_the_typed_record(self, letter_agent):
        described = describe_shapes(letter_agent)
        assert isinstance(described, LiveConformanceInputs)
        assert described.a_shape == tuple(letter_agent.A[0].shape)
        assert described.as_dict()["a_shape"] == described.a_shape


# ── POSITIVE CONTROLS: the checker must be able to fail ──────────────────────


class TestPositiveControls:
    """Genuinely mis-wired models and self-contradictory specs must be caught.

    Every control asserts on the violation TEXT. Asserting only that something
    was raised would be satisfied by an unrelated failure — an import error, a
    typo in a fixture — and the control would stop measuring the checker.
    """

    def test_swapped_a_and_b_fails_the_letter_spec(self, brown_bigram):
        """THE historical bug: bigram in A, identity in B.

        Constructed for real with pymdp. Note that this agent has *identical
        matrix shapes* to the correct wiring — (1, 30, 30) and (1, 30, 30, 1) —
        so a purely dimensional checker would pass it. It is caught by the
        structural forms the spec declares.
        """
        import jax.numpy as jnp
        from pymdp.agent import Agent

        from ailm.levels.letter import N_SYMBOLS

        swapped = Agent(
            A=[brown_bigram],  # learned bigram in the emission slot — WRONG
            B=[jnp.eye(N_SYMBOLS)[jnp.newaxis, :, :, jnp.newaxis]],  # inert
            C=[jnp.zeros((1, N_SYMBOLS))],
            D=[jnp.ones((1, N_SYMBOLS)) / N_SYMBOLS],
            num_controls=[1],
            batch_size=1,
        )
        described = describe_shapes(swapped)
        correct_shapes = {
            key: getattr(described, key) for key in ("a_shape", "b_shape", "c_shape", "d_shape")
        }
        assert correct_shapes == {
            "a_shape": (1, 30, 30),
            "b_shape": (1, 30, 30, 1),
            "c_shape": (1, 30),
            "d_shape": (1, 30),
        }, "the swap must be shape-identical, otherwise this control is trivial"

        spec = parse_gnn_spec(spec_path("L1_letter"))
        violations = check_conformance(spec, described)
        assert violations, "the A/B swap was NOT caught — the checker is vacuous"

        joined = "\n".join(violations)
        assert "A: declared Identity(30)" in joined
        assert "is NOT the identity" in joined
        assert "PRECEDING symbol" in joined
        assert "B: declared Learned(" in joined
        assert "IS the identity" in joined
        assert "sequence statistics belong in B" in joined

        with pytest.raises(GnnConformanceError) as excinfo:
            assert_conforms(spec, described)
        message = str(excinfo.value)
        assert "conformance violation(s)" in message
        assert "never edit the spec just to make this pass" in message

    def test_shape_only_check_cannot_catch_the_swap(self, brown_bigram):
        """NEGATIVE CONTROL on the control above, and on this file's honesty.

        Demonstrates the claim the previous test rests on: with the structural
        arrays withheld, the swapped agent is indistinguishable from the correct
        one. This is why the checker reads values for declared structural forms,
        and why SPEC.md says shapes alone are insufficient at this level.
        """
        import jax.numpy as jnp
        from pymdp.agent import Agent

        from ailm.levels.letter import N_SYMBOLS

        swapped = Agent(
            A=[brown_bigram],
            B=[jnp.eye(N_SYMBOLS)[jnp.newaxis, :, :, jnp.newaxis]],
            C=[jnp.zeros((1, N_SYMBOLS))],
            D=[jnp.ones((1, N_SYMBOLS)) / N_SYMBOLS],
            num_controls=[1],
            batch_size=1,
        )
        described = describe_shapes(swapped)
        spec = parse_gnn_spec(spec_path("L1_letter"))
        shapes_only = check_conformance(
            spec,
            LiveConformanceInputs(
                a_shape=described.a_shape,
                b_shape=described.b_shape,
                c_shape=described.c_shape,
                d_shape=described.d_shape,
                n_modalities=described.n_modalities,
                n_factors=described.n_factors,
            ),
        )
        assert shapes_only == [], (
            "shape-only conformance unexpectedly rejected the swap; if this "
            "starts failing, the claim that shapes are insufficient here is "
            "stale and SPEC.md must be corrected."
        )

    def test_transposed_g2p_emission_fails_the_g2p_spec(self, g2p_train, brown_bigram):
        """The inverted map: phones observed, letters hidden.

        A real, pymdp-valid agent whose emission is the transpose renormalized
        to (39, 30). The spec declares RecognitionMatrix as num_obs by
        num_hidden_states = 30 by 39, so the direction of the map is a named
        violation rather than a slightly different accuracy.
        """
        import jax.numpy as jnp
        from pymdp.agent import Agent

        from ailm.levels.phoneme import build_g2p_emission

        emission = np.asarray(build_g2p_emission(g2p_train))[0]  # (30, 39)
        inverted = emission.T.copy()  # (39, 30) — phone given letter
        inverted = inverted / inverted.sum(axis=0, keepdims=True)

        agent = Agent(
            A=[jnp.array(inverted[np.newaxis, :, :])],
            B=[brown_bigram[:, :, :, jnp.newaxis]],  # letters as the hidden chain
            C=[jnp.zeros((1, 39))],
            D=[jnp.ones((1, 30)) / 30.0],
            num_controls=[1],
            batch_size=1,
        )
        described = describe_shapes(agent)
        assert described.a_shape == (1, 39, 30), "control must really be transposed"

        spec = parse_gnn_spec(spec_path("L2b_grapheme_to_phoneme"))
        violations = check_conformance(spec, described)
        assert violations, "the inverted G2P direction was NOT caught"

        joined = "\n".join(violations)
        assert "A (RecognitionMatrix): declared dims (30, 39)" in joined
        assert "canonical (39, 30)" in joined
        assert "!= (num_obs, num_hidden_states) = (30, 39)" in joined
        assert "the transpose is a different model" in joined
        assert "D (Prior): live canonical dims (30,) != (num_hidden_states,)" in joined

    def test_spec_whose_parameters_contradict_its_state_space_fails(self, g2p_agent):
        """A self-inconsistent spec must be rejected against a correct agent."""
        text = mutate(
            spec_text("L2b_grapheme_to_phoneme"), "num_hidden_states: 39", "num_hidden_states: 41"
        )
        spec = parse_gnn_string(text, origin="mutated_g2p.md")
        violations = check_conformance(spec, describe_shapes(g2p_agent))
        assert violations, "a spec contradicting itself was accepted"
        joined = "\n".join(violations)
        assert "The spec contradicts itself" in joined
        assert "ModelParameters num_obs=30, num_hidden_states=41 require" in joined

    def test_mutating_a_declared_dimension_fails_against_the_real_agent(self, phoneme_agent):
        """The spec must track the code: change a declared dim and it breaks."""
        text = spec_text("L2a_phoneme_sequence")
        text = mutate(text, "B[39,39,1,type=float]", "B[38,38,1,type=float]")
        text = mutate(text, "num_hidden_states: 39", "num_hidden_states: 38")
        text = mutate(text, "A[39,39,type=float]", "A[39,38,type=float]")
        text = mutate(text, "D[39,1,type=float]", "D[38,1,type=float]")
        text = mutate(text, "A=Identity(39)", "A=Identity(38)")
        # The belief vectors are part of the state space and have to shrink with
        # it. Leaving them at 39 is what this mutation originally did, and the
        # spec was then internally INCONSISTENT while asserting it was not —
        # nothing read s_t's declared size, so nothing said so. Shrinking them
        # here is what makes the assertion below true rather than merely stated.
        text = mutate(text, "s_t[39,1,type=float]", "s_t[38,1,type=float]")
        text = mutate(text, "s_prime[39,1,type=float]", "s_prime[38,1,type=float]")
        # num_obs stays at 39, so the mutated spec is internally CONSISTENT: the
        # violations below can only come from comparing it to the live agent.
        spec = parse_gnn_string(text, origin="mutated_phoneme.md")

        violations = check_conformance(spec, describe_shapes(phoneme_agent))
        assert violations, "a shrunken state space was accepted"
        joined = "\n".join(violations)
        assert "The spec contradicts itself" not in joined, (
            "the mutated spec must stay internally consistent, so that these "
            "violations are attributable to the live agent comparison alone:\n" + joined
        )
        assert "B (TransitionMatrix): declared dims (38, 38)" in joined
        assert "canonical (39, 39)" in joined
        assert "live state dimension 39 != ModelParameters num_hidden_states=38" in joined

    def test_withholding_a_shape_is_reported_not_ignored(self, letter_agent):
        """An unverified slot must be a violation, not a silent pass."""
        described = describe_shapes(letter_agent)
        described = replace(described, d_shape=None)
        described = replace(described, d_vector=None)
        spec = parse_gnn_spec(spec_path("L1_letter"))
        violations = check_conformance(spec, described)
        assert any("no d_shape was supplied" in v for v in violations), violations
        assert any("the slot is unverified" in v for v in violations), violations

    def test_empty_model_parameters_is_reported(self, letter_agent):
        """A spec that declares no sizes cannot pin any slot."""
        raw = spec_text("L1_letter")
        before, marker, rest = raw.partition("## ModelParameters")
        assert marker, "anchor section absent"
        _dropped, footer_marker, footer = rest.partition("## Footer")
        assert footer_marker, "Footer anchor absent"
        spec = parse_gnn_string(f"{before}{marker}\n\n{footer_marker}{footer}", origin="mutated.md")
        assert spec.parameters == {}
        violations = check_conformance(spec, describe_shapes(letter_agent))
        joined = "\n".join(violations)
        assert "missing 'num_hidden_states'" in joined
        assert "missing 'num_obs'" in joined

    def test_ambiguous_role_annotation_is_reported(self, letter_agent):
        """Two variables claiming the same role pin nothing."""
        text = mutate(
            spec_text("L1_letter"),
            "C=LogPreferenceVector",
            "C=TransitionMatrix",
        )
        spec = parse_gnn_string(text, origin="mutated.md")
        violations = check_conformance(spec, describe_shapes(letter_agent))
        assert any("annotated on 2 variables" in v for v in violations), violations

    def test_non_positive_model_parameter_is_reported(self, letter_agent):
        text = mutate(spec_text("L1_letter"), "num_obs: 30", "num_obs: not_a_number")
        spec = parse_gnn_string(text, origin="mutated.md")
        violations = check_conformance(spec, describe_shapes(letter_agent))
        assert any("is not a positive integer" in v for v in violations), violations

    def test_declared_factor_count_is_checked(self, letter_agent):
        text = mutate(spec_text("L1_letter"), "num_factors: 1", "num_factors: 2")
        spec = parse_gnn_string(text, origin="mutated.md")
        violations = check_conformance(spec, describe_shapes(letter_agent))
        assert any("num_factors=2 but the live model has 1" in v for v in violations)

    def test_uniform_prior_declaration_is_checked(self, letter_agent, phoneme_counts):
        """L1 declares D uniform; a learned prior in that slot must be caught."""
        from ailm.levels.phoneme import build_phone_unigram

        spec = parse_gnn_spec(spec_path("L1_letter"))
        described = describe_shapes(letter_agent)
        # A non-uniform 30-vector: the first 30 phone unigram weights, renormalized.
        skewed = np.asarray(build_phone_unigram(phoneme_counts["phone_unigrams"]))
        skewed = skewed.ravel()[:30]
        described = replace(described, d_vector=(skewed / skewed.sum())[np.newaxis, :])
        violations = check_conformance(spec, described)
        assert any("declared Uniform(30)" in v and "not uniform" in v for v in violations), (
            violations
        )

    def test_zeros_preference_declaration_is_checked(self, letter_agent):
        spec = parse_gnn_spec(spec_path("L1_letter"))
        described = describe_shapes(letter_agent)
        described = replace(described, c_vector=np.full((1, 30), 0.5))
        violations = check_conformance(spec, described)
        assert any("declared Zeros(30)" in v for v in violations), violations

    def test_identity_declaration_with_wrong_size_is_reported(self, letter_agent):
        spec = parse_gnn_spec(spec_path("L1_letter"))
        described = describe_shapes(letter_agent)
        described = replace(described, a_matrix=np.eye(29)[np.newaxis, :, :])
        violations = check_conformance(spec, described)
        assert any("cannot be a 30x30 identity" in v for v in violations), violations

    def test_the_policy_prior_check_can_fail(self, brown_bigram):
        """POSITIVE CONTROL on :func:`policy_prior_violations`.

        The defect: the learned habit baked into ``E`` at construction. That is
        the model L1g's spec says it is NOT — the habit is installed per step by
        the generation loop, and a construction-time habit would make the "same
        table, different role" argument for annotating it PolicyPrior false.

        A stand-in object rather than a real Agent because the guard reads
        ``.E`` and nothing else, and pymdp Agents are frozen: there is no way to
        build the mis-wired one through the shipped builder, which takes no E.
        """

        class WithPolicyPrior:
            def __init__(self, prior):
                self.E = prior

        spec = parse_gnn_spec(spec_path("L1g_letter_generation"))
        habit = np.asarray(brown_bigram)[0][:, 0]
        baked_in = (habit / habit.sum())[np.newaxis, :]
        assert baked_in.shape == (1, 30), "the control must be right-sized to be non-trivial"

        violations = policy_prior_violations(spec, WithPolicyPrior(baked_in))
        assert any("declared Uniform(30)" in v and "not uniform" in v for v in violations), (
            violations
        )

        wrong_size = policy_prior_violations(spec, WithPolicyPrior(np.full((1, 29), 1.0 / 29)))
        assert any("the live E has 29 entries" in v for v in wrong_size), wrong_size

        absent = policy_prior_violations(spec, WithPolicyPrior(None))
        assert any("the live agent has no E" in v for v in absent), absent

    def test_provenance_citing_a_missing_generator_is_reported(self):
        """POSITIVE CONTROL: a deleted generator must not read as reproducible."""
        text = mutate(
            spec_text("L1_letter"),
            "generator=scripts/generate_counts.py",
            "generator=scripts/deleted_generator.py",
        )
        spec = parse_gnn_string(text, origin="mutated.md")
        violations, resolved = provenance_violations(spec)
        assert any("does not exist at" in v for v in violations), violations
        assert resolved == 0, resolved

    def test_provenance_citing_an_undefined_builder_is_reported(self):
        """POSITIVE CONTROL for the symbol branch: a renamed builder must fail.

        Four shipped specs cite a builder function rather than a file — the
        structural forms have no artifact on disk — so that branch carries a
        third of the provenance checking and needs its own control.
        """
        text = mutate(
            spec_text("L1g_letter_generation"),
            "artifact=build_generative_letter_agent",
            "artifact=build_renamed_letter_agent",
        )
        spec = parse_gnn_string(text, origin="mutated.md")
        violations, _resolved = provenance_violations(spec)
        assert any("is not defined in src/ailm/generation.py" in v for v in violations), violations


class TestNewlyReadDeclarationsCanFail:
    """One positive control per declaration the audit brought under a reader.

    Every one asserts on the reported violation TEXT. A control asserting only
    that the list is non-empty would be satisfied by any pre-existing violation
    the mutation happened to drag along, and would stop measuring the check it
    was written for the moment something else started failing.

    Each mutation is applied to a real shipped spec and compared against a real
    live agent, so a control cannot pass by describing a model nobody builds.
    """

    def test_a_resized_belief_vector_is_reported(self, letter_agent):
        """``s_t`` was declared and compared to nothing.

        Resizing the level in ModelParameters and in A/B/D while leaving ``s_t``
        behind produced a spec that conformed while contradicting itself about
        the size of its own state space.
        """
        text = mutate(spec_text("L1_letter"), "s_t[30,1,type=float]", "s_t[7,1,type=float]")
        spec = parse_gnn_string(text, origin="mutated.md")
        violations = check_conformance(spec, describe_shapes(letter_agent))
        joined = "\n".join(violations)
        assert "s_t (HiddenState): declared size 7" in joined, violations
        assert "num_hidden_states=30" in joined, violations

    def test_an_observation_sized_by_nothing_is_reported(self, letter_agent):
        text = mutate(spec_text("L1_letter"), "o_t[30,1,type=int]", "o_t[29,1,type=int]")
        spec = parse_gnn_string(text, origin="mutated.md")
        joined = "\n".join(check_conformance(spec, describe_shapes(letter_agent)))
        assert "o_t (Observation): declared dims (29,)" in joined, joined
        assert "num_obs=30 requires (30,)" in joined, joined

    def test_a_wrong_dtype_is_reported(self, letter_agent):
        """``type=`` was parsed onto every variable and read by nothing."""
        text = mutate(spec_text("L1_letter"), "o_t[30,1,type=int]", "o_t[30,1,type=float]")
        spec = parse_gnn_string(text, origin="mutated.md")
        joined = "\n".join(check_conformance(spec, describe_shapes(letter_agent)))
        assert "o_t (Observation) is declared type=float" in joined, joined
        assert "is an index, which is int" in joined, joined

    def test_an_unwired_variable_is_reported(self, letter_agent):
        """A declared, annotated variable the graph never mentions."""
        text = mutate(spec_text("L1_letter"), "\nC-o_t", "")
        spec = parse_gnn_string(text, origin="mutated.md")
        joined = "\n".join(check_conformance(spec, describe_shapes(letter_agent)))
        assert "['C'] are declared and annotated but appear in no" in joined, joined
        assert "Connections line" in joined, joined

    def test_a_prior_that_seeds_nothing_is_reported(self, letter_agent):
        """``D>s_prime`` seeds the NEXT state, which no inference step reads."""
        text = mutate(spec_text("L1_letter"), "D>s_t", "D>s_prime")
        spec = parse_gnn_string(text, origin="mutated.md")
        joined = "\n".join(check_conformance(spec, describe_shapes(letter_agent)))
        assert "D (Prior) is directed into ['s_prime']" in joined, joined
        assert "seeds the hidden state or it seeds nothing" in joined, joined

    def test_a_time_index_naming_nothing_is_reported(self, letter_agent):
        """The Time block was retained verbatim and interpreted nowhere."""
        text = mutate(spec_text("L1_letter"), "Time=t\n", "Time=tau\n")
        spec = parse_gnn_string(text, origin="mutated.md")
        joined = "\n".join(check_conformance(spec, describe_shapes(letter_agent)))
        assert "Time=tau names no variable declared in StateSpaceBlock" in joined, joined

    def test_a_matrix_normalized_over_the_wrong_axis_is_reported(self, letter_agent, brown_bigram):
        """``normalization=`` was declared on 18 forms and read by none.

        The transpose of a column-stochastic bigram is row-stochastic. pymdp
        accepts either, and the shapes are identical — which is the same blind
        spot the A/B swap exploited, one declaration further in.
        """
        described = describe_shapes(letter_agent)
        transposed = np.transpose(np.asarray(brown_bigram, dtype=np.float64), (0, 2, 1))
        column_sums = transposed[0].sum(axis=0)
        assert not np.allclose(column_sums, 1.0, atol=1e-4), (
            "the transposed bigram is column-stochastic too, so this control "
            "would pass against the correct declaration and measure nothing"
        )
        described = replace(described, b_matrix=transposed[..., np.newaxis])

        spec = parse_gnn_spec(spec_path("L1_letter"))
        joined = "\n".join(check_conformance(spec, described))
        assert "B: declared normalization=column-stochastic-over-next-letter" in joined, joined
        assert "axis-0 sums deviate from 1" in joined, joined
        assert "normalized over the wrong axis" in joined, joined

    def test_an_unsmoothed_matrix_under_a_smoothed_estimator_is_reported(self, letter_agent):
        """``estimator=add-1-Laplace`` promises no zero entry, and promised it alone.

        The zeroed column is renormalized so that ONLY the positivity claim can
        fire: a control that also broke the axis sum would be satisfied by the
        normalization violation and would not measure this check.
        """
        described = describe_shapes(letter_agent)
        table = np.asarray(described.b_matrix, dtype=np.float64).reshape(30, 30).copy()
        table[5, 7] = 0.0
        table[:, 7] /= table[:, 7].sum()
        assert np.allclose(table.sum(axis=0), 1.0, atol=1e-6), (
            "the control must stay column-stochastic, or the normalization "
            "violation would satisfy the assertions below on its own"
        )
        assert float(table.min()) == 0.0
        described = replace(described, b_matrix=table.reshape(1, 30, 30, 1))

        spec = parse_gnn_spec(spec_path("L1_letter"))
        violations = check_conformance(spec, described)
        joined = "\n".join(violations)
        assert "B: declared estimator=add-1-Laplace" in joined, joined
        assert "minimum entry of 0" in joined, joined
        assert "unsmoothed table under a smoothed declaration" in joined, joined
        assert not any("normalization" in violation for violation in violations), (
            f"the control leaked into the normalization check: {violations}"
        )

    def test_an_unknown_normalization_claim_is_refused_not_skipped(self, letter_agent):
        """Silently ignoring a claim would report the axis verified."""
        text = mutate(
            spec_text("L1_letter"),
            "normalization=column-stochastic-over-next-letter)",
            "normalization=row-stochastic-mystery)",
        )
        spec = parse_gnn_string(text, origin="mutated.md")
        joined = "\n".join(check_conformance(spec, describe_shapes(letter_agent)))
        assert "B: declared normalization=row-stochastic-mystery" in joined, joined
        assert "which this checker does not know" in joined, joined

    def test_a_disagreeing_gamma_is_reported(self, generative_letter_agent):
        """``gamma: 16.0`` was in ModelParameters and read by nothing.

        Precision is the one parameter this project measured a two-sided failure
        in, so a spec and a model that name different values describe different
        experiments.
        """
        text = mutate(spec_text("L1g_letter_generation"), "gamma: 16.0", "gamma: 4.0")
        spec = parse_gnn_string(text, origin="mutated.md")
        joined = "\n".join(check_conformance(spec, describe_shapes(generative_letter_agent)))
        assert "ModelParameters gamma=4.0 but the live agent has gamma=[16.0]" in joined, joined

    def test_the_checker_itself_now_rejects_a_baked_in_habit(
        self, generative_letter_agent, brown_bigram
    ):
        """The original ``E`` gap, closed inside ``check_conformance``.

        Until this, the only reader of a declared ``E`` was
        :func:`policy_prior_violations` in this file — so a caller using the
        documented one-liner got no policy-prior check at all.
        """
        described = describe_shapes(generative_letter_agent)
        habit = np.asarray(brown_bigram)[0][:, 0]
        described = replace(described, e_vector=(habit / habit.sum())[np.newaxis, :])
        assert described.e_vector.shape == (1, 30), "the control must be right-sized"

        spec = parse_gnn_spec(spec_path("L1g_letter_generation"))
        joined = "\n".join(check_conformance(spec, described))
        assert "E: declared Uniform(30)" in joined, joined
        assert "not uniform" in joined, joined

    def test_withholding_the_policy_prior_is_reported_not_ignored(self, generative_letter_agent):
        """An unverified E must be a violation, exactly as an unverified D is."""
        described = describe_shapes(generative_letter_agent)
        described = replace(described, e_shape=None)
        described = replace(described, e_vector=None)
        spec = parse_gnn_spec(spec_path("L1g_letter_generation"))
        joined = "\n".join(check_conformance(spec, described))
        assert "no e_shape was supplied" in joined, joined
        assert "the slot is unverified" in joined, joined

    def test_an_action_set_no_policy_reaches_is_reported(self, generative_letter_agent):
        """``pi>B`` is how the graph says an action reaches the dynamics."""
        text = mutate(spec_text("L1g_letter_generation"), "pi>B", "pi-B")
        spec = parse_gnn_string(text, origin="mutated.md")
        joined = "\n".join(check_conformance(spec, describe_shapes(generative_letter_agent)))
        assert "num_controls=30 but no PolicyVector is directed into the" in joined, joined
        assert "no path by which a chosen action reaches the dynamics" in joined, joined

    def test_an_emission_depending_on_an_undeclared_factor_is_reported(self, word_agent):
        """L3's ``s_word-A`` and ``s_pos-A`` declare a two-factor emission.

        Dropping one edge leaves a graph claiming the letter depends on word
        identity alone, against a live ``A`` that indexes position too. The
        dimension checks read ``A``'s dims from StateSpaceBlock, so they pass
        either way; only the graph notices.
        """
        text = mutate(spec_text("L3_word"), "\ns_pos-A", "")
        spec = parse_gnn_string(text, origin="mutated.md")
        joined = "\n".join(check_conformance(spec, describe_shapes(word_agent)))
        assert "Connections wire 1 hidden state(s) ['s_word']" in joined, joined
        assert "2 state axis/axes" in joined, joined

    def test_an_artifact_key_that_resolves_to_nothing_is_reported(self):
        """``key=`` says WHICH payload parameterizes the slot."""
        text = mutate(spec_text("L2a_phoneme_sequence"), "key=phone_unigrams", "key=phone_unigrums")
        spec = parse_gnn_string(text, origin="mutated.md")
        violations, resolved = artifact_key_violations(spec)
        assert any("is not a top-level key of" in v for v in violations), violations
        assert any("phone_unigrums" in v for v in violations), violations
        assert resolved == 1, f"only the mutated key must fail, got resolved={resolved}"

    def test_a_footer_citing_a_deleted_module_is_reported(self):
        """The Footer maps the spec to the code; a stale map is worse than none."""
        text = mutate(
            spec_text("L1_letter"), "src/ailm/levels/letter.py", "src/ailm/levels/deleted.py"
        )
        spec = parse_gnn_string(text, origin="mutated.md")
        violations, resolved = footer_path_violations(spec)
        assert any("src/ailm/levels/deleted.py" in v for v in violations), violations
        assert any("does not exist at" in v for v in violations), violations
        assert resolved == 1, f"the remaining path must still resolve, got {resolved}"


class TestEveryLevelTracksItsSpecDimensions:
    """The spec must track the code at EVERY level, not only L2a.

    ``test_mutating_a_declared_dimension_fails_against_the_real_agent`` proved
    the dimension-tracking property for one level. A property proved at one
    level and assumed at seven is exactly the shape of gap this file exists to
    close, so the mutation is now a table exercised against every shipped spec
    and its live builder-built agent: shrink the hidden-factor size by one
    everywhere it appears (StateSpaceBlock matrices, belief vectors, the
    structural Identity form where A is the identity, and ModelParameters),
    leaving ``num_obs`` alone so the mutated spec stays internally consistent
    and every violation below is attributable to the live-agent comparison.

    The positive control is
    :meth:`TestLiveAgentsConform.test_live_agent_conforms_to_its_spec` — the
    pristine spec conforms — and :func:`mutate` asserts every anchor matched,
    so a spec rewording turns this into a loud failure rather than a silent
    no-op mutation testing the unmodified spec.
    """

    #: (old, new) replacement pairs shrinking each level's hidden factor by one.
    #: Every pair is an anchor that must exist verbatim; a spec edit that moves
    #: any of these lines must update this table in the same commit.
    DIMENSION_MUTATIONS: dict[str, tuple[tuple[str, str], ...]] = {
        "L1_letter": (
            ("A[30,30,type=float]", "A[30,29,type=float]"),
            ("B[30,30,1,type=float]", "B[29,29,1,type=float]"),
            ("D[30,1,type=float]", "D[29,1,type=float]"),
            ("s_t[30,1,type=float]", "s_t[29,1,type=float]"),
            ("s_prime[30,1,type=float]", "s_prime[29,1,type=float]"),
            ("A=Identity(30)", "A=Identity(29)"),
            ("num_hidden_states: 30", "num_hidden_states: 29"),
        ),
        "L1g_letter_generation": (
            ("A[30,30,type=float]", "A[30,29,type=float]"),
            ("B[30,30,30,type=float]", "B[29,29,30,type=float]"),
            ("D[30,1,type=float]", "D[29,1,type=float]"),
            ("s_t[30,1,type=float]", "s_t[29,1,type=float]"),
            ("s_prime[30,1,type=float]", "s_prime[29,1,type=float]"),
            ("A=Identity(30)", "A=Identity(29)"),
            ("num_hidden_states: 30", "num_hidden_states: 29"),
        ),
        "L2a_phoneme_sequence": (
            ("A[39,39,type=float]", "A[39,38,type=float]"),
            ("B[39,39,1,type=float]", "B[38,38,1,type=float]"),
            ("D[39,1,type=float]", "D[38,1,type=float]"),
            ("s_t[39,1,type=float]", "s_t[38,1,type=float]"),
            ("s_prime[39,1,type=float]", "s_prime[38,1,type=float]"),
            ("num_hidden_states: 39", "num_hidden_states: 38"),
        ),
        "L2b_grapheme_to_phoneme": (
            ("A[30,39,type=float]", "A[30,38,type=float]"),
            ("B[39,39,1,type=float]", "B[38,38,1,type=float]"),
            ("D[39,1,type=float]", "D[38,1,type=float]"),
            ("s_t[39,1,type=float]", "s_t[38,1,type=float]"),
            ("s_prime[39,1,type=float]", "s_prime[38,1,type=float]"),
            ("num_hidden_states: 39", "num_hidden_states: 38"),
        ),
        "L3_word": (
            ("A[27,2000,21,type=float]", "A[27,2000,20,type=float]"),
            ("B_f1[21,21,1,type=float]", "B_f1[20,20,1,type=float]"),
            ("D_f1[21,1,type=float]", "D_f1[20,1,type=float]"),
            ("s_pos[21,1,type=float]", "s_pos[20,1,type=float]"),
            ("num_hidden_states_f1: 21", "num_hidden_states_f1: 20"),
        ),
        "L4_case": (
            ("A[12,8,type=float]", "A[12,7,type=float]"),
            ("B[8,8,1,type=float]", "B[7,7,1,type=float]"),
            ("D[8,1,type=float]", "D[7,1,type=float]"),
            ("s_t[8,1,type=float]", "s_t[7,1,type=float]"),
            ("s_prime[8,1,type=float]", "s_prime[7,1,type=float]"),
            ("num_hidden_states: 8", "num_hidden_states: 7"),
        ),
        "L5_clause": (
            ("A[8,16,type=float]", "A[8,15,type=float]"),
            ("B[16,16,1,type=float]", "B[15,15,1,type=float]"),
            ("D[16,1,type=float]", "D[15,1,type=float]"),
            ("s_t[16,1,type=float]", "s_t[15,1,type=float]"),
            ("s_prime[16,1,type=float]", "s_prime[15,1,type=float]"),
            ("num_hidden_states: 16", "num_hidden_states: 15"),
        ),
        "L5w_clause_widened": (
            ("A[10,20,type=float]", "A[10,19,type=float]"),
            ("B[20,20,1,type=float]", "B[19,19,1,type=float]"),
            ("D[20,1,type=float]", "D[19,1,type=float]"),
            ("s_t[20,1,type=float]", "s_t[19,1,type=float]"),
            ("s_prime[20,1,type=float]", "s_prime[19,1,type=float]"),
            ("num_hidden_states: 20", "num_hidden_states: 19"),
        ),
        "L6_narrative": (
            ("A[2,3,type=float]", "A[2,2,type=float]"),
            ("B[3,3,1,type=float]", "B[2,2,1,type=float]"),
            ("D[3,1,type=float]", "D[2,1,type=float]"),
            ("s_t[3,1,type=float]", "s_t[2,1,type=float]"),
            ("s_prime[3,1,type=float]", "s_prime[2,1,type=float]"),
            ("num_hidden_states: 3", "num_hidden_states: 2"),
        ),
    }

    #: The one violation each mutated spec MUST produce, pinned exactly so the
    #: transition-axis check — the one that compares the LIVE state dimension
    #: to ModelParameters — is confirmed present, not merely that something
    #: somewhere disagreed.
    EXPECTED_DIMENSION_VIOLATION: dict[str, str] = {
        "L1_letter": "B (TransitionMatrix): live state dimension 30 != ModelParameters num_hidden_states=29",
        "L1g_letter_generation": "B (TransitionMatrix): live state dimension 30 != ModelParameters num_hidden_states=29",
        "L2a_phoneme_sequence": "B (TransitionMatrix): live state dimension 39 != ModelParameters num_hidden_states=38",
        "L2b_grapheme_to_phoneme": "B (TransitionMatrix): live state dimension 39 != ModelParameters num_hidden_states=38",
        "L3_word": "B_f1 (TransitionMatrix, factor 1): live state dimension 21 != num_hidden_states_f1=20",
        "L4_case": "B (TransitionMatrix): live state dimension 8 != ModelParameters num_hidden_states=7",
        "L5_clause": "B (TransitionMatrix): live state dimension 16 != ModelParameters num_hidden_states=15",
        "L5w_clause_widened": "B (TransitionMatrix): live state dimension 20 != ModelParameters num_hidden_states=19",
        "L6_narrative": "B (TransitionMatrix): live state dimension 3 != ModelParameters num_hidden_states=2",
    }

    @pytest.mark.parametrize("name", SPEC_NAMES)
    def test_a_shrunken_hidden_factor_is_reported(self, name, live_agents):
        replacements = self.DIMENSION_MUTATIONS[name]
        text = spec_text(name)
        for old, new in replacements:
            text = mutate(text, old, new)
        spec = parse_gnn_string(text, origin=f"mutated_dims:{name}")
        violations = check_conformance(spec, describe_shapes(live_agents[name]))
        joined = "\n".join(violations)
        assert joined, (
            f"{name}: a spec whose hidden factor shrank by one was accepted "
            f"against the live builder-built agent — the spec no longer tracks "
            f"the code at this level."
        )
        assert "The spec contradicts itself" not in joined, (
            f"{name}: the mutated spec must stay internally consistent so the "
            f"violations are attributable to the live-agent comparison alone:\n{joined}"
        )
        assert self.EXPECTED_DIMENSION_VIOLATION[name] in joined, joined

    def test_the_mutation_table_covers_every_shipped_spec(self):
        """Default-deny: a spec added without a mutation row fails here."""
        assert set(self.DIMENSION_MUTATIONS) == set(SPEC_NAMES)
        assert set(self.EXPECTED_DIMENSION_VIOLATION) == set(SPEC_NAMES)


# ── UPSTREAM INTEROPERABILITY (verified, not assumed) ────────────────────────

_UPSTREAM_PARSER = UPSTREAM_SRC / "gnn" / "parsers" / "markdown_parser.py"
_UPSTREAM_AVAILABLE = _UPSTREAM_PARSER.is_file()

#: A checkout that exists but cannot supply the parser. Distinguished from an
#: absent one because the two call for opposite responses and the skip message
#: was telling the reader the wrong one: it said "no checkout" while a checkout
#: sat at that path with its source files reaped by temporary-directory cleanup
#: and only ``__pycache__`` left behind. Thirty-three interoperability tests
#: stopped running, and the reason they gave for stopping was false.
_UPSTREAM_GUTTED = (
    not _UPSTREAM_AVAILABLE and UPSTREAM_ROOT.is_dir() and (UPSTREAM_SRC / "gnn").is_dir()
)

#: The upstream parser imports ``psutil``, which this project only installs
#: through the optional ``gnn-interop`` extra. A checkout with no ``psutil`` is
#: a fourth state, and it used to be reported as a bare ``importorskip`` with
#: no mention of the extra that fixes it.
_PSUTIL_AVAILABLE = importlib.util.find_spec("psutil") is not None
_UPSTREAM_RUNNABLE = _UPSTREAM_AVAILABLE and _PSUTIL_AVAILABLE

#: Set by the full release profile (``scripts/verify_release.py``), which forbids
#: skips. Under it, a missing upstream checkout or a missing ``psutil`` is a
#: release failure that names the gap, not a skip the profile would reject
#: anyway without saying what to install.
RELEASE_MANDATORY_ENVIRONMENT_ENV = "AILM_RELEASE_MANDATORY_ENVIRONMENT"


def _upstream_skip_reason() -> str:
    """Why the interoperability checks are not running, said accurately.

    Four states, not two. The function previously had no branch for the case
    where the parser IS available, so it fell through to "no checkout" and
    claimed an absence at a path holding a working checkout — the same class of
    false reason it was written to fix, one level up. The fourth state is a
    usable checkout whose parser cannot import ``psutil``.
    """
    if _UPSTREAM_AVAILABLE and not _PSUTIL_AVAILABLE:
        return (
            f"the upstream parser is available at {_UPSTREAM_PARSER} but psutil, which it "
            f"imports, is not installed in this interpreter; install the gnn-interop extra "
            f"(uv sync --extra gnn-interop) to run the interoperability check"
        )
    if _UPSTREAM_AVAILABLE:
        return f"the upstream parser is available at {_UPSTREAM_PARSER}; nothing is skipped"
    if _UPSTREAM_GUTTED:
        return (
            f"a GeneralizedNotationNotation checkout is present at {UPSTREAM_ROOT} but does "
            f"not supply {_UPSTREAM_PARSER.name}; the parsers package is there and its "
            f"sources are not, which is what temporary-directory cleanup leaves behind. "
            f"This is a BROKEN environment rather than an absent one: re-clone, or point "
            f"AILM_GNN_UPSTREAM somewhere that is not reaped."
        )
    return (
        f"no GeneralizedNotationNotation checkout at {UPSTREAM_ROOT} "
        f"(set AILM_GNN_UPSTREAM to enable the interoperability check)"
    )


@pytest.fixture(scope="module")
def upstream_results(tmp_path_factory):
    """Parse every spec with real upstream code in a fresh isolated interpreter.

    Interpreter flags select a private empty bytecode prefix and disable writes,
    so verified source is compiled without replacing dependencies or changing
    the parent interpreter's global configuration. The subprocess emits the
    real parser's result data; no replacement parser object is constructed.
    The class guard owns missing-prerequisite skip/failure decisions.
    """
    if not _UPSTREAM_RUNNABLE:
        yield None
        return
    pin = load_release_pin(PROJECT_ROOT / "gnn/upstream.json")
    identity = verify_release_checkout(UPSTREAM_ROOT, pin)
    cache_prefix = tmp_path_factory.mktemp("gnn-release-bytecode")
    code = """
import json
import sys
from pathlib import Path

source, cache_prefix, *specs = sys.argv[1:]
assert sys.dont_write_bytecode and sys.pycache_prefix == cache_prefix
sys.path.insert(0, source)
from gnn.parsers.markdown_parser import MarkdownGNNParser

module_file = Path(sys.modules['gnn.parsers.markdown_parser'].__file__).resolve()
assert Path(source).resolve() in module_file.parents, module_file
parser = MarkdownGNNParser()
results = {}
for spec in specs:
    result = parser.parse_file(spec)
    model = result.model
    results[Path(spec).name.removesuffix('.gnn.md')] = {
        'success': result.success,
        'errors': result.errors,
        'warnings': result.warnings,
        'model': {
            'model_name': model.model_name,
            'variables': [{'name': v.name, 'dimensions': v.dimensions} for v in model.variables],
            'ontology_mappings': [
                {'variable_name': m.variable_name, 'ontology_term': m.ontology_term}
                for m in model.ontology_mappings
            ],
            'parameters': [{'name': p.name, 'value': p.value} for p in model.parameters],
        },
    }
print(json.dumps(results))
"""
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-X",
                f"pycache_prefix={cache_prefix}",
                "-c",
                code,
                str(UPSTREAM_SRC),
                str(cache_prefix),
                *(str(spec_path(name)) for name in SPEC_NAMES),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        results = json.loads(completed.stdout)
        assert set(results) == set(SPEC_NAMES)
        assert not list(cache_prefix.iterdir()), "the no-bytecode parser invocation wrote a cache"
        yield results
    finally:
        assert verify_release_checkout(UPSTREAM_ROOT, pin) == identity, (
            "upstream source changed during interoperability tests"
        )


class TestUpstreamInteroperability:
    """AILM's specs must parse with the REAL GNN parser, not only with ours.

    ``ailm.gnn`` is deliberately a strict subset reader. That buys independence
    from a 25-step pipeline, but it also means a spec could satisfy AILM while
    being unreadable to the notation's own tooling — which would make "expressed
    in GNN" a claim about a private dialect. These tests close that gap.
    """

    @pytest.fixture(autouse=True)
    def _require_the_upstream_results(self) -> None:
        """Skip when the parser cannot run; fail instead under the release profile.

        A class-level ``skipif`` would skip unconditionally, and the full
        release profile forbids skips, so it could never pass on a host missing
        the checkout or the extra. Under ``AILM_RELEASE_MANDATORY_ENVIRONMENT=1``
        the same condition is a failure whose message says what to install.
        """
        if _UPSTREAM_RUNNABLE:
            return
        reason = _upstream_skip_reason()
        if os.environ.get(RELEASE_MANDATORY_ENVIRONMENT_ENV) == "1":
            pytest.fail(reason)
        pytest.skip(reason)

    @pytest.mark.parametrize("name", SPEC_NAMES)
    def test_upstream_parses_the_spec(self, name, upstream_results):
        result = upstream_results[name]
        assert result["success"], f"{name}: upstream parse failed: {result['errors']}"
        assert not result["errors"], f"{name}: {result['errors']}"
        assert not result["warnings"], f"{name}: {result['warnings']}"

    @pytest.mark.parametrize("name", SPEC_NAMES)
    def test_upstream_and_ailm_agree_on_the_model(self, name, upstream_results):
        """Both readers must extract the SAME model, not merely both succeed."""
        ours = parse_gnn_spec(spec_path(name))
        theirs = upstream_results[name]["model"]

        assert theirs["model_name"] == ours.model_name

        upstream_vars = {
            UPSTREAM_NAME_ALIASES.get(v["name"], v["name"]): tuple(v["dimensions"])
            for v in theirs["variables"]
        }
        our_vars = {n: dims for n, (dims, _dtype) in ours.variables.items()}
        assert upstream_vars == our_vars, (
            f"{name}: dimension disagreement between the two readers: "
            f"upstream={upstream_vars} ailm={our_vars}"
        )

        upstream_ontology = {
            UPSTREAM_NAME_ALIASES.get(m["variable_name"], m["variable_name"]): m["ontology_term"]
            for m in theirs["ontology_mappings"]
        }
        assert upstream_ontology == ours.ontology

        # Every ModelParameters key we read, not a fixed four: L3 has no scalar
        # num_hidden_states and L1g adds num_controls and gamma, so a hardcoded
        # list would silently stop covering whatever a spec declares next.
        upstream_params = {p["name"]: p["value"] for p in theirs["parameters"]}
        assert len(ours.parameters) >= 4, ours.parameters
        for key, value in ours.parameters.items():
            assert upstream_params[key] == value, key

    @pytest.mark.parametrize("name", SPEC_NAMES)
    def test_the_recorded_rename_is_the_only_reader_disagreement(self, name, upstream_results):
        """An alias must never grow into a place a real disagreement can hide.

        :data:`UPSTREAM_NAME_ALIASES` is applied above to reconcile the two
        readers. This pins what it is reconciling: upstream renaming ``pi`` to
        ``π`` in L1g, and nothing else anywhere.
        """
        ours = set(parse_gnn_spec(spec_path(name)).variables)
        theirs = {v["name"] for v in upstream_results[name]["model"]["variables"]}
        renamed = theirs - ours
        assert renamed <= set(UPSTREAM_NAME_ALIASES), (
            f"{name}: upstream reports variable(s) {sorted(renamed)} that AILM "
            f"does not, and they are not the recorded rename. That is a real "
            f"disagreement between the readers, not a notation difference."
        )
        expected = {"π"} if name == "L1g_letter_generation" else set()
        assert renamed == expected, (
            f"{name}: expected the rename set {sorted(expected)}, got {sorted(renamed)}. "
            f"If upstream stopped normalizing 'pi', delete the alias rather than "
            f"leaving an unexercised one behind."
        )

    @pytest.mark.parametrize("name", SINGLE_FACTOR_SPEC_NAMES)
    def test_upstream_sees_the_matrix_roles(self, name, upstream_results):
        """The ontology block is the load-bearing part; upstream must read it."""
        theirs = upstream_results[name]["model"]
        roles = {m["variable_name"]: m["ontology_term"] for m in theirs["ontology_mappings"]}
        assert roles["A"] == "RecognitionMatrix"
        assert roles["B"] == "TransitionMatrix"
        assert roles["D"] == "Prior"

    @pytest.mark.parametrize("name", MULTIFACTOR_SPEC_NAMES)
    def test_upstream_sees_the_per_factor_matrix_roles(self, name, upstream_results):
        """The factor-suffixed roles are what bind a matrix to a factor."""
        theirs = upstream_results[name]["model"]
        roles = {m["variable_name"]: m["ontology_term"] for m in theirs["ontology_mappings"]}
        assert roles["A"] == "RecognitionMatrix"
        assert roles["B_f0"] == roles["B_f1"] == "TransitionMatrix"
        assert roles["D_f0"] == roles["D_f1"] == "Prior"

    def test_upstream_reads_the_g2p_asymmetry(self, upstream_results):
        """30 letter observations over 39 hidden phones, read by upstream."""
        theirs = upstream_results["L2b_grapheme_to_phoneme"]["model"]
        dims = {v["name"]: tuple(v["dimensions"]) for v in theirs["variables"]}
        assert dims["A"] == (30, 39)
        assert dims["B"] == (39, 39, 1)
        params = {p["name"]: p["value"] for p in theirs["parameters"]}
        assert params["num_obs"] == 30
        assert params["num_hidden_states"] == 39


class TestInferenceConfiguration:
    """The claims the matrices cannot carry.

    Every other check in this file reads a slot. These read the configuration
    that decides whether the slots are used at all, and until `describe_shapes`
    carried them they were unverifiable: an agent with a correct non-flat `C`, a
    correct `E` and a genuinely controllable `B` still does not minimize expected
    free energy over them if `use_utility` is off. No shape changes, no entry
    changes, and the conformance report stays clean while the manuscript's claim
    about generation is false.
    """

    def _controlled_spec(self):
        return load_spec("L1g_letter_generation")

    def test_the_shipped_generative_agent_conforms_with_its_configuration(
        self, generative_letter_agent
    ):
        """The negative control: the real agent passes the new checks."""
        described = describe_shapes(generative_letter_agent)
        assert check_conformance(self._controlled_spec(), described) == []

    def test_describe_shapes_reports_the_configuration_at_all(self, generative_letter_agent):
        """A check cannot fire on a field the description never carries.

        This is the guard on the guard. The mutants below all flip a value in
        the description, so they would pass vacuously if `describe_shapes` had
        quietly stopped reporting the field.
        """
        described = describe_shapes(generative_letter_agent)
        for key in (
            "use_utility",
            "use_states_info_gain",
            "use_param_info_gain",
            "action_selection",
            "policy_len",
            "n_policies",
            "num_controls",
        ):
            assert hasattr(described, key), f"{key} is no longer described, so nothing can check it"

    @pytest.mark.parametrize(
        "mutation,fragment",
        [
            ({"use_utility": False}, "preference vector never enters"),
            ({"use_states_info_gain": False}, "epistemic term is absent"),
            ({"num_controls": (7,)}, "different sizes"),
            ({"n_policies": 900}, "distribution over the policy space"),
        ],
        ids=["utility-off", "epistemic-off", "control-count-drift", "policy-space-drift"],
    )
    def test_each_configuration_mutant_is_caught(self, generative_letter_agent, mutation, fragment):
        """Named mutants, one per way the configuration can lie.

        `num_controls` is checked against the agent's own attribute rather than
        against `B`'s shape because the shape reduction drops trailing size-one
        axes, and the docstring on that reduction already concedes it does not
        compare the number of controls.
        """
        described = describe_shapes(generative_letter_agent)
        described = replace(described, **mutation)
        violations = check_conformance(self._controlled_spec(), described)
        assert any(fragment in v for v in violations), (
            f"mutation {mutation} produced {violations or 'no violations'}, "
            f"expected one containing {fragment!r}"
        )

    def test_an_uncontrolled_level_is_not_asked_for_a_preference_term(self):
        """The checks must not fire on the comprehension levels.

        Every comprehension level runs with a flat `C` and one action by design,
        and that design is what the paper says about them. A rule demanding a
        preference term of every model would fail all six for conforming exactly
        as intended, which is how a guard becomes something people delete.
        """
        spec = load_spec("L1_letter")
        violations = _check_inference_configuration(
            spec, use_utility=False, use_states_info_gain=False, num_controls=(1,)
        )
        assert violations == []


def test_a_gutted_upstream_checkout_is_reported_as_broken_not_absent() -> None:
    """A skip must say why, and this one said the wrong why.

    The interoperability suite verifies that AILM's specs parse with the REAL
    GNN parser rather than only with this repository's strict subset reader.
    That is what makes "expressed in GNN" a claim about the notation instead of
    about a private dialect, so those tests silently ceasing to run is a loss of
    the evidence behind a manuscript claim.

    They ceased because the default checkout path is a temporary directory, and
    cleanup reaped the parser sources while leaving the package directory and its
    ``__pycache__`` in place. The suite then reported "no checkout" at a path
    where a checkout plainly was. An absent checkout is a normal condition on a
    fresh machine; a gutted one is a broken environment. Reporting the second as
    the first sends the reader to install something already installed.
    """
    if _UPSTREAM_AVAILABLE:
        reason = _upstream_skip_reason()
        assert "no GeneralizedNotationNotation checkout" not in reason
        if not _PSUTIL_AVAILABLE:
            assert "psutil" in reason and "gnn-interop" in reason, (
                f"the checkout is usable but psutil is missing, and the skip reason does not "
                f"name the extra that installs it: {reason!r}"
            )
        return
    reason = _upstream_skip_reason()
    if _UPSTREAM_GUTTED:
        assert "BROKEN environment" in reason and "reaped" in reason, (
            f"the checkout at {UPSTREAM_ROOT} is present but unusable, and the skip reason "
            f"does not say so: {reason!r}"
        )
    else:
        assert "no GeneralizedNotationNotation checkout" in reason


def test_the_interoperability_check_does_not_depend_on_a_scratch_directory() -> None:
    """Every discovery candidate shares the checkout's lifetime.

    A scratch probe was once the only default and was reaped, which silently
    disabled the interoperability tests; the candidates are now the checkout's
    sibling namespace and a plain sibling directory, and nothing under a
    temporary root is searched.
    """
    volatile = ("/tmp", "/private/tmp", "/var/folders")
    assert _UPSTREAM_CANDIDATES, "discovery needs at least the plain sibling candidate"
    for candidate in _UPSTREAM_CANDIDATES:
        assert not str(candidate).startswith(volatile), (
            f"{candidate} is a temporary directory; the interoperability check "
            f"would be lost on the next cleanup"
        )
        assert candidate.name == "Generalized_Notation_Notation", candidate
    assert _UPSTREAM_CANDIDATES[-1] == PROJECT_ROOT.parent / "Generalized_Notation_Notation"


@pytest.mark.parametrize(
    "canonical_kind,fallback_exists,selected_index",
    [("file", True, 0), ("missing", True, 1), ("directory", True, 1), ("missing", False, 0)],
)
def test_upstream_discovery_requires_a_real_parser_file(
    tmp_path: Path, canonical_kind: str, fallback_exists: bool, selected_index: int
) -> None:
    candidates = (tmp_path / "canonical", tmp_path / "fallback")
    relative = Path("src/gnn/parsers/markdown_parser.py")
    if canonical_kind != "missing":
        parser = candidates[0] / relative
        parser.parent.mkdir(parents=True)
        if canonical_kind == "directory":
            parser.mkdir()
        else:
            parser.write_text("# canonical parser source\n", encoding="utf-8")
    if fallback_exists:
        parser = candidates[1] / relative
        parser.parent.mkdir(parents=True)
        parser.write_text("# fallback parser source\n", encoding="utf-8")
    assert _default_upstream_root(candidates) == candidates[selected_index]


class TestCombinationRuleDeclaration:
    """The rule a level's numbers were measured under, declared in its spec.

    Decision 47 established that the combination rule — not the insertion site —
    decides whether an inter-level message can help: under `linear` the two sites
    behave differently and a uniform message moves posteriors, while under
    `geometric` the sites are one computation and a uniform message is inert. A
    spec silent about the rule leaves a reader unable to tell which regime a
    number came from.

    The parameter is optional, because most levels carry no inter-level message
    and should not be made to declare something they do not use.
    """

    def test_a_spec_without_the_parameter_is_silent_rather_than_wrong(self):
        spec = parse_gnn_string(spec_text("L1_letter"))
        assert _check_combination_rule(spec) == []

    def test_the_shipped_g2p_spec_declares_the_rule_its_numbers_used(self):
        spec = parse_gnn_string(spec_text("L2b_grapheme_to_phoneme"))
        assert spec.parameters.get("combination_rule") == "linear", (
            "L2B's recorded channel numbers were measured under linear mixing; if the "
            "shipped decoder ever changes rule, this declaration and every recorded "
            "lexical-channel number move together or the spec is lying"
        )
        assert _check_combination_rule(spec) == []

    @pytest.mark.parametrize("value", ["additive", "Linear", "product-of-experts"])
    def test_an_unimplemented_rule_is_rejected(self, value):
        """POSITIVE CONTROL. A rule the project cannot run reads as a checked
        claim and is not one, so declaring one must fail rather than pass.

        ``Linear`` is in the list on purpose: a case-insensitive match would let a
        spec declare a rule the code would then fail to dispatch on, which is the
        same defect as an invented name wearing a familiar spelling.
        """
        text = mutate(
            spec_text("L2b_grapheme_to_phoneme"),
            "combination_rule: linear",
            f"combination_rule: {value}",
        )
        violations = _check_combination_rule(parse_gnn_string(text))
        assert violations, f"{value!r} was accepted as a combination rule"
        assert "not one of" in violations[0]

    def test_an_empty_declaration_fails_at_parse_rather_than_at_conformance(self):
        """The layering, asserted so it cannot quietly invert.

        A key with no value is a malformed spec, and the parser rejects it before
        any conformance check runs. A well-formed spec naming a rule the project
        does not implement is a different failure and belongs to conformance. If
        the parser ever started tolerating the empty form, this check would move
        to `_check_combination_rule` — and silently not exist until someone
        noticed, which is what this test prevents.
        """
        text = mutate(
            spec_text("L2b_grapheme_to_phoneme"),
            "combination_rule: linear",
            "combination_rule:",
        )
        with pytest.raises(GnnSpecError, match="malformed ModelParameters entry"):
            parse_gnn_string(text)

    def test_every_declared_rule_names_something_the_code_implements(self):
        """The declaration and the implementation must not drift apart.

        `COMBINATION_RULES` is the spec-side vocabulary; the code side is the
        geometric path in `ailm.channels` and the linear path in the shipped
        decoders. A rule listed here with no implementation would pass
        conformance and fail at run time.
        """
        from ailm.channels import _geometric_positions
        from ailm.levels.lexical_channel import coupled_g2p_positions

        assert set(COMBINATION_RULES) == {"linear", "geometric"}
        assert callable(coupled_g2p_positions) and callable(_geometric_positions)


# ── PACKAGE SHAPE ────────────────────────────────────────────────────────────


def test_the_conformance_package_layers_do_not_cycle():
    """The dependency order is linear, and a cycle would make it unreadable.

    `conformance.py` was one file of 1,260 lines holding three concerns: checks
    that read only the spec, checks that need the built agent, and the
    orchestration over both. The seam was in the code and invisible in it — you
    could not tell from a function name whether it would touch a live array, so
    you could not tell what a change to it could break.

    The split only helps while the layering holds. `declarations` must not reach
    into `live`, and neither may reach back into `conformance`, or the entry
    point stops being an entry point.

    NEGATIVE CONTROL: the expected edges must actually be present, or this would
    pass on three modules that import nothing from each other at all.
    """
    import ast
    from pathlib import Path

    package = Path(__file__).resolve().parents[1] / "src" / "ailm" / "gnn"
    order = ["types", "parsing", "shapes", "declarations", "live", "conformance", "registry"]
    rank = {name: index for index, name in enumerate(order)}

    edges: dict[str, set[str]] = {}
    for name in order:
        source = (package / f"{name}.py").read_text(encoding="utf-8")
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("ailm.gnn."):
                imported.add((node.module or "").rsplit(".", 1)[-1])
        edges[name] = imported

    for name, imports in edges.items():
        for target in imports:
            assert target in rank, f"{name} imports unknown sibling {target}"
            assert rank[target] < rank[name], (
                f"{name} imports {target}, which is later in the layer order; "
                f"the dependency graph has a cycle or an inversion"
            )

    # Negative control: the split modules must actually be wired to each other.
    assert "declarations" in edges["live"], "live no longer builds on declarations"
    assert {"declarations", "live"} <= edges["conformance"], (
        "the entry point no longer collects both families of check"
    )


def test_no_conformance_module_is_a_monolith():
    """Each module stays reviewable on its own.

    The bound is what the split bought; without it the largest file drifts back
    toward the 1,260 lines it started at, one check at a time, and no single
    commit is ever the one that made it unreadable.
    """
    from pathlib import Path

    package = Path(__file__).resolve().parents[1] / "src" / "ailm" / "gnn"
    for path in sorted(package.glob("*.py")):
        length = len(path.read_text(encoding="utf-8").splitlines())
        assert length < 600, f"{path.name} is {length} lines; split it along a concern"
