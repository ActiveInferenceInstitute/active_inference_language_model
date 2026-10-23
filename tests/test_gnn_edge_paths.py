"""Error branches of ``ailm.gnn`` the shipped specs never walk.

``tests/test_gnn_conformance.py`` binds every shipped spec to a live agent and
mutates real specs into positive controls. That strategy has a blind spot by
construction: a refusal path a *correct* spec never triggers — a malformed
dimension token, a duplicate factor binding, a multifactor slot with no live
array — stays unexecuted, and an unexecuted refusal is a refusal nobody has
seen fail. Every test here follows the house rule of that file: the assertion
is on the reported violation TEXT (or the raised message), never on the bare
fact that something was non-empty, so an unrelated failure cannot satisfy it.

Synthetic minimal specs are used instead of shipped ones because the branches
under test are exactly the ones no shipped spec can reach — a shipped spec that
reached them would fail its own gate. The specs are still parsed by the real
parser and checked by the real checker; nothing project-owned is stubbed.
"""

from __future__ import annotations

import numpy as np
import pytest

from ailm.gnn import (
    GnnSpec,
    GnnSpecError,
    _check_inference_configuration,
    check_conformance,
    parse_gnn_string,
    parse_learned_arguments,
)
from ailm.gnn.declarations import _check_declaration_coherence, _factor_sizes
from ailm.gnn.live import (
    _check_emission_factor_count,
    _check_learned_arguments,
    _check_precision,
    _check_structural_form,
)
from ailm.gnn.types import LiveConformanceInputs

# ── SYNTHETIC SPEC BUILDERS ──────────────────────────────────────────────────


def build_spec_text(
    *,
    state_space: str,
    ontology: str,
    parameters: str | None = None,
    connections: str | None = None,
    parameterization: str | None = None,
    time: str | None = None,
    model_name: str = "AILM synthetic probe",
) -> str:
    """Assemble a GNN document from section bodies.

    Only the three required sections are always present, so a test can probe
    exactly the optional-section branches it means to.
    """
    parts = [f"## ModelName\n{model_name}", f"## StateSpaceBlock\n{state_space}"]
    if connections is not None:
        parts.append(f"## Connections\n{connections}")
    if parameterization is not None:
        parts.append(f"## InitialParameterization\n{parameterization}")
    if time is not None:
        parts.append(f"## Time\n{time}")
    parts.append(f"## ActInfOntologyAnnotation\n{ontology}")
    if parameters is not None:
        parts.append(f"## ModelParameters\n{parameters}")
    return "\n\n".join(parts) + "\n"


def single_factor_spec(
    *,
    extra_state: str = "",
    extra_ontology: str = "",
    parameters: str = "num_hidden_states: 3\nnum_obs: 3",
    **sections: str | None,
) -> GnnSpec:
    """A minimal single-factor spec: the three mandatory matrix roles, size 3."""
    return parse_gnn_string(
        build_spec_text(
            state_space="A[3,3,type=float]\nB[3,3,type=float]\nD[3,1,type=float]" + extra_state,
            ontology="A=RecognitionMatrix\nB=TransitionMatrix\nD=Prior" + extra_ontology,
            parameters=parameters,
            **sections,
        ),
        origin="synthetic_single.md",
    )


def two_factor_spec(
    *,
    state_space: str = (
        "A_m0[3,2,2,type=float]\n"
        "B_f0[2,2,type=float]\n"
        "B_f1[2,2,type=float]\n"
        "D_f0[2,1,type=float]\n"
        "D_f1[2,1,type=float]"
    ),
    ontology: str = (
        "A_m0=RecognitionMatrix\n"
        "B_f0=TransitionMatrix\n"
        "B_f1=TransitionMatrix\n"
        "D_f0=Prior\n"
        "D_f1=Prior"
    ),
    parameters: str = (
        "num_factors: 2\nnum_hidden_states_f0: 2\nnum_hidden_states_f1: 2\nnum_obs: 3"
    ),
    **sections: str | None,
) -> GnnSpec:
    """A minimal two-factor spec that conforms to :func:`two_factor_live`."""
    return parse_gnn_string(
        build_spec_text(
            state_space=state_space, ontology=ontology, parameters=parameters, **sections
        ),
        origin="synthetic_two_factor.md",
    )


def two_factor_live(**overrides: object) -> LiveConformanceInputs:
    """The typed live record that conforms to the default :func:`two_factor_spec`."""
    live: dict[str, object] = {
        "a_matrices": [np.full((3, 2, 2), 1.0 / 3.0)],
        "b_matrices": [np.eye(2), np.eye(2)],
        "d_vectors": [np.full(2, 0.5), np.full(2, 0.5)],
        "n_modalities": 1,
        "n_factors": 2,
    }
    live.update(overrides)
    return LiveConformanceInputs(**live)  # type: ignore[arg-type]


def test_the_synthetic_baselines_conform() -> None:
    """NEGATIVE CONTROL for every mutant below.

    If the unmutated synthetic spec already carried violations, each mutant
    test's ``any(...)`` could be satisfied by baseline noise rather than by the
    branch it was written for.
    """
    assert check_conformance(two_factor_spec(), two_factor_live()) == []
    assert _check_declaration_coherence(single_factor_spec()) == []


# ── PARSING REFUSALS (ailm.gnn.parsing) ──────────────────────────────────────


class TestParsingRefusals:
    """Malformed lines must raise :class:`GnnSpecError` naming the problem.

    Each anchors on the message fragment unique to its branch, so a different
    parse failure on the same input cannot fake a pass.
    """

    def _parse_state_space(self, block: str) -> GnnSpec:
        return parse_gnn_string(
            build_spec_text(state_space=block, ontology="A=RecognitionMatrix"),
            origin="bad.md",
        )

    def test_empty_dimension_token_is_rejected(self) -> None:
        """``A[3,,type=float]`` would silently declare fewer dims than typed."""
        with pytest.raises(GnnSpecError, match="empty dimension token"):
            self._parse_state_space("A[3,,type=float]")

    def test_empty_type_specification_is_rejected(self) -> None:
        """``type=`` with nothing after it is a dtype claim about nothing."""
        with pytest.raises(GnnSpecError, match="empty 'type=' specification"):
            self._parse_state_space("A[3,type=]")

    def test_dimension_after_the_type_token_is_rejected(self) -> None:
        """A dim after ``type=`` would be dropped by a lenient reader."""
        with pytest.raises(GnnSpecError, match="dimension token after the 'type=' token"):
            self._parse_state_space("A[type=float,3]")

    def test_variable_with_only_a_type_is_rejected(self) -> None:
        """A variable with a dtype and no dims has no shape to check."""
        with pytest.raises(GnnSpecError, match="declares no dimensions"):
            self._parse_state_space("A[type=float]")

    def test_empty_state_space_block_is_rejected(self) -> None:
        """A StateSpaceBlock with no variables pins nothing at all."""
        with pytest.raises(GnnSpecError, match="declares no variables"):
            self._parse_state_space("# only a comment")

    def test_connection_with_an_empty_endpoint_is_rejected(self) -> None:
        """``s>`` names a source and no target — half an edge is not an edge."""
        with pytest.raises(GnnSpecError, match="empty endpoint"):
            single_factor_spec(
                extra_state="\ns[3,1,type=float]",
                extra_ontology="\ns=HiddenState",
                connections="s>",
            )

    def test_duplicate_parameterization_is_rejected(self) -> None:
        """Two forms for one variable: the second would silently win."""
        with pytest.raises(GnnSpecError, match="duplicate parameterization for 'D'"):
            single_factor_spec(parameterization="D=Uniform(3)\nD=Zeros(3)")

    def test_malformed_ontology_line_is_rejected(self) -> None:
        """An annotation that is not ``Variable=Term`` binds no role."""
        with pytest.raises(GnnSpecError, match="malformed ontology annotation"):
            parse_gnn_string(
                build_spec_text(
                    state_space="A[3,3,type=float]",
                    ontology="A -> RecognitionMatrix",
                ),
                origin="bad.md",
            )

    def test_duplicate_ontology_annotation_is_rejected(self) -> None:
        """Two annotations for one variable: which role would be checked?"""
        with pytest.raises(GnnSpecError, match="duplicate ontology annotation for 'A'"):
            parse_gnn_string(
                build_spec_text(
                    state_space="A[3,3,type=float]",
                    ontology="A=RecognitionMatrix\nA=TransitionMatrix",
                ),
                origin="bad.md",
            )

    def test_duplicate_model_parameter_key_is_rejected(self) -> None:
        """Two ``num_obs`` lines: the second would silently win."""
        with pytest.raises(GnnSpecError, match="duplicate ModelParameters key 'num_obs'"):
            single_factor_spec(parameters="num_obs: 3\nnum_obs: 4")


class TestParsingAcceptedEdges:
    """Accepted inputs whose parse result must be exactly right, not just parse."""

    def test_grouped_connection_endpoints_are_split_into_names(self) -> None:
        """``(s,o)>A`` declares two sources; collapsing them to one string
        would make every graph check compare against a name nobody declared."""
        spec = single_factor_spec(
            extra_state="\ns[3,1,type=float]\no[3,1,type=int]",
            extra_ontology="\ns=HiddenState\no=Observation",
            connections="(s,o)>A\nD>s\ns>B\nA>o",
        )
        assert spec.connections[0].source == ("s", "o")
        assert spec.connections[0].target == ("A",)
        assert spec.connections[0].directed is True

    def test_absent_optional_sections_parse_to_empty_collections(self) -> None:
        """A spec with only the required sections must yield empty (not None,
        not missing) connections, parameterization and parameters — the
        conformance checker indexes all three unconditionally."""
        spec = parse_gnn_string(
            build_spec_text(state_space="A[3,3,type=float]", ontology="A=RecognitionMatrix"),
            origin="minimal.md",
        )
        assert spec.connections == []
        assert spec.parameterization == {}
        assert spec.parameters == {}
        # And the declaration checks must treat the absences as "no claim made"
        # rather than crash or invent violations: no Time section, no
        # Connections graph, no ModelParameters sizes.
        assert _check_declaration_coherence(spec) == []

    def test_positional_learned_argument_is_dropped_not_misread(self) -> None:
        """A ``Learned`` entry with no ``=`` names nothing; it must not become
        a key, a value, or a crash."""
        assert parse_learned_arguments("brown_counts, generator=scripts/g.py") == {
            "generator": "scripts/g.py"
        }


# ── DECLARATION-ONLY REFUSALS (ailm.gnn.declarations) ────────────────────────


class TestDeclarationRefusals:
    """Spec-internal incoherences no shipped spec carries."""

    def test_a_non_positive_factor_size_is_reported(self) -> None:
        """``num_hidden_states_f1: 0`` sizes a factor with nothing."""
        spec = two_factor_spec(
            parameters="num_factors: 2\nnum_hidden_states_f0: 2\nnum_hidden_states_f1: 0\nnum_obs: 3"
        )
        sizes, problems = _factor_sizes(spec, 2)
        assert sizes == [2, None]
        assert any("num_hidden_states_f1=0 is not a positive integer" in p for p in problems), (
            problems
        )

    def test_a_matrix_shaped_belief_is_reported(self) -> None:
        """A HiddenState declared ``[2,3]`` is a joint pymdp never forms."""
        spec = single_factor_spec(
            extra_state="\ns[2,3,type=float]",
            extra_ontology="\ns=HiddenState",
        )
        violations = _check_declaration_coherence(spec)
        assert any(
            "s (HiddenState): declared dims (2, 3)" in v and "is a vector" in v for v in violations
        ), violations

    def test_a_multifactor_belief_multiset_mismatch_is_reported(self) -> None:
        """Two factors sized (2, 3) with both beliefs declared size 2: a factor
        was resized in ModelParameters and its belief left behind."""
        spec = two_factor_spec(
            state_space=(
                "A_m0[3,2,3,type=float]\n"
                "B_f0[2,2,type=float]\n"
                "B_f1[3,3,type=float]\n"
                "D_f0[2,1,type=float]\n"
                "D_f1[3,1,type=float]\n"
                "s_a[2,1,type=float]\n"
                "s_b[2,1,type=float]"
            ),
            ontology=(
                "A_m0=RecognitionMatrix\n"
                "B_f0=TransitionMatrix\n"
                "B_f1=TransitionMatrix\n"
                "D_f0=Prior\n"
                "D_f1=Prior\n"
                "s_a=HiddenState\n"
                "s_b=HiddenState"
            ),
            parameters="num_factors: 2\nnum_hidden_states_f0: 2\nnum_hidden_states_f1: 3\nnum_obs: 3",
        )
        violations = _check_declaration_coherence(spec)
        assert any(
            "declared with sizes [2, 2]" in v and "declared factor sizes are [2, 3]" in v
            for v in violations
        ), violations

    def test_a_multi_modality_observation_is_skipped_not_guessed(self) -> None:
        """With ``num_modalities: 2`` nothing binds an observation variable to a
        modality, so its size must be SKIPPED — and the negative control shows
        the identical wrong size IS reported at one modality, so the skip is a
        real branch and not a checker that never sizes observations."""
        mis_sized = "\no[5,1,type=int]"
        annotated = "\no=Observation"
        skipped = _check_declaration_coherence(
            single_factor_spec(
                extra_state=mis_sized,
                extra_ontology=annotated,
                parameters="num_hidden_states: 3\nnum_obs: 3\nnum_modalities: 2",
            )
        )
        assert not any("Observation" in v for v in skipped), skipped

        caught = _check_declaration_coherence(
            single_factor_spec(
                extra_state=mis_sized,
                extra_ontology=annotated,
                parameters="num_hidden_states: 3\nnum_obs: 3\nnum_modalities: 1",
            )
        )
        assert any("o (Observation): declared dims (5,)" in v for v in caught), caught

    def test_a_policy_vector_with_no_control_count_is_reported(self) -> None:
        """A PolicyVector with no ``num_controls`` ranges over an unpinned space."""
        spec = single_factor_spec(
            extra_state="\npi[3,1,type=float]",
            extra_ontology="\npi=PolicyVector",
        )
        violations = _check_declaration_coherence(spec)
        assert any("no positive num_controls" in v for v in violations), violations

    def test_a_mis_sized_policy_vector_is_reported(self) -> None:
        """``pi[4,1]`` against ``num_controls: 3`` contradicts the spec itself."""
        spec = single_factor_spec(
            extra_state="\npi[4,1,type=float]",
            extra_ontology="\npi=PolicyVector",
            parameters="num_hidden_states: 3\nnum_obs: 3\nnum_controls: 3",
        )
        violations = _check_declaration_coherence(spec)
        assert any(
            "pi (PolicyVector): declared dims (4,)" in v and "requires (3,)" in v
            for v in violations
        ), violations

    def test_an_integer_belief_is_reported(self) -> None:
        """A HiddenState holds a probability; ``type=int`` declares an index.
        The int-role direction has its own control in test_gnn_conformance."""
        spec = single_factor_spec(
            extra_state="\ns[3,1,type=int]",
            extra_ontology="\ns=HiddenState",
        )
        violations = _check_declaration_coherence(spec)
        assert any(
            "s (HiddenState) is declared type=int" in v and "which is float" in v
            for v in violations
        ), violations

    def test_two_time_assignments_are_reported(self) -> None:
        """Two ``Time=`` lines leave the model with two time indexes."""
        spec = single_factor_spec(
            extra_state="\nt[1,type=int]",
            extra_ontology="\nt=Time",
            time="Time=t\nTime=t",
        )
        violations = _check_declaration_coherence(spec)
        assert any("carries 2 'Time=<variable>' line(s)" in v for v in violations), violations

    def test_a_time_index_with_the_wrong_role_is_reported(self) -> None:
        """``Time=A`` where A is the RecognitionMatrix names a slot, not a clock."""
        spec = single_factor_spec(time="Time=A")
        violations = _check_declaration_coherence(spec)
        assert any(
            "Time=A but A is annotated 'RecognitionMatrix' rather than Time" in v
            for v in violations
        ), violations

    def test_a_policy_driving_an_uncontrolled_transition_is_reported(self) -> None:
        """``pi>B`` with no ``num_controls`` wires a choice among one action."""
        spec = single_factor_spec(
            extra_state="\ns[3,1,type=float]\no[3,1,type=int]\npi[3,1,type=float]",
            extra_ontology="\ns=HiddenState\no=Observation\npi=PolicyVector",
            connections="D>s\ns>A\nA>o\npi>B",
        )
        violations = _check_declaration_coherence(spec)
        assert any("A policy over one action is not a choice" in v for v in violations), violations

    def test_inference_configuration_without_a_control_count_still_checks_e(self) -> None:
        """A description omitting ``num_controls`` makes no claim about the
        action set — but the policy-prior/policy-count comparison must still
        run, or withholding one field would silence an unrelated check."""
        spec = single_factor_spec(
            parameters="num_hidden_states: 3\nnum_obs: 3\nnum_controls: 3",
        )
        violations = _check_inference_configuration(
            spec,
            num_controls=None,
            e_vector=np.full(4, 0.25),
            n_policies=3,
        )
        assert any(
            "has 4 entries but the agent enumerated" in v and "3 policies" in v for v in violations
        ), violations
        assert not any("was built with" in v for v in violations), (
            "num_controls=None must be 'no claim', not a fabricated comparison"
        )


# ── LIVE-OBJECT REFUSALS (ailm.gnn.live) ─────────────────────────────────────


class TestPrecisionCheck:
    """``gamma`` branches the shipped specs (which all conform) never hit."""

    def _spec_with_gamma(self, value: str) -> GnnSpec:
        return single_factor_spec(parameters=f"num_hidden_states: 3\nnum_obs: 3\ngamma: {value}")

    def test_a_non_positive_declared_gamma_is_reported(self) -> None:
        """A negative precision is not an operating point at all."""
        violations = _check_precision(self._spec_with_gamma("-2.0"), gamma=np.array([16.0]))
        assert violations == ["ModelParameters gamma=-2.0 is not a positive number."]

    def test_a_declared_gamma_with_no_live_value_is_unverified(self) -> None:
        """Withholding the live gamma must be a violation, not a silent pass."""
        violations = _check_precision(self._spec_with_gamma("16.0"), gamma=None)
        assert any("no live gamma was supplied" in v for v in violations), violations

    def test_an_empty_live_gamma_is_reported(self) -> None:
        """An empty array carries no precision to compare against."""
        violations = _check_precision(self._spec_with_gamma("16.0"), gamma=np.array([]))
        assert any("the live gamma is empty" in v for v in violations), violations

    @pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
    def test_nonfinite_declared_precision_is_rejected(self, value: str) -> None:
        spec = self._spec_with_gamma(value)
        violations = _check_precision(spec, gamma=np.array([16.0]))
        assert any("gamma=" in v and "finite" in v for v in violations), violations

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_nonfinite_live_precision_is_rejected(self, value: float) -> None:
        spec = self._spec_with_gamma("16.0")
        violations = _check_precision(spec, gamma=np.array([value]))
        assert any("live gamma" in v and "non-finite" in v for v in violations), violations

    def test_matching_finite_precision_passes(self) -> None:
        spec = self._spec_with_gamma("16.0")
        assert _check_precision(spec, gamma=np.array([16.0])) == []


class TestEmissionFactorCount:
    """The Connections-vs-live-A axis count, on graphs no shipped spec draws."""

    def test_a_graph_without_a_recognition_matrix_makes_no_claim(self) -> None:
        """With no RecognitionMatrix annotated there is no emission to count
        axes for; the check must skip rather than invent a comparison. The
        companion test below proves the same helper CAN fail, so this skip is
        not vacuous coverage of a dead function."""
        spec = parse_gnn_string(
            build_spec_text(
                state_space="B[3,3,type=float]\nD[3,1,type=float]\ns[3,1,type=float]",
                ontology="B=TransitionMatrix\nD=Prior\ns=HiddenState",
                connections="D>s\ns>B",
            ),
            origin="no_recognition.md",
        )
        assert _check_emission_factor_count(spec, np.full((3, 3), 1.0 / 3.0)) == []

    def test_a_recognition_matrix_wired_to_no_hidden_state_is_reported(self) -> None:
        """A graph whose emission touches only the observation declares an
        emission depending on no hidden state at all."""
        spec = single_factor_spec(
            extra_state="\ns[3,1,type=float]\no[3,1,type=int]",
            extra_ontology="\ns=HiddenState\no=Observation",
            connections="D>s\ns>B\nA>o",
        )
        violations = _check_emission_factor_count(spec, np.full((3, 3), 1.0 / 3.0))
        assert any(
            "no HiddenState is connected to the RecognitionMatrix" in v for v in violations
        ), violations


class TestMultifactorSlotRefusals:
    """Per-slot multifactor branches: missing arrays, wrong sizes, bad bindings.

    All run through :func:`check_conformance` so the dispatch to the
    multifactor path is exercised too, and each mutates exactly one thing away
    from the conformant baseline pinned above.
    """

    def test_a_missing_num_obs_is_reported(self) -> None:
        spec = two_factor_spec(
            parameters="num_factors: 2\nnum_hidden_states_f0: 2\nnum_hidden_states_f1: 2"
        )
        violations = check_conformance(spec, two_factor_live())
        assert any("missing 'num_obs'" in v for v in violations), violations

    def test_a_non_positive_num_obs_is_reported(self) -> None:
        spec = two_factor_spec(
            parameters="num_factors: 2\nnum_hidden_states_f0: 2\nnum_hidden_states_f1: 2\nnum_obs: 0"
        )
        violations = check_conformance(spec, two_factor_live())
        assert any("num_obs=0 is not a positive integer" in v for v in violations), violations

    def test_a_disagreeing_modality_count_is_reported(self) -> None:
        spec = two_factor_spec(
            parameters=(
                "num_factors: 2\nnum_hidden_states_f0: 2\nnum_hidden_states_f1: 2\n"
                "num_obs: 3\nnum_modalities: 2"
            )
        )
        violations = check_conformance(spec, two_factor_live())
        assert any("num_modalities=2 but the live model has 1" in v for v in violations), violations

    def test_two_transitions_binding_one_factor_is_reported(self) -> None:
        """``B_f0`` and an unsuffixed ``B_extra`` both bind factor 0, leaving
        factor 1 unpinned while the per-role COUNT is still right."""
        spec = two_factor_spec(
            state_space=(
                "A_m0[3,2,2,type=float]\n"
                "B_f0[2,2,type=float]\n"
                "B_extra[2,2,type=float]\n"
                "D_f0[2,1,type=float]\n"
                "D_f1[2,1,type=float]"
            ),
            ontology=(
                "A_m0=RecognitionMatrix\n"
                "B_f0=TransitionMatrix\n"
                "B_extra=TransitionMatrix\n"
                "D_f0=Prior\n"
                "D_f1=Prior"
            ),
        )
        violations = check_conformance(spec, two_factor_live())
        assert any(
            "B_extra and B_f0 both bind to factor 0" in v and "_f<N> suffix" in v
            for v in violations
        ), violations

    def test_a_transition_slot_with_no_live_array_is_unverified(self) -> None:
        violations = check_conformance(two_factor_spec(), two_factor_live(b_matrices=[np.eye(2)]))
        assert any(
            "B_f1: spec declares a TransitionMatrix for factor 1 but no live array was supplied"
            in v
            for v in violations
        ), violations

    def test_a_transition_contradicting_its_factor_size_is_reported(self) -> None:
        """``B_f0[3,3]`` under ``num_hidden_states_f0: 2``: the spec against itself."""
        spec = two_factor_spec(
            state_space=(
                "A_m0[3,2,2,type=float]\n"
                "B_f0[3,3,type=float]\n"
                "B_f1[2,2,type=float]\n"
                "D_f0[2,1,type=float]\n"
                "D_f1[2,1,type=float]"
            ),
        )
        violations = check_conformance(spec, two_factor_live())
        assert any(
            "B_f0 (TransitionMatrix, factor 0)" in v
            and "requires (2, 2). The spec contradicts itself" in v
            for v in violations
        ), violations

    def test_a_non_square_live_transition_is_reported(self) -> None:
        violations = check_conformance(
            two_factor_spec(),
            two_factor_live(b_matrices=[np.full((2, 3), 0.5), np.eye(2)]),
        )
        assert any("are not square in the state axes" in v for v in violations), violations

    def test_a_live_transition_of_the_wrong_size_is_reported(self) -> None:
        violations = check_conformance(
            two_factor_spec(),
            two_factor_live(b_matrices=[np.eye(3), np.eye(2)]),
        )
        assert any("live state dimension 3 != num_hidden_states_f0=2" in v for v in violations), (
            violations
        )

    def test_a_prior_slot_with_no_live_array_is_unverified(self) -> None:
        violations = check_conformance(two_factor_spec(), two_factor_live(d_vectors=None))
        assert any(
            "D_f0: spec declares a Prior for factor 0 but no live array was supplied" in v
            for v in violations
        ), violations

    def test_a_mis_sized_live_prior_is_reported(self) -> None:
        violations = check_conformance(
            two_factor_spec(),
            two_factor_live(d_vectors=[np.full(2, 0.5), np.full(3, 1.0 / 3.0)]),
        )
        assert any(
            "D_f1 (Prior, factor 1): live canonical dims (3,) != (num_hidden_states_f1,) = (2,)"
            in v
            for v in violations
        ), violations

    def test_a_recognition_slot_with_no_live_array_is_unverified(self) -> None:
        violations = check_conformance(two_factor_spec(), two_factor_live(a_matrices=None))
        assert any(
            "A_m0: spec declares a RecognitionMatrix for modality 0 but no live array was supplied"
            in v
            for v in violations
        ), violations

    def test_a_live_emission_not_indexed_by_observation_is_reported(self) -> None:
        violations = check_conformance(
            two_factor_spec(),
            two_factor_live(a_matrices=[np.full((4, 2, 2), 0.25)]),
        )
        assert any("do not start with num_obs=3" in v for v in violations), violations

    def test_a_live_emission_with_swapped_factor_axes_is_reported(self) -> None:
        """State axes (2, 3) against declared factor sizes (2, 2): a modality
        indexing its factors out of order is a different model."""
        violations = check_conformance(
            two_factor_spec(),
            two_factor_live(a_matrices=[np.full((3, 2, 3), 1.0 / 3.0)]),
        )
        joined = "\n".join(violations)
        assert "live state axes (2, 3) != the declared factor sizes (2, 2)" in joined, violations
        assert "declared dims (3, 2, 2) but the live canonical dims are (3, 2, 3)" in joined, (
            violations
        )

    def test_a_withheld_preference_vector_makes_no_claim_but_a_wrong_one_fails(self) -> None:
        """C with no live array is skipped (pymdp always has one; a description
        may omit it) — while the identical spec with a mis-sized live C is a
        violation, so the skip is a branch and not a blind spot."""
        spec = two_factor_spec(
            state_space=(
                "A_m0[3,2,2,type=float]\n"
                "B_f0[2,2,type=float]\n"
                "B_f1[2,2,type=float]\n"
                "C[3,1,type=float]\n"
                "D_f0[2,1,type=float]\n"
                "D_f1[2,1,type=float]"
            ),
            ontology=(
                "A_m0=RecognitionMatrix\n"
                "B_f0=TransitionMatrix\n"
                "B_f1=TransitionMatrix\n"
                "C=LogPreferenceVector\n"
                "D_f0=Prior\n"
                "D_f1=Prior"
            ),
        )
        withheld = check_conformance(spec, two_factor_live())
        assert not any("LogPreferenceVector" in v for v in withheld), withheld

        mis_sized = check_conformance(spec, two_factor_live(c_vectors=[np.zeros(4)]))
        assert any(
            "C (LogPreferenceVector): live canonical dims (4,) != (num_obs,) = (3,)" in v
            for v in mis_sized
        ), mis_sized

    def test_a_non_positive_control_count_is_reported(self) -> None:
        spec = two_factor_spec(
            parameters=(
                "num_factors: 2\nnum_hidden_states_f0: 2\nnum_hidden_states_f1: 2\n"
                "num_obs: 3\nnum_controls: 0"
            )
        )
        violations = check_conformance(spec, two_factor_live())
        assert any("num_controls=0 is not a positive integer" in v for v in violations), violations

    def test_a_declared_action_set_missing_from_the_live_transitions_is_reported(self) -> None:
        """``num_controls: 3`` with 2-axis live transitions: the action set is
        declared and the agent cannot take it. With the arrays withheld the
        same spec reports the slots unverified instead of crashing."""
        spec = two_factor_spec(
            parameters=(
                "num_factors: 2\nnum_hidden_states_f0: 2\nnum_hidden_states_f1: 2\n"
                "num_obs: 3\nnum_controls: 3"
            )
        )
        violations = check_conformance(spec, two_factor_live())
        assert any(
            "num_controls=3 but the live transition's canonical dims" in v for v in violations
        ), violations

        withheld = check_conformance(spec, two_factor_live(b_matrices=None))
        assert not any("live transition's canonical dims" in v for v in withheld), withheld
        assert any("no live array was supplied" in v for v in withheld), withheld


class TestMultifactorPolicyPrior:
    """The E slot of a factorized model: one policy space, three failure modes."""

    def _spec(self, e_dims: str = "3,1") -> GnnSpec:
        return two_factor_spec(
            state_space=(
                "A_m0[3,2,2,type=float]\n"
                "B_f0[2,2,type=float]\n"
                "B_f1[2,2,type=float]\n"
                "D_f0[2,1,type=float]\n"
                "D_f1[2,1,type=float]\n"
                f"E[{e_dims},type=float]"
            ),
            ontology=(
                "A_m0=RecognitionMatrix\n"
                "B_f0=TransitionMatrix\n"
                "B_f1=TransitionMatrix\n"
                "D_f0=Prior\n"
                "D_f1=Prior\n"
                "E=PolicyPrior"
            ),
            parameters=(
                "num_factors: 2\nnum_hidden_states_f0: 2\nnum_hidden_states_f1: 2\n"
                "num_obs: 3\nnum_controls: 3"
            ),
            parameterization="E=Uniform(3)",
        )

    def _live(self, **overrides: object) -> LiveConformanceInputs:
        """The baseline plus 3-action transitions, so only E can be at fault."""
        controlled = np.stack([np.eye(2)] * 3, axis=2)
        return two_factor_live(b_matrices=[controlled, controlled], **overrides)

    def test_a_policy_prior_sized_off_the_action_set_is_reported(self) -> None:
        violations = check_conformance(
            self._spec(e_dims="4,1"), self._live(e_vector=np.full(4, 0.25))
        )
        assert any(
            "E (PolicyPrior): declared dims (4,)" in v and "num_controls=3 requires" in v
            for v in violations
        ), violations

    def test_a_withheld_policy_prior_is_unverified_not_ignored(self) -> None:
        violations = check_conformance(self._spec(), self._live())
        assert any(
            "E: spec declares a PolicyPrior but no live E was supplied" in v for v in violations
        ), violations
        # And withholding E must also mute its structural form rather than
        # crash on a missing array: no Uniform violation may be fabricated.
        assert not any("Uniform" in v for v in violations), violations

    def test_a_live_policy_prior_of_the_wrong_size_is_reported(self) -> None:
        violations = check_conformance(self._spec(), self._live(e_vector=np.full(4, 0.25)))
        assert any(
            "E (PolicyPrior): declared dims (3,) but the live E has canonical dims (4,)" in v
            for v in violations
        ), violations

    def test_a_structural_form_on_a_belief_variable_is_inert(self) -> None:
        """A parameterization whose variable carries no matrix role binds to no
        live slot; it must be skipped, not crash indexing ``slot_arrays``."""
        spec = two_factor_spec(
            state_space=(
                "A_m0[3,2,2,type=float]\n"
                "B_f0[2,2,type=float]\n"
                "B_f1[2,2,type=float]\n"
                "D_f0[2,1,type=float]\n"
                "D_f1[2,1,type=float]\n"
                "s_a[2,1,type=float]\n"
                "s_b[2,1,type=float]"
            ),
            ontology=(
                "A_m0=RecognitionMatrix\n"
                "B_f0=TransitionMatrix\n"
                "B_f1=TransitionMatrix\n"
                "D_f0=Prior\n"
                "D_f1=Prior\n"
                "s_a=HiddenState\n"
                "s_b=HiddenState"
            ),
            parameterization="s_a=Uniform(2)",
        )
        violations = check_conformance(spec, two_factor_live())
        assert violations == []


class TestStructuralFormEdges:
    """Branches of the form checker that only a mis-sized array can reach."""

    @pytest.mark.parametrize(
        "form,args,array",
        [
            ("Identity", "2", np.eye(2)),
            ("Uniform", "2", np.full(2, 0.5)),
            ("Zeros", "2", np.zeros(2)),
            ("Learned", "normalization=column-stochastic-per-column", np.full((2, 2), 0.5)),
        ],
    )
    @pytest.mark.parametrize("value", [None, np.nan, np.inf, -np.inf])
    def test_structural_forms_require_finite_arrays(
        self, form: str, args: str, array: np.ndarray, value: float | None
    ) -> None:
        candidate = array.copy()
        if value is not None:
            candidate.flat[0] = value
        violations = _check_structural_form("X", form, args, candidate)
        if value is None:
            assert violations == []
        else:
            assert any("X:" in v and "non-finite" in v for v in violations), violations

    def test_a_uniform_declaration_over_the_wrong_size_is_reported(self) -> None:
        """``Uniform(4)`` against 3 entries: the size, not the values, is wrong."""
        violations = _check_structural_form("D", "Uniform", "4", np.full(3, 1.0 / 3.0))
        assert violations == ["D: declared Uniform(4) but the live array holds 3 entries."]

    def test_a_non_square_learned_array_skips_the_identity_probe_not_the_claims(self) -> None:
        """The identity degeneracy test only applies to square matrices, but a
        non-square Learned array must still have its normalization checked —
        skipping both would leave a G2P-shaped emission entirely unread."""
        violations = _check_structural_form(
            "A",
            "Learned",
            "normalization=column-stochastic-per-column",
            np.full((2, 3), 1.0),  # axis-0 sums are 2, not 1
        )
        assert any("axis-0 sums deviate from 1" in v for v in violations), violations
        assert not any("identity" in v.lower() for v in violations), violations

    def test_an_unknown_form_is_inert_here_because_the_parser_is_the_gate(self) -> None:
        """The parser rejects unknown structural forms before they reach the
        checker (see test_unknown_structural_form_raises), so an unknown form
        arriving here makes no claim. This pins that seam: if the parser's
        whitelist ever loosens, this documents which layer must refuse."""
        assert _check_structural_form("X", "Mystery", "3", np.ones(3)) == []

    def test_a_sums_to_one_claim_is_checked_against_the_whole_array(self) -> None:
        """``sums-to-one-*`` is a claim about the global sum; an array summing
        to 3 under it is a per-column table in a prior's slot."""
        violations = _check_learned_arguments("D", "normalization=sums-to-one-global", np.ones(3))
        assert any(
            "declared normalization=sums-to-one-global" in v and "sums to 3" in v
            for v in violations
        ), violations
        # Negative control: a genuinely global distribution passes.
        assert (
            _check_learned_arguments("D", "normalization=sums-to-one-global", np.full(4, 0.25))
            == []
        )
