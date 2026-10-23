"""Tests for source-faithful CEREBRUM roles, crosswalks, and free energies."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ailm.cases import (
    ADPOSITION_TO_CASE,
    AILM_PYMDP_HEURISTIC_CROSSWALK,
    AILM_SCALAR_CASE_WEIGHTS,
    CEREBRUM_CASE_FORMULATIONS,
    CEREBRUM_FUNCTION_ABBREVIATIONS,
    CEREBRUM_POMDP_SYMBOLS,
    PYMDP_COMPONENTS_WITHOUT_CEREBRUM_TUPLE_ANALOGUE,
    UD_DEPREL_TO_CASE,
    CaseRole,
    CerebrumCaseFormulationKind,
    CerebrumFunctionRole,
    case_conditioned_free_energy,
    expected_free_energy_transition,
    transformation_cost,
    validate_ailm_pymdp_crosswalk,
    validate_cerebrum_function_roles,
)


class TestCaseRoleEnum:
    def test_has_eight_core_cases(self):
        assert len(CaseRole) == 8

    def test_all_case_names(self):
        expected = {"NOM", "ACC", "GEN", "DAT", "INS", "LOC", "ABL", "VOC"}
        assert {c.name for c in CaseRole} == expected

    def test_gen_is_genitive(self):
        assert CaseRole.GEN.value == "Genitive"

    def test_nom_is_nominative(self):
        assert CaseRole.NOM.value == "Nominative"

    def test_all_have_string_values(self):
        assert all(isinstance(c.value, str) for c in CaseRole)


class TestCerebrumFunctionRole:
    def test_retains_core_inventory_and_separate_novel_generative_role(self):
        assert len(CerebrumFunctionRole) == len(CaseRole) + 1

    def test_core_genitive_exists(self):
        role = CerebrumFunctionRole.CORE_GENITIVE
        assert role.value == "cerebrum:core:GEN:RelationalSource"

    def test_novel_generative_exists_separately(self):
        role = CerebrumFunctionRole.NOVEL_GENERATIVE
        assert role.value == "cerebrum:novel:GEN:OutputGenerator"

    def test_all_have_string_values(self):
        assert all(isinstance(r.value, str) for r in CerebrumFunctionRole)


class TestRoleSystemsAreUnambiguous:
    """The source abbreviation collision must survive without serialization collision."""

    def test_case_role_gen_is_linguistic_genitive(self):
        assert CaseRole.GEN.value == "Genitive"

    def test_source_gen_abbreviation_is_preserved_for_both_distinct_roles(self):
        gen_roles = {
            role
            for role, abbreviation in CEREBRUM_FUNCTION_ABBREVIATIONS.items()
            if abbreviation == "GEN"
        }
        assert gen_roles == {
            CerebrumFunctionRole.CORE_GENITIVE,
            CerebrumFunctionRole.NOVEL_GENERATIVE,
        }

    def test_values_are_disjoint_across_the_two_systems(self):
        """A role's value string identifies exactly one of the two role systems."""
        case_values = {c.value for c in CaseRole}
        function_values = {r.value for r in CerebrumFunctionRole}
        shared = case_values & function_values
        assert not shared, (
            f"Value strings shared by CaseRole and CerebrumFunctionRole: {shared}. "
            f"A shared value makes a serialised role ambiguous."
        )

    def test_enums_are_distinct_types(self):
        assert type(CaseRole.GEN) is not type(CerebrumFunctionRole.CORE_GENITIVE)

    def test_serialized_function_roles_are_unique(self):
        serialized = [role.value for role in CerebrumFunctionRole]
        assert len(serialized) == len(set(serialized))

    def test_validator_rejects_collapsing_core_genitive(self):
        """POSITIVE CONTROL: the role validator must fail on the historical omission."""
        planted = dict(CEREBRUM_FUNCTION_ABBREVIATIONS)
        planted.pop(CerebrumFunctionRole.CORE_GENITIVE)
        with pytest.raises(ValueError, match="incomplete"):
            validate_cerebrum_function_roles(planted)


class TestCerebrumPymdpHeuristic:
    def test_source_tuple_preserves_every_supplement_symbol(self):
        assert tuple(CEREBRUM_POMDP_SYMBOLS) == ("S", "A", "T", "Ω", "O", "F", "π")
        assert CEREBRUM_POMDP_SYMBOLS["A"] == "action space"
        assert CEREBRUM_POMDP_SYMBOLS["O"].startswith("likelihood mapping")

    def test_heuristic_is_source_symbol_complete(self):
        assert set(AILM_PYMDP_HEURISTIC_CROSSWALK) == set(CEREBRUM_POMDP_SYMBOLS)

    def test_likelihood_and_transition_correspondences_are_explicit(self):
        assert AILM_PYMDP_HEURISTIC_CROSSWALK["O"].pymdp_component == "A"
        assert AILM_PYMDP_HEURISTIC_CROSSWALK["T"].pymdp_component == "B"

    def test_source_action_space_is_not_pymdp_likelihood_a(self):
        assert AILM_PYMDP_HEURISTIC_CROSSWALK["A"].pymdp_component != "A"

    def test_vfe_and_precision_are_not_g_and_e(self):
        assert AILM_PYMDP_HEURISTIC_CROSSWALK["F"].caveat == "not pymdp G"
        assert AILM_PYMDP_HEURISTIC_CROSSWALK["π"].caveat == "not pymdp E"

    def test_c_d_e_have_no_direct_source_tuple_analogue(self):
        assert {"C", "D", "E"} == PYMDP_COMPONENTS_WITHOUT_CEREBRUM_TUPLE_ANALOGUE

    def test_validator_rejects_retired_precision_to_e_mapping(self):
        """POSITIVE CONTROL: the exact historical pi-to-E error must fail."""
        planted = dict(AILM_PYMDP_HEURISTIC_CROSSWALK)
        planted["π"] = dataclasses.replace(planted["π"], pymdp_component="E", caveat="")
        with pytest.raises(ValueError, match="miswired"):
            validate_ailm_pymdp_crosswalk(planted, PYMDP_COMPONENTS_WITHOUT_CEREBRUM_TUPLE_ANALOGUE)

    def test_validator_rejects_negated_and_swapped_transition_likelihood_mappings(self):
        planted_negation = dict(AILM_PYMDP_HEURISTIC_CROSSWALK)
        planted_negation["T"] = dataclasses.replace(
            planted_negation["T"], caveat="not pymdp B; actually likelihood"
        )
        with pytest.raises(ValueError, match="miswired"):
            validate_ailm_pymdp_crosswalk(
                planted_negation, PYMDP_COMPONENTS_WITHOUT_CEREBRUM_TUPLE_ANALOGUE
            )

        planted_swap = dict(AILM_PYMDP_HEURISTIC_CROSSWALK)
        planted_swap["T"], planted_swap["O"] = planted_swap["O"], planted_swap["T"]
        with pytest.raises(ValueError, match="miswired"):
            validate_ailm_pymdp_crosswalk(
                planted_swap, PYMDP_COMPONENTS_WITHOUT_CEREBRUM_TUPLE_ANALOGUE
            )


class TestTransformationCost:
    def test_zero_cost_identical_distributions(self):
        q = np.array([0.5, 0.3, 0.2])
        assert transformation_cost(q, q, target_free_energy=1.25) == pytest.approx(1.25)

    def test_positive_cost_different_distributions(self):
        q_src = np.array([0.5, 0.3, 0.2])
        q_tgt = np.array([0.2, 0.3, 0.5])
        assert transformation_cost(q_src, q_tgt, target_free_energy=0.0) > 0

    def test_asymmetric(self):
        """KL divergence is asymmetric: KL(q_tgt||q_src) != KL(q_src||q_tgt) in general.

        Use distributions that are highly skewed in opposite directions so the
        asymmetry is large and not an accident of floating-point coincidence.
        """
        # Use a genuinely asymmetric pair; mirrored vectors would have equal KL
        # in both directions and would be a vacuous test of asymmetry.
        q_e = np.array([0.9, 0.08, 0.02])
        q_f = np.array([0.3, 0.5, 0.2])
        cost_ef = transformation_cost(q_e, q_f, target_free_energy=0.0)
        cost_fe = transformation_cost(q_f, q_e, target_free_energy=0.0)
        # These are not symmetric, so KL must differ
        assert cost_ef != pytest.approx(cost_fe, abs=1e-6), (
            f"KL(q_e||q_f)={cost_ef:.6f} should differ from KL(q_f||q_e)={cost_fe:.6f}"
        )

    def test_raises_on_unnormalized_src(self):
        with pytest.raises(ValueError, match="q_src"):
            transformation_cost(
                np.array([0.5, 0.5, 0.5]),
                np.array([1 / 3] * 3),
                target_free_energy=0.0,
            )

    def test_raises_on_unnormalized_tgt(self):
        with pytest.raises(ValueError, match="q_tgt"):
            transformation_cost(
                np.array([1 / 3] * 3),
                np.array([0.5, 0.5, 0.5]),
                target_free_energy=0.0,
            )

    def test_raises_on_shape_mismatch(self):
        with pytest.raises(ValueError):
            transformation_cost(
                np.array([0.5, 0.5]),
                np.array([1 / 3] * 3),
                target_free_energy=0.0,
            )

    def test_target_free_energy_is_added_without_scaling_kl(self):
        q_src = np.array([0.5, 0.3, 0.2])
        q_tgt = np.array([0.2, 0.3, 0.5])
        kl = float(np.sum(q_tgt * np.log(q_tgt / q_src)))
        assert transformation_cost(q_src, q_tgt, target_free_energy=2.0) == pytest.approx(2.0 + kl)

    def test_expected_free_energy_transition_is_the_separate_scaled_equation(self):
        q_src = np.array([0.5, 0.3, 0.2])
        q_tgt = np.array([0.2, 0.3, 0.5])
        kl = float(np.sum(q_tgt * np.log(q_tgt / q_src)))
        vfe_transition = transformation_cost(q_src, q_tgt, target_free_energy=2.0)
        efe_transition = expected_free_energy_transition(
            q_src, q_tgt, base_expected_free_energy=2.0, gamma=3.0
        )
        assert vfe_transition == pytest.approx(2.0 + kl)
        assert efe_transition == pytest.approx(2.0 + 3.0 * kl)
        assert efe_transition != pytest.approx(vfe_transition)

    def test_expected_free_energy_requires_positive_gamma(self):
        q = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="gamma"):
            expected_free_energy_transition(q, q, base_expected_free_energy=0.0, gamma=0.0)

    def test_support_mismatch_produces_infinite_kl(self):
        q_src = np.array([1.0, 0.0])
        q_tgt = np.array([0.5, 0.5])
        assert np.isinf(transformation_cost(q_src, q_tgt, target_free_energy=0.0))


class TestCaseFreeEnergySourceFidelity:
    def test_source_formulations_cover_every_core_case(self):
        assert set(CEREBRUM_CASE_FORMULATIONS) == set(CaseRole)

    def test_only_nom_and_acc_are_scalar_reweightings(self):
        kinds = {role: formulation.kind for role, formulation in CEREBRUM_CASE_FORMULATIONS.items()}
        assert kinds[CaseRole.NOM] is CerebrumCaseFormulationKind.LIKELIHOOD_WEIGHT
        assert kinds[CaseRole.ACC] is CerebrumCaseFormulationKind.COMPLEXITY_WEIGHT
        assert all(
            kinds[role] is CerebrumCaseFormulationKind.AUGMENTED_LATENT
            for role in set(CaseRole) - {CaseRole.NOM, CaseRole.ACC}
        )

    def test_augmented_cases_name_their_distinct_auxiliary_variables(self):
        expected = {
            CaseRole.GEN: "r",
            CaseRole.DAT: "g",
            CaseRole.INS: "a",
            CaseRole.LOC: "e",
            CaseRole.ABL: "o_origin",
            CaseRole.VOC: "c",
        }
        assert {
            role: CEREBRUM_CASE_FORMULATIONS[role].auxiliary_symbol for role in expected
        } == expected

    def test_ailm_numeric_defaults_are_owned_only_for_nom_and_acc(self):
        assert set(AILM_SCALAR_CASE_WEIGHTS) == {CaseRole.NOM, CaseRole.ACC}
        assert AILM_SCALAR_CASE_WEIGHTS[CaseRole.NOM].alpha > 1.0
        assert AILM_SCALAR_CASE_WEIGHTS[CaseRole.ACC].beta < 1.0
        assert all(
            weights.alpha > 0 and weights.beta > 0 for weights in AILM_SCALAR_CASE_WEIGHTS.values()
        )


class TestCaseFreeEnergy:
    def setup_method(self):
        self.n = 4
        self.q = np.array([0.4, 0.3, 0.2, 0.1])  # valid posterior
        self.log_ll = np.log(np.array([0.5, 0.3, 0.15, 0.05]))

    def test_rejects_finite_but_unnormalized_log_prior(self):
        """NEGATIVE CONTROL: a shifted log measure is not a probability prior."""
        with pytest.raises(ValueError, match="normalized log-probability"):
            case_conditioned_free_energy(
                np.array([0.5, 0.5]),
                np.log(np.array([0.5, 0.5])),
                CaseRole.NOM,
                log_prior=np.array([0.0, 0.0]),
            )

    @pytest.mark.parametrize(
        "positive_log_likelihood",
        (np.array([0.0, 0.2]), np.array([100.0, 100.0])),
    )
    def test_rejects_positive_discrete_log_likelihoods(self, positive_log_likelihood):
        """NEGATIVE CONTROL: log probability mass cannot encode p(o|s) > 1."""

        with pytest.raises(ValueError, match="log-probability masses"):
            case_conditioned_free_energy(
                np.array([0.5, 0.5]), positive_log_likelihood, CaseRole.NOM
            )

    def test_accepts_a_normalized_nonuniform_log_prior(self):
        q = np.array([0.5, 0.5])
        likelihood = np.array([0.7, 0.3])
        prior = np.array([0.8, 0.2])
        expected_kl = float(np.sum(q * (np.log(q) - np.log(prior))))
        expected_ll = float(np.dot(q, np.log(likelihood)))
        expected = expected_kl - AILM_SCALAR_CASE_WEIGHTS[CaseRole.NOM].alpha * expected_ll
        assert case_conditioned_free_energy(
            q,
            np.log(likelihood),
            CaseRole.NOM,
            log_prior=np.log(prior),
        ) == pytest.approx(expected)

    def test_zero_probability_mass_off_posterior_support_is_well_defined(self):
        q = np.array([1.0, 0.0])
        observed = case_conditioned_free_energy(
            q,
            np.array([np.log(0.5), -np.inf]),
            CaseRole.NOM,
            log_prior=np.array([0.0, -np.inf]),
        )
        expected = -AILM_SCALAR_CASE_WEIGHTS[CaseRole.NOM].alpha * np.log(0.5)
        assert observed == pytest.approx(expected)

    def test_zero_mass_on_posterior_support_returns_infinite_vfe(self):
        q = np.array([0.5, 0.5])
        zero_likelihood = case_conditioned_free_energy(
            q,
            np.array([np.log(0.5), -np.inf]),
            CaseRole.NOM,
        )
        zero_prior = case_conditioned_free_energy(
            q,
            np.log(np.array([0.5, 0.5])),
            CaseRole.NOM,
            log_prior=np.array([0.0, -np.inf]),
        )
        assert zero_likelihood == float("inf")
        assert zero_prior == float("inf")

    @pytest.mark.parametrize("invalid", (np.array([np.nan, -1.0]), np.array([np.inf, -1.0])))
    def test_log_inputs_reject_nan_and_positive_infinity(self, invalid):
        q = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match=r"not NaN or \+inf"):
            case_conditioned_free_energy(q, invalid, CaseRole.NOM)
        with pytest.raises(ValueError, match=r"not NaN or \+inf"):
            case_conditioned_free_energy(
                q,
                np.log(np.array([0.5, 0.5])),
                CaseRole.NOM,
                log_prior=invalid,
            )

    def test_log_input_shape_diagnostics_are_separate_from_value_diagnostics(self):
        with pytest.raises(ValueError, match="log_likelihood must have.*shape"):
            case_conditioned_free_energy(np.array([0.5, 0.5]), np.array([[0.0, 0.0]]), CaseRole.NOM)
        with pytest.raises(ValueError, match="log_prior must have.*shape"):
            case_conditioned_free_energy(
                np.array([0.5, 0.5]),
                np.log(np.array([0.5, 0.5])),
                CaseRole.NOM,
                log_prior=np.array([[0.0, 0.0]]),
            )

    def test_returns_float(self):
        result = case_conditioned_free_energy(self.q, self.log_ll, CaseRole.NOM)
        assert isinstance(result, float)

    def test_nom_different_from_acc(self):
        """Different case roles produce different free energies (α, β differ)."""
        fe_nom = case_conditioned_free_energy(self.q, self.log_ll, CaseRole.NOM)
        fe_acc = case_conditioned_free_energy(self.q, self.log_ll, CaseRole.ACC)
        assert fe_nom != pytest.approx(fe_acc, rel=1e-6)

    def test_raises_on_unnormalized_q(self):
        bad_q = np.array([0.5, 0.5, 0.5, 0.5])
        with pytest.raises(ValueError):
            case_conditioned_free_energy(bad_q, self.log_ll, CaseRole.NOM)

    def test_uniform_q_finite(self):
        uniform = np.ones(self.n) / self.n
        log_ll = np.full(self.n, -np.log(self.n))
        result = case_conditioned_free_energy(uniform, log_ll, CaseRole.ACC)
        assert np.isfinite(result)

    @pytest.mark.parametrize(
        "role",
        (CaseRole.GEN, CaseRole.DAT, CaseRole.INS, CaseRole.LOC, CaseRole.ABL, CaseRole.VOC),
    )
    def test_non_scalar_source_formulations_fail_closed(self, role):
        with pytest.raises(NotImplementedError, match="auxiliary latent state"):
            case_conditioned_free_energy(self.q, self.log_ll, role)

    def test_scalar_equation_matches_an_independent_calculation(self):
        role = CaseRole.NOM
        weights = AILM_SCALAR_CASE_WEIGHTS[role]
        log_prior = np.full(self.n, -np.log(self.n))
        expected_kl = float(np.sum(self.q * (np.log(self.q) - log_prior)))
        expected_ll = float(np.dot(self.q, self.log_ll))
        expected = weights.beta * expected_kl - weights.alpha * expected_ll
        observed = case_conditioned_free_energy(self.q, self.log_ll, role, log_prior=log_prior)
        assert observed == pytest.approx(expected)

    @pytest.mark.parametrize(
        "bad_q",
        (
            np.array([]),
            np.array([[0.5, 0.5]]),
            np.array([0.5, np.nan, 0.5]),
            np.array([1.1, -0.1]),
            np.array([0.5, 0.5000011]),
        ),
    )
    def test_invalid_probability_vectors_fail_closed(self, bad_q):
        with pytest.raises(ValueError, match="q"):
            case_conditioned_free_energy(bad_q, np.zeros(bad_q.size), CaseRole.NOM)


class TestUdDeprelMapping:
    def test_nsubj_is_nom(self):
        assert UD_DEPREL_TO_CASE["nsubj"] == CaseRole.NOM

    def test_obj_is_acc(self):
        assert UD_DEPREL_TO_CASE["obj"] == CaseRole.ACC

    def test_iobj_is_dat(self):
        assert UD_DEPREL_TO_CASE["iobj"] == CaseRole.DAT

    def test_nmod_poss_is_gen(self):
        assert UD_DEPREL_TO_CASE["nmod:poss"] == CaseRole.GEN

    def test_vocative_is_voc(self):
        assert UD_DEPREL_TO_CASE["vocative"] == CaseRole.VOC

    def test_adposition_to_is_dat(self):
        assert ADPOSITION_TO_CASE["to"] == CaseRole.DAT

    def test_adposition_with_is_ins(self):
        assert ADPOSITION_TO_CASE["with"] == CaseRole.INS

    def test_adposition_from_is_abl(self):
        assert ADPOSITION_TO_CASE["from"] == CaseRole.ABL
