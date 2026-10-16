"""Fillmore's subject-selection hierarchy, and the circularity it can hide.

The rule is external to the L4 model: it predicts which token bears the UD
``nsubj`` relation, a label the model never observes. That makes it a real check
on a case assignment — and it makes the *derived* assignment a trap, because
this project's heuristic maps ``nsubj`` to NOM directly, so scoring the derived
path against ``nsubj`` is scoring a label against its own relabeling.
"""

from __future__ import annotations

import pytest

from ailm.cases import UD_DEPREL_TO_CASE, CaseRole
from ailm.levels.case import (
    FILLMORE_SUBJECT_PRIORITY,
    fillmore_subject,
    subject_prediction_accuracy,
)


class TestTheRule:
    def test_the_agentive_wins_when_present(self) -> None:
        assert fillmore_subject([CaseRole.ACC, CaseRole.INS, CaseRole.NOM]) == 2

    def test_the_instrumental_wins_without_an_agentive(self) -> None:
        assert fillmore_subject([CaseRole.ACC, CaseRole.INS]) == 1

    def test_the_objective_is_the_last_resort(self) -> None:
        assert fillmore_subject([CaseRole.LOC, CaseRole.ACC]) == 1

    def test_no_role_in_the_hierarchy_refuses_rather_than_defaulting(self) -> None:
        """A rule that always names a token cannot be told apart from a
        positional baseline, because both would be right exactly as often as the
        subject happens to sit where the default points."""
        assert fillmore_subject([CaseRole.LOC, CaseRole.GEN, None]) is None
        assert fillmore_subject([]) is None

    def test_ties_go_left_and_that_is_a_convention(self) -> None:
        assert fillmore_subject([CaseRole.NOM, CaseRole.NOM]) == 0

    def test_the_priority_order_is_fillmores_and_not_the_inventory_order(self) -> None:
        """The hierarchy is a claim from the source, not a re-listing of the enum.

        If it were sorted by the inventory it would carry no content, and a
        change to the enum's declaration order would silently change the rule.
        """
        assert FILLMORE_SUBJECT_PRIORITY == (CaseRole.NOM, CaseRole.INS, CaseRole.ACC)
        assert (
            list(FILLMORE_SUBJECT_PRIORITY)
            != [role for role in CaseRole if role in FILLMORE_SUBJECT_PRIORITY]
            or len(FILLMORE_SUBJECT_PRIORITY) < 3
        ), (
            "the priority order coincides with the enum's order, so this test "
            "cannot tell the two apart; assert the source's order explicitly"
        )


class TestScoring:
    def test_a_perfect_assignment_scores_one(self) -> None:
        sentences = [[(CaseRole.NOM, "nsubj"), (CaseRole.ACC, "obj")]]
        result = subject_prediction_accuracy(sentences)
        assert (result["accuracy"], result["coverage"], result["correct"]) == (1.0, 1.0, 1)

    def test_a_wrong_assignment_scores_zero_rather_than_being_skipped(self) -> None:
        sentences = [[(CaseRole.ACC, "nsubj"), (CaseRole.NOM, "obj")]]
        result = subject_prediction_accuracy(sentences)
        assert result["accuracy"] == 0.0
        assert result["predicted"] == 1, "a wrong prediction must still count as a prediction"

    def test_refusals_lower_coverage_and_leave_accuracy_alone(self) -> None:
        """Reported separately, because a rule that refuses often looks accurate."""
        sentences = [
            [(CaseRole.NOM, "nsubj"), (CaseRole.ACC, "obj")],
            [(CaseRole.LOC, "nsubj"), (CaseRole.GEN, "obj")],
        ]
        result = subject_prediction_accuracy(sentences)
        assert result["accuracy"] == 1.0
        assert result["coverage"] == 0.5
        assert result["sentences"] == 2 and result["predicted"] == 1

    @pytest.mark.parametrize(
        "deprels", [["obj", "obl"], ["nsubj", "nsubj"]], ids=["no-subject", "two-subjects"]
    )
    def test_sentences_without_exactly_one_subject_are_not_scored(self, deprels) -> None:
        """Selection and detection are different questions, and mixing them would
        let a rule be credited for finding a subject that was never in doubt."""
        sentences = [[(CaseRole.NOM, deprel) for deprel in deprels]]
        assert subject_prediction_accuracy(sentences)["sentences"] == 0


def test_the_derived_path_is_circular_and_the_test_suite_says_so() -> None:
    """THE POINT OF THE CONTROL, asserted rather than described.

    ``UD_DEPREL_TO_CASE`` maps ``nsubj`` to NOM. So on the derived assignment,
    the token bearing ``nsubj`` is *by construction* the token bearing NOM, and
    the hierarchy's first clause selects it every time. The derived score is
    therefore near-perfect for a reason that has nothing to do with the rule
    being right, which is exactly why the model path is the number to read.

    If this mapping is ever changed, the derived path stops being a pure
    circularity and this control needs rewriting rather than re-running.
    """
    assert UD_DEPREL_TO_CASE["nsubj"] is CaseRole.NOM

    derived = [
        [(UD_DEPREL_TO_CASE.get(deprel), deprel) for deprel in sentence]
        for sentence in (["nsubj", "obj"], ["obj", "nsubj"], ["nsubj", "obl"])
    ]
    result = subject_prediction_accuracy(derived)
    assert result["accuracy"] == 1.0, (
        "the derived path did not score perfectly, so either the deprel mapping "
        "changed or the hierarchy no longer selects NOM first; the control is "
        "not measuring what it claims"
    )


def test_the_model_path_cannot_see_the_target() -> None:
    """The target must stay outside the observation alphabet, or the whole
    experiment is the circularity above with more steps."""
    from ailm.levels.case import FORBIDDEN_OBSERVATIONS

    assert "deprel" in FORBIDDEN_OBSERVATIONS


class TestPerTokenRerunRule:
    """Decision 42's registered per-token rule, tested before it is implemented.

    The old rule dropped a whole sentence when ANY token carried a tag outside
    the L4 observation alphabet, leaving nine scorable sentences — no sample at
    all. The registered fix scores per TOKEN over the sentences whose subject
    token and candidate set are entirely inside the alphabet.
    """

    def make_module(self):
        from scripts.fillmore_subject_rerun import (
            leftmost_role_choice,
            per_token_aggregate,
        )

        return leftmost_role_choice, per_token_aggregate

    def test_an_out_of_alphabet_tag_elsewhere_no_longer_drops_the_sentence(self) -> None:
        """THE defect of the retired rule, asserted absent.

        Sentence: [DET NOUN] with DET out of alphabet. Under the whole-sentence
        drop rule nothing remains; under the registered rule the subject-bearing
        candidate pair survives untouched.
        """
        from scripts.fillmore_subject_rerun import sentence_is_alphabet_complete

        roles = {1: CaseRole.NOM, 2: CaseRole.ACC}
        upos = {0: "DET", 1: "PRON", 2: "NOUN"}
        # DET (index 0) carries no role: the per-token rule scores around it,
        # so the sentence is kept (the retired whole-sentence rule dropped it):
        assert sentence_is_alphabet_complete(upos, roles, alphabet={"PRON", "NOUN"}) is True
        # The explicit-subject form also keeps it: the stray DET is not a candidate:
        assert sentence_is_alphabet_complete(
            upos, roles, alphabet={"PRON", "NOUN"}, subject_index=1
        )

    def test_a_candidate_outside_the_alphabet_still_excludes_the_sentence(self) -> None:
        from scripts.fillmore_subject_rerun import sentence_is_alphabet_complete

        upos = {0: "NOUN", 1: "X"}
        roles = {0: CaseRole.NOM, 1: CaseRole.ACC}
        assert not sentence_is_alphabet_complete(
            upos, roles, alphabet={"NOUN", "X"} - {"X"}, subject_index=0
        )

    def test_the_subject_itself_must_be_in_the_alphabet(self) -> None:
        from scripts.fillmore_subject_rerun import sentence_is_alphabet_complete

        upos = {0: "PART", 1: "NOUN"}
        roles = {1: CaseRole.NOM}
        assert not sentence_is_alphabet_complete(upos, roles, alphabet={"NOUN"}, subject_index=0)

    def test_leftmost_role_baseline_refuses_without_any_role(self) -> None:
        leftmost_role_choice, _ = self.make_module()
        assert leftmost_role_choice([None, None]) is None

    def test_leftmost_role_baseline_names_the_first_role_bearer(self) -> None:
        leftmost_role_choice, _ = self.make_module()
        assert leftmost_role_choice([None, CaseRole.INS, CaseRole.NOM]) == 1

    def test_per_token_aggregation_refusals_lower_coverage_not_accuracy(self) -> None:
        _, per_token_aggregate = self.make_module()
        results = [
            {"correct": True},  # scored, predicted, correct
            {"correct": False},  # scored, predicted, wrong
            None,  # refusal: scored, not predicted
        ]
        agg = per_token_aggregate(results)
        assert agg["scored"] == 3
        assert agg["predicted"] == 2
        assert agg["correct"] == 1
        assert agg["accuracy"] == 0.5
        assert agg["coverage"] == round(2 / 3, 4)

    def test_per_token_aggregate_of_nothing_is_defined_as_zero_not_crash(self) -> None:
        """Mirrors the registered refusal vocabulary: empty input reports zeros."""
        _, per_token_aggregate = self.make_module()
        agg = per_token_aggregate([])
        assert agg["scored"] == 0 and agg["accuracy"] == 0.0 and agg["coverage"] == 0.0

    def test_entry_bar_predicate_compares_predicted_count_to_minimum_pairs(self) -> None:
        from scripts.fillmore_subject_rerun import entry_bar_verdict

        assert entry_bar_verdict(predicted=145, minimum_pairs=420) == (
            False,
            "unresolvable",
        )
        assert entry_bar_verdict(predicted=420, minimum_pairs=420)[0] is True


def test_the_registered_configuration_reproduces_the_documented_counts() -> None:
    """Research-queue evidence for the decision-42 re-run, on the shipped corpus.

    On ALL of EWT dev under the registered rule, 702 sentences have exactly one
    nsubj with subject token and every role-bearing candidate inside the 12-tag
    L4 alphabet. This anchors the collector to the number its registrants
    computed, so drift in the alphabet or the nsubj handling cannot pass
    silently.
    """
    from collections import Counter

    from ailm.corpus.ud import case_oracle_assignment, load_ud_ewt_dev
    from ailm.levels.case import load_case_artifact
    from scripts.fillmore_subject_rerun import GOLD_CASE_MAP, sentence_is_alphabet_complete

    def role_for(token):
        return GOLD_CASE_MAP.get(token.case_morphological) or case_oracle_assignment(token)[0]

    artifact = load_case_artifact("derived")
    alphabet = set(artifact.upos_tags)
    count = Counter()
    n_subj_all = 0
    for sentence in load_ud_ewt_dev():
        roles = {}
        subject_index = None
        for i, token in enumerate(sentence.tokens):
            role = role_for(token)
            if role is not None:
                roles[i] = role
            if token.deprel == "nsubj":
                count["nsubj_seen"] += 1
                if subject_index is None:
                    subject_index = i
                else:
                    subject_index = -1
        if subject_index is not None and subject_index >= 0:
            n_subj_all += 1
            token_upos = {i: t.upos for i, t in enumerate(sentence.tokens)}
            if sentence_is_alphabet_complete(
                token_upos, roles, alphabet, subject_index=subject_index
            ):
                count["scorable"] += 1
    assert count["scorable"] == 702
