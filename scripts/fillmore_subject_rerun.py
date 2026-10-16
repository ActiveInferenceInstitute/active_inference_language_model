#!/usr/bin/env python3
"""Decision 42's registered per-token re-run of the Fillmore subject arm.

Business logic for the re-run lives HERE, not in ``ailm.levels.case``: the
owning lane writes new files under ``scripts/`` only. The functions are pure
and take their inputs explicitly, so they test against real corpus data
without artifact coupling beyond what the tests import themselves.

Why per token
-------------
The shipped rule scored whole sentences and dropped any sentence containing a
token whose tag fell outside the L4 UPOS observation alphabet -- one unseen
AUX or PUNCT discarded the whole sentence, leaving nine scorable sentences.
Decision 42 registers the fix before it is implemented: score PER TOKEN over
the sentences whose subject token and candidate set are entirely inside the
alphabet (research-queue evidence: 702 such sentences on all of EWT dev,
against 9 under the retired rule).

What counts, verbatim from the registration
-------------------------------------------
* The arm enters the main line only if its scorable count clears the minimum
  an equivalence claim needs at declared coverage; the count is reported
  either way.
* Refusals lower coverage, not accuracy.
* The model path is compared against the leftmost-role positional baseline
  and the derived-assignment control in ONE table; a model path near the
  control is circularity, not success.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def sentence_is_alphabet_complete(
    token_upos: Mapping[int, str],
    roles: Mapping[int, object],
    alphabet: set[str] | frozenset[str],
    *,
    subject_index: int | None = None,
) -> bool:
    """True when the registered per-token rule keeps the sentence.

    The rule scores per token, so a tag outside the alphabet matters only where
    it would have been USED: the subject token itself, or any token bearing a
    role in Fillmore's hierarchy (NOM/INS/ACC) among the candidates the rule
    could select. A stray AUX with no role is scored around, not dropped --
    that is precisely what recovers n=9 to hundreds of sentences.

    Args:
        token_upos: Token index -> UPOS tag for every token of the sentence.
        roles: Token index -> assigned case role, role-bearing tokens only.
        alphabet: The L4 observation alphabet (artifact ``upos_tags``).
        subject_index: Index of the single ``nsubj`` token; when omitted, only
            the role population is checked.

    Returns:
        False when the subject token is outside the alphabet, or when any
        role-bearing candidate is outside it. True otherwise.
    """
    if subject_index is not None:
        upos_subject = token_upos.get(subject_index)
        if upos_subject is None or upos_subject not in alphabet:
            return False
    return all(token_upos.get(i) in alphabet for i in roles)


def leftmost_role_choice(roles: Sequence[object]) -> int | None:
    """The strongest positional rule that uses no case knowledge at all.

    Names the first index carrying ANY role, regardless of which role. Refuses
    (``None``) on a sentence with no role-bearing tokens, so refusals stay
    measurable rather than being converted into a default prediction.

    Registered as the positional comparator because a leftmost-TOKEN variant
    cannot separate "subjecthood" from "sentence start" over English's
    near-universal initial-PRON subject pattern.
    """
    for index, role in enumerate(roles):
        if role is not None:
            return index
    return None


def per_token_aggregate(
    results: Sequence[Mapping[str, object] | None],
) -> dict[str, float | int]:
    """Aggregate per-sentence scoring results under the refusal vocabulary.

    Args:
        results: One entry per scorable sentence; each carries ``correct``
            (bool). ``None`` marks a REFUSAL -- the rule named no token --
            which lowers coverage and never touches accuracy.

    Returns:
        ``scored``, ``predicted``, ``correct``, ``accuracy`` over predictions,
        ``coverage`` over scored. Empty input reports zeros rather than
        raising, so a fully-refused arm records its zeros instead of
        vanishing from the ledger.
    """
    scored = len(results)
    predicted = sum(1 for r in results if r is not None)
    correct = sum(1 for r in results if r is not None and r["correct"])
    return {
        "scored": scored,
        "predicted": predicted,
        "correct": correct,
        "accuracy": round(correct / predicted, 4) if predicted else 0.0,
        "coverage": round(predicted / scored, 4) if scored else 0.0,
    }


def entry_bar_verdict(predicted: int, minimum_pairs: int) -> tuple[bool, str]:
    """Decision 42's entry bar: does the arm resolve at its predicted count?

    Returns ``(enters_main_line, verdict_word)``. At or above the minimum an
    equivalence null was reachable whatever fell, so the outcome means
    something; below it the arm is registered-unresolvable regardless of its
    numbers, and reporting its accuracy as a finding would be the defect the
    registration exists to prevent.
    """
    if predicted >= minimum_pairs:
        return True, "resolvable"
    return False, "unresolvable"


#: UD feats[Case] morphology value -> CaseRole name, exactly as the vendored
#: artifact generator applies it before falling back to the syntactic oracle.
GOLD_CASE_MAP: dict[str, str] = {"Nom": "NOM", "Acc": "ACC", "Gen": "GEN"}
