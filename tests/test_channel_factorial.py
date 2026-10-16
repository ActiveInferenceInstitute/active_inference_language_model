"""The gate on the channel-factorial executor, written before the executor.

Decision 46 pre-registers a 2x2 crossing message source with insertion point,
because [@sec:hierarchy] states that "the two cells that would separate source
from insertion point are not built" and decisions 18-19 therefore rest on a
diagonal.

The risk in that arm is not the measurement, it is the instrument: an executor
that reimplements the shipped decoders rather than dispatching into them measures
itself. So the binding test compares per-position **hit vectors** against the
shipped paths, not accuracies. Two decoders can agree to four decimal places on
an aggregate while disagreeing on thousands of positions in both directions, and
an executor that reproduces only the number would pass a weaker check.

These tests are expected to FAIL until `run_cell` is built. That is deliberate:
`gnn/AGENTS.md` forbids editing a spec to make a check pass, and the same rule
applies here — the gate exists first so that the executor is written against it
rather than the other way round.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ailm.channels import (
    CELLS,
    effective_weight,
    letters_to_phones,
    phones_to_letters,
    reverse_emission,
)

#: The fixture chain re-runs the git-history epoch machinery; the file carries
#: its own budget instead of the repository default.
pytestmark = pytest.mark.timeout(900)


@pytest.fixture(scope="module")
def g2p_matrices():
    """The shipped G2P matrices, built by the shipped builder on the shipped split.

    Module-scoped because building them is the expensive part and every test here
    reads the same object. Built from the real artifact rather than a synthetic
    stand-in: the transport asymmetry these tests document is a property of the
    fitted emission's mass distribution, and a toy matrix would not have it.
    """
    from pathlib import Path as _Path

    from ailm.levels.coupling import build_g2p_matrices
    from ailm.levels.phoneme import g2p_split, load_phoneme_counts

    vendor = _Path(__file__).resolve().parents[1] / "vendor"
    if not (vendor / "cmudict_counts.json").exists():
        pytest.skip("vendored phone counts absent; run the generators")
    train, _test = g2p_split(load_phoneme_counts(vendor))
    return build_g2p_matrices(train)


class TestTheTransportsAreProperDistributions:
    """The off-diagonal cells need an alphabet transport; the diagonal does not.

    This asymmetry is decision 46's amendment and it is the module's own finding.
    The evidence site consumes a distribution over letters and the predictive site
    one over phones, so a letter message at the predictive site and a lexical
    message at the evidence site each have to cross alphabets, while neither
    shipped cell does.
    """

    def test_the_reverse_conditional_is_a_distribution_for_every_letter(self, g2p_matrices):
        reverse = reverse_emission(g2p_matrices.emission, g2p_matrices.start)
        assert reverse.shape == np.asarray(g2p_matrices.emission).shape
        assert np.allclose(reverse.sum(axis=1), 1.0, atol=1e-12), (
            "P(phone | letter) does not sum to one for every letter, so the "
            "letter-to-phone transport is not a distribution"
        )

    def test_a_letter_with_no_evidence_mass_raises_rather_than_going_uniform(self):
        """A uniform row would invent a message for a letter the model is silent about.

        This is the failure mode the project has paid for before: a reconstruction
        that fills an unobserved slot with a default reads as a measurement.
        """
        emission = np.zeros((3, 2))
        emission[0, :] = [1.0, 1.0]
        with pytest.raises(ValueError, match="no evidence mass"):
            reverse_emission(emission, np.array([0.5, 0.5]))

    def test_both_transports_preserve_normalization(self, g2p_matrices):
        emission = np.asarray(g2p_matrices.emission)
        letters = np.eye(emission.shape[0])[:5]
        phones = np.eye(emission.shape[1])[:5]
        carried = letters_to_phones(letters, emission, g2p_matrices.start)
        back = phones_to_letters(phones, emission)
        assert np.allclose(carried.sum(axis=1), 1.0)
        assert np.allclose(back.sum(axis=1), 1.0)

    def test_the_two_directions_are_not_mirror_images(self, g2p_matrices):
        """The asymmetry is the finding, so it is asserted rather than assumed.

        Phone-to-letter needs no inversion because the emission is
        column-stochastic over letters. Letter-to-phone does, because the raw
        product is a likelihood summing to well over one, and normalizing it is
        what `soft_observation_likelihood` says "would turn the evidence into a
        prior and silently change the model".
        """
        emission = np.asarray(g2p_matrices.emission)
        raw = emission.T @ np.full(emission.shape[0], 1.0 / emission.shape[0])
        assert raw.sum() > 1.2, (
            f"the raw letter-to-phone product sums to {raw.sum()}, close enough to one "
            f"that the transport question this test documents may no longer arise"
        )
        proper = emission @ np.full(emission.shape[1], 1.0 / emission.shape[1])
        assert proper.sum() == pytest.approx(1.0), (
            "phone-to-letter no longer yields a distribution without inversion"
        )


def test_the_declared_weight_is_not_a_common_dose_across_sites(g2p_matrices):
    """Registered in decision 46: a declared alpha is a different effective weight
    depending on the observed letter, so the "common weight grid" claim across
    sites is false unless the executor scale-matches and reports both."""
    emission = np.asarray(g2p_matrices.emission)
    masses = emission.sum(axis=1)
    assert masses.max() / masses.min() > 5.0, (
        f"evidence mass varies by only {masses.max() / masses.min():.1f}x across letters; "
        f"if that ratio has collapsed, the scale-matching this arm performs is no longer "
        f"needed and decision 46's second registered confound should be withdrawn"
    )
    low, high = effective_weight(0.5, masses.min()), effective_weight(0.5, masses.max())
    assert high - low > 0.2, (
        f"a declared alpha of 0.5 spans effective weights {low:.3f}-{high:.3f}; the "
        f"spread is what makes the raw grid incomparable across sites"
    )


def test_the_factorial_declares_four_cells_two_of_which_are_shipped():
    assert len(CELLS) == 4 and len(set(CELLS)) == 4
    assert ("letter", "evidence") in CELLS and ("lexicon", "predictive") in CELLS
    assert ("letter", "predictive") in CELLS and ("lexicon", "evidence") in CELLS


def test_the_executor_reproduces_the_shipped_hit_vectors_exactly(g2p_matrices):
    """THE BINDING GATE, now satisfiable.

    Compares per-position booleans against the shipped coupled decoder, not
    accuracies. Two decoders can agree to four decimal places on an aggregate
    while disagreeing on thousands of positions in both directions, so an
    executor that reproduces only the number would pass a weaker check and the
    factorial would then be measuring the executor rather than the channels.

    A stride subsample rather than the full split, because the recorder asserts
    the same identity over all 42,820 positions as a precondition before it
    writes any claim. This is the fast version of a check that also runs where it
    matters.
    """
    from pathlib import Path as _Path

    from ailm.channels import run_cell
    from ailm.levels.coupling import build_letter_matrices, coupled_positions
    from ailm.levels.letter import load_brown_counts
    from ailm.levels.lexical_channel import build_lexical_message
    from ailm.levels.phoneme import PHONES, g2p_split, load_phoneme_counts

    vendor = _Path(__file__).resolve().parents[1] / "vendor"
    brown = vendor / "brown_counts.json"
    if not brown.exists():
        pytest.skip("vendored Brown counts absent; run the generators")
    train, test = g2p_split(load_phoneme_counts(vendor))
    letters = build_letter_matrices(load_brown_counts(vendor))
    phones = list(PHONES)

    for alpha in (0.0, 0.1):
        mine = run_cell(
            "letter",
            "evidence",
            letter_matrices=letters,
            g2p=g2p_matrices,
            message=build_lexical_message(train, phones),
            pairs=test,
            alpha=alpha,
            phone_to_index={p: i for i, p in enumerate(phones)},
            limit=400,
        )
        shipped = coupled_positions(letters, g2p_matrices, test, alpha=alpha, limit=400)
        assert len(mine) == len(shipped) and mine == shipped, (
            f"at alpha={alpha} the executor disagrees with the shipped coupled decoder on "
            f"{sum(a != b for a, b in zip(mine, shipped, strict=False))} of {len(mine)} "
            f"positions; it is a reimplementation, not a dispatcher"
        )


def test_the_factorial_result_is_an_interaction_not_a_main_effect():
    """Guards the reading decision 46 records, against the recorded numbers.

    The completed 2x2 says the 2.01-point diagonal gap is carried by neither
    factor alone: no message helps at the evidence site at any weight, and at the
    predictive site the letter message does not clear significance. If a re-run
    made either main effect large, the sentence in [@sec:hierarchy] attributing
    the result to the combination would be false and would have to be rewritten
    rather than this test relaxed.
    """
    from pathlib import Path as _Path

    ledger = _Path(__file__).resolve().parents[1] / "data" / "claims.json"
    if not ledger.exists():
        pytest.skip("ledger absent")
    claims = json.loads(ledger.read_text())["claims"]
    if "channel_interaction_pts" not in claims:
        pytest.skip("channel factorial not yet recorded")

    def value(label: str) -> float:
        return float(claims[label]["value"])

    interaction = value("channel_interaction_pts")
    main_source = abs(value("channel_source_effect_at_evidence_pts"))
    main_site = abs(value("channel_site_effect_for_letter_pts"))
    assert interaction > 1.0, (
        f"the interaction is {interaction} points; decision 46 reads the result as "
        f"carried by the combination of source and site"
    )
    assert main_source < interaction / 2 and main_site < interaction / 2, (
        f"a main effect is now comparable to the interaction (source {main_source}, "
        f"site {main_site}, interaction {interaction}); the combination reading has to "
        f"be rewritten rather than this threshold moved"
    )


class TestGeometricPooling:
    """Decision 47: the rule the factorial held fixed, and what varying it settles.

    Two of these are algebraic identities rather than measurements, so they are
    asserted at machine precision. An identity that fails is a bug in the
    implementation and never a finding, which is the opposite of how the rest of
    this file's assertions should be read.
    """

    def _setup(self):
        from pathlib import Path as _Path

        import numpy as np

        from ailm.levels.coupling import build_g2p_matrices
        from ailm.levels.letter import ALPHA_TO_IDX
        from ailm.levels.lexical_channel import build_lexical_message, lexical_message_for
        from ailm.levels.phoneme import PHONES, g2p_split, load_phoneme_counts

        vendor = _Path(__file__).resolve().parents[1] / "vendor"
        if not (vendor / "cmudict_counts.json").exists():
            pytest.skip("vendored phone counts absent")
        train, test = g2p_split(load_phoneme_counts(vendor))
        g2p = build_g2p_matrices(train)
        message = build_lexical_message(train, list(PHONES))
        word = test[7][0]
        rows = np.asarray(lexical_message_for(message, word)[0])
        return g2p, [ALPHA_TO_IDX[c] for c in word], rows

    def test_the_two_insertion_sites_are_the_same_computation(self):
        """The identity decision 47 rests on, checked on the shipped matrices.

        At the evidence site the posterior is proportional to A[l] * m**e * pred;
        at the predictive site to A[l] * (pred * m**e). Multiplication is
        associative, the intermediate normalization cancels, and the posterior is
        what propagates, so the trajectories coincide. If this ever fails, the
        implementation is not pooling geometrically.
        """
        import numpy as np

        g2p, letters, rows = self._setup()
        emission = np.asarray(g2p.emission)
        transition = np.asarray(g2p.transition)
        start = np.asarray(g2p.start)

        def trajectory(site: str, exponent: float):
            predictive, out = start.copy(), []
            for index, letter in enumerate(letters):
                weighted = np.power(np.maximum(rows[index], 1e-300), exponent)
                if site == "evidence":
                    posterior = emission[letter, :] * weighted * predictive
                else:
                    shifted = predictive * weighted
                    posterior = emission[letter, :] * (shifted / shifted.sum())
                posterior = posterior / posterior.sum()
                out.append(posterior)
                predictive = transition @ posterior
            return np.array(out)

        for exponent in (0.0, 0.1, 0.3, 0.5, 1.0):
            gap = np.abs(
                trajectory("evidence", exponent) - trajectory("predictive", exponent)
            ).max()
            assert gap < 1e-12, (
                f"the two sites differ by {gap} at exponent {exponent}; under geometric "
                f"pooling they are algebraically one computation, so this is an "
                f"implementation bug and not a measurement"
            )

    def test_a_uniform_message_is_inert_which_is_what_separates_content(self):
        """THE CONTROL the whole rule is worth having for.

        A constant raised to any power cancels in normalization, so a uniform
        message cannot move a posterior. Under linear mixing it can, which is why
        a linear result mixes information with regularisation separably only by
        argument. This assertion is what licenses reading every other geometric
        margin as content.
        """
        import numpy as np

        from ailm.channels import _geometric_positions

        g2p, letters, _rows = self._setup()
        emission = np.asarray(g2p.emission)
        transition = np.asarray(g2p.transition)
        start = np.asarray(g2p.start)
        n_phones = emission.shape[1]
        truth = [0] * len(letters)
        uniform = np.full((len(letters), n_phones), 1.0 / n_phones)

        anchor = _geometric_positions(emission, transition, start, letters, truth, uniform, 0.0)
        for exponent in (0.1, 0.5, 1.0):
            hits = _geometric_positions(
                emission, transition, start, letters, truth, uniform, exponent
            )
            assert hits == anchor, (
                f"a uniform message moved the decode at exponent {exponent}; it cannot "
                f"under geometric pooling, so either the implementation is not "
                f"geometric or the message is not uniform"
            )

    def test_the_letter_message_still_harms_under_a_rule_that_cannot_dilute(self):
        """Guards the reading that vindicates decision 18.

        The letter message harms monotonically in the exponent under a rule whose
        uniform control is provably inert, so the harm is content rather than
        dilution. If a re-run made it help, decision 18's original claim — that
        the message carries nothing the receiver does not already observe — would
        be refuted, and the prose saying so must be rewritten rather than this
        test relaxed.
        """
        import json
        from pathlib import Path as _Path

        ledger = _Path(__file__).resolve().parents[1] / "data" / "claims.json"
        if not ledger.exists():
            pytest.skip("ledger absent")
        claims = json.loads(ledger.read_text())["claims"]
        if "geometric_letter_best_margin_pts" not in claims:
            pytest.skip("geometric pooling not yet recorded")
        best = float(claims["geometric_letter_best_margin_pts"]["value"])
        full = float(claims["geometric_letter_full_exponent_margin_pts"]["value"])
        uniform = float(claims["geometric_uniform_best_margin_pts"]["value"])
        assert uniform == 0.0, (
            f"the uniform control moved by {uniform} points, so dilution is not excluded "
            f"and the letter arm's harm cannot be read as content"
        )
        assert best < 0.0 and full < best, (
            f"the letter message no longer harms monotonically (best {best}, full {full}); "
            f"decision 18's claim is refuted and the prose must be rewritten"
        )


# ── The executor's guards and the two cells the gate above does not reach ────


class TestTransportGuards:
    """The refusals that keep a transport from inventing a message."""

    def test_an_emission_and_prior_that_do_not_compose_are_refused(self):
        """Failure caught: a shape mismatch silently broadcasting into a wrong
        joint. NumPy would happily broadcast some of these; the guard names the
        problem instead of letting the arithmetic proceed on garbage."""
        with pytest.raises(ValueError, match="do not compose"):
            reverse_emission(np.full((3, 2), 0.5), np.array([0.5, 0.25, 0.25]))
        with pytest.raises(ValueError, match="do not compose"):
            reverse_emission(np.array([0.5, 0.5]), np.array([0.5, 0.5]))
        # Negative control: a composable pair is not refused.
        assert reverse_emission(np.full((3, 2), 0.5), np.array([0.5, 0.5])).shape == (3, 2)

    def test_effective_weight_at_zero_mass_is_zero_not_a_division(self):
        """Failure caught: a dead evidence vector turning the scale-match into a
        ZeroDivisionError mid-census. Zero mass means the site receives nothing,
        so the effective weight is zero by definition, not by crash."""
        assert effective_weight(0.5, 0.0) == 0.0
        assert effective_weight(0.5, -1.0) == 0.0
        # Negative controls: unit mass applies the declared alpha exactly, and
        # the weight moves with the mass — a constant 0.0 would pass the guard
        # above vacuously.
        assert effective_weight(0.5, 1.0) == pytest.approx(0.5)
        assert effective_weight(0.3, 4.0) > 0.3 > effective_weight(0.3, 0.25)


def test_a_dead_posterior_falls_back_to_the_prediction_not_a_zero_division():
    """Failure caught: an observed letter whose emission row is all zeros
    annihilates the posterior, and normalizing it divides by zero. The filter
    falls back to the predictive prior — the model's answer when the observation
    says nothing — and keeps decoding.
    """
    from ailm.channels import _geometric_positions

    emission = np.array([[0.0, 0.0], [0.1, 1.0]])
    transition = np.eye(2)
    start = np.array([0.9, 0.1])
    uniform = np.full((1, 2), 0.5)

    # Letter 0 has a dead row, so the decode is the prior's argmax: phone 0.
    assert _geometric_positions(emission, transition, start, [0], [0], uniform, 0.0) == [True]
    # The same fallback under a positive exponent, where the message is applied
    # and still cannot resurrect a zero posterior.
    assert _geometric_positions(emission, transition, start, [0], [1], uniform, 0.5) == [False]
    # NEGATIVE CONTROL: a live emission row moves the argmax off the prior, so
    # the assertions above cannot be satisfied by a filter that always returns
    # the prior's answer.
    assert _geometric_positions(emission, transition, start, [1], [1], uniform, 0.0) == [True]


def test_site_equivalence_is_measured_at_zero_exactly_and_beyond_at_float_noise(g2p_matrices):
    """The measurement decision 47 replaced a typed number with.

    The two variants are separate implementations on purpose, and the gap
    between them is what the recorder writes to the ledger; at every exponent it
    must be float noise with no argmax disagreement, or the pooling is not
    geometric. Even at exponent zero the gap is float noise rather than an exact
    zero: the predictive-site variant renormalizes the prior before the emission
    product, and that extra division is one rounding the evidence-site variant
    never performs. (Measured here at ~1.1e-16 on the shipped matrices — the
    function's own docstring says the zero-exponent difference "is exactly
    zero", which this measurement shows is off by one ulp.)
    """
    from ailm.channels import geometric_site_equivalence

    emission = np.asarray(g2p_matrices.emission)
    transition = np.asarray(g2p_matrices.transition)
    start = np.asarray(g2p_matrices.start)
    n_phones = emission.shape[1]
    letters = [0, 1, 2, 3]
    rows = np.tile(np.arange(1.0, n_phones + 1.0) / np.arange(1.0, n_phones + 1.0).sum(), (4, 1))

    for exponent in (0.0, 0.7):
        worst, disagreements = geometric_site_equivalence(
            emission, transition, start, letters, rows, exponent
        )
        assert worst < 1e-12 and disagreements == 0, (
            f"the two insertion sites differ by {worst} with {disagreements} argmax "
            f"disagreements at exponent {exponent}; decision 47's proof says they are "
            f"one computation, so this is an implementation bug"
        )


@pytest.fixture(scope="module")
def factorial(g2p_matrices):
    """Everything `run_cell` needs, built once from the shipped artifacts."""
    from pathlib import Path as _Path

    from ailm.levels.coupling import build_letter_matrices
    from ailm.levels.letter import load_brown_counts
    from ailm.levels.lexical_channel import build_lexical_message
    from ailm.levels.phoneme import PHONES, g2p_split, load_phoneme_counts

    vendor = _Path(__file__).resolve().parents[1] / "vendor"
    brown = vendor / "brown_counts.json"
    if not brown.exists():
        pytest.skip("vendored Brown counts absent; run the generators")
    train, test = g2p_split(load_phoneme_counts(vendor))
    return {
        "letter_matrices": build_letter_matrices(load_brown_counts(vendor)),
        "g2p": g2p_matrices,
        "message": build_lexical_message(train, list(PHONES)),
        "pairs": test,
        "phone_to_index": {p: i for i, p in enumerate(PHONES)},
    }


class TestAllFourCellsExecute:
    """The two off-diagonal cells run through the same dispatcher as the gate.

    The binding gate above pins the (letter, evidence) diagonal to the shipped
    decoder. These tests run the other three cells — including both transports —
    against the one property every cell must share and one it must not.
    """

    def test_at_alpha_zero_every_cell_is_the_same_uncoupled_decode(self, factorial):
        """The identity anchor holds across the whole factorial.

        At alpha zero no message reaches either site, so source and site can
        change nothing and all four cells must return the identical per-position
        hit vector. A cell that differed at the anchor would be reading its
        message through some path other than the declared insertion point.
        """
        from ailm.channels import run_cell

        vectors = {cell: run_cell(*cell, alpha=0.0, limit=60, **factorial) for cell in CELLS}
        anchor = vectors[("letter", "evidence")]
        assert len(anchor) > 300, "the subsample is too small to anchor anything"
        for cell, vector in vectors.items():
            assert vector == anchor, (
                f"cell {cell} disagrees with the uncoupled decode at alpha=0 on "
                f"{sum(a != b for a, b in zip(vector, anchor, strict=True))} positions; "
                f"the message is leaking in at the anchor"
            )

    def test_at_a_positive_alpha_every_cell_moves_off_the_anchor(self, factorial):
        """NEGATIVE CONTROL for the anchor test above.

        A cell that ignored its message entirely would pass the identity anchor
        vacuously. Each of the four must change at least one position at a
        positive weight — including the two off-diagonal cells, whose messages
        arrive through an alphabet transport.
        """
        from ailm.channels import run_cell

        anchor = run_cell("letter", "evidence", alpha=0.0, limit=60, **factorial)
        for cell in CELLS:
            moved = run_cell(*cell, alpha=0.3, limit=60, **factorial)
            changed = sum(a != b for a, b in zip(moved, anchor, strict=True))
            assert changed > 0, (
                f"cell {cell} decodes identically with and without its message at "
                f"alpha=0.3; the dispatcher is not delivering the message to this cell"
            )

    def test_a_length_mismatched_pair_is_named_a_bug_not_scored(self, factorial):
        """A decode that returns the wrong number of positions must refuse.

        The corpus is length-matched by construction, so a mismatch here is a
        normalization bug in the executor, and scoring around it would let a
        broken cell report a plausible accuracy over the positions it kept.
        """
        from ailm.channels import run_cell

        broken = {**factorial, "pairs": [("ab", ["AA"])]}
        with pytest.raises(ValueError, match="normalization bug"):
            run_cell("letter", "evidence", alpha=0.1, **broken)
