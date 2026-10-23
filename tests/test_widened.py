"""The widened Level 5 alphabet: one definition, two consumers.

``ailm.levels.widened`` exists because the alphabet, its extraction and its
Markov fit were written twice — once in the dashboard engine and once in
``scripts/probe_l5_widened_alphabet.py`` — with nothing at runtime that would
notice them diverging. A swapped VERB/AUX index still produces a valid-looking
posterior, just a wrong one.

These tests hold the module to what that consolidation promised: the symbols
mean what the level says they mean, the fit is oriented the way this project
orients transition matrices, and the two consumers get identical chains from
identical input. Each behavioral check carries a negative control, because an
assertion that passes for any alphabet is not evidence about this one.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from ailm.cases import CaseRole
from ailm.levels.widened import (
    AUX_SYMBOL,
    MIN_SEQUENCE_LEN,
    N_WIDENED_SYMBOLS,
    TYPE_BY_FINAL_PUNCT,
    VERB_SYMBOL,
    WIDENED_SYMBOL_LABELS,
    extract_widened_sequences,
    fit_markov_chain,
    widened_symbols,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = REPO_ROOT / "scripts" / "probe_l5_widened_alphabet.py"


@dataclass
class FakeToken:
    """The two fields the widened alphabet reads off a UD token."""

    upos: str
    case_derived: CaseRole | None = None
    form: str = "x"


@dataclass
class FakeSentence:
    tokens: list[FakeToken]


def _sentence(spec: list[tuple[str, CaseRole | None]], final: str) -> FakeSentence:
    tokens = [FakeToken(upos=upos, case_derived=case) for upos, case in spec]
    tokens.append(FakeToken(upos="PUNCT", form=final))
    return FakeSentence(tokens=tokens)


# --------------------------------------------------------------------------
# The alphabet
# --------------------------------------------------------------------------


def test_the_two_non_case_symbols_sit_above_the_case_roles():
    """VERB and AUX extend the case alphabet; they do not overwrite part of it."""
    assert len(CaseRole) == 8
    assert (VERB_SYMBOL, AUX_SYMBOL) == (8, 9)
    assert len(CaseRole) + 2 == N_WIDENED_SYMBOLS
    assert len(WIDENED_SYMBOL_LABELS) == N_WIDENED_SYMBOLS
    assert WIDENED_SYMBOL_LABELS[VERB_SYMBOL] == "VERB"
    assert WIDENED_SYMBOL_LABELS[AUX_SYMBOL] == "AUX"
    # Negative control: the case portion is the case enum in enum order, not
    # merely eight labels of the right count.
    assert WIDENED_SYMBOL_LABELS[: len(CaseRole)] == tuple(role.name for role in CaseRole)


def test_verb_and_aux_are_distinct_symbols():
    """Collapsing them would delete subject-auxiliary inversion silently.

    Inversion is the strongest interrogative cue in English and the entire
    reason the alphabet was widened. If VERB and AUX shared a symbol, ``Did she
    go`` and ``She did go`` would produce the same sequence and the level would
    still return a confident, useless posterior.
    """
    inverted = widened_symbols([FakeToken("AUX"), FakeToken("VERB")])
    upright = widened_symbols([FakeToken("VERB"), FakeToken("AUX")])
    assert inverted != upright
    assert inverted == [AUX_SYMBOL, VERB_SYMBOL]


def test_punctuation_contributes_no_symbol():
    """The chains are fitted with PUNCT skipped, so inference must skip it too."""
    with_punct = widened_symbols(
        [FakeToken("AUX"), FakeToken("PUNCT", form="?"), FakeToken("VERB")]
    )
    assert with_punct == [AUX_SYMBOL, VERB_SYMBOL]


def test_a_token_with_neither_a_tag_nor_a_case_emits_nothing():
    """A caseless non-verb has no position in this alphabet.

    Emitting a default for it would put mass on a role the sentence never
    filled, which is the failure mode that made AUX unreachable in the
    dashboard: a fallback applied before the tag was resolved.
    """
    assert widened_symbols([FakeToken("DET", None)]) == []
    assert widened_symbols([FakeToken("DET", CaseRole.NOM)]) == [list(CaseRole).index(CaseRole.NOM)]


def test_every_case_role_maps_to_its_enum_position():
    for index, role in enumerate(CaseRole):
        assert widened_symbols([FakeToken("NOUN", role)]) == [index]


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def test_the_final_mark_names_the_type_and_the_exclamation_is_dropped():
    """``!`` is neither modeled type, so those sentences leave the corpus.

    Forcing them into ``declarative`` because the mark is not ``?`` would put
    exclamatives into the declarative chain and quietly change what the
    declarative model is a model of.
    """
    assert set(TYPE_BY_FINAL_PUNCT) == {".", "?"}
    spec = [("AUX", None), ("NOUN", CaseRole.NOM), ("VERB", None)]
    pairs = extract_widened_sequences(
        [_sentence(spec, "."), _sentence(spec, "?"), _sentence(spec, "!")]
    )
    assert [stype for stype, _ in pairs] == ["declarative", "interrogative"]


def test_a_sequence_too_short_to_carry_a_transition_is_dropped():
    """One symbol is a start prior and nothing else; the level is about order."""
    single = _sentence([("VERB", None)], ".")
    assert extract_widened_sequences([single]) == []
    pair = _sentence([("VERB", None), ("NOUN", CaseRole.ACC)], ".")
    assert len(extract_widened_sequences([pair])[0][1]) == MIN_SEQUENCE_LEN


# --------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------


def test_the_transition_is_column_stochastic_in_next_given_current():
    """``B[next, current]`` is this project's orientation throughout.

    Stating it the other way is the orientation trap decision 1 records: the
    transposed matrix is also a valid stochastic matrix, so nothing raises and
    every downstream number is wrong.
    """
    sequences = [[0, 1, 1], [1, 0, 0]]
    B, D = fit_markov_chain(sequences)
    assert B.shape == (N_WIDENED_SYMBOLS, N_WIDENED_SYMBOLS)
    assert D.shape == (N_WIDENED_SYMBOLS,)
    np.testing.assert_allclose(B.sum(axis=0), 1.0)
    np.testing.assert_allclose(D.sum(), 1.0)
    # The observed 0->1 transition must outweigh 0->2, which was never seen.
    # A transposed fit would put that asymmetry on the wrong axis, so this is
    # the check that distinguishes the two orientations.
    assert B[1, 0] > B[2, 0]
    assert B[0, 1] > B[2, 1]


def test_smoothing_leaves_no_transition_impossible():
    """An unseen bigram must be unlikely, never impossible.

    Zero smoothing turns one absent transition into an infinite log-loss on the
    first test sentence that uses it, which reads as a broken model rather than
    as a sparse corpus.
    """
    B, D = fit_markov_chain([[0, 1]])
    assert (B > 0).all()
    assert (D > 0).all()


def test_a_non_positive_pseudocount_is_refused():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="pseudocount"):
            fit_markov_chain([[0, 1]], pseudocount=bad)


def test_more_smoothing_flattens_the_chain():
    """The pseudocount is a real knob, not an argument the fit ignores."""
    sequences = [[0, 1, 1, 1], [0, 1, 1, 1]]
    light, _ = fit_markov_chain(sequences, pseudocount=0.01)
    heavy, _ = fit_markov_chain(sequences, pseudocount=100.0)
    uniform = 1.0 / N_WIDENED_SYMBOLS
    assert abs(heavy[1, 0] - uniform) < abs(light[1, 0] - uniform)


def test_an_empty_corpus_fits_the_uniform_chain():
    B, D = fit_markov_chain([])
    np.testing.assert_allclose(B, 1.0 / N_WIDENED_SYMBOLS)
    np.testing.assert_allclose(D, 1.0 / N_WIDENED_SYMBOLS)


# --------------------------------------------------------------------------
# One definition, two consumers
# --------------------------------------------------------------------------


def test_the_probe_and_the_dashboard_extract_the_same_pairs():
    """Same corpus in, same pairs out — the property consolidation bought.

    The probe splits and the dashboard does not, so this recombines the probe's
    two halves in corpus order and compares against what the dashboard sees.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import probe_l5_widened_alphabet as probe

    from ailm.corpus.ud import load_ud_ewt_dev

    sentences = load_ud_ewt_dev()
    dashboard_pairs = extract_widened_sequences(sentences)
    train, test = probe.extract_widened_sequences()

    assert len(train) + len(test) == len(dashboard_pairs)
    recombined = sorted(train + test)
    assert recombined == sorted(dashboard_pairs)

    B_dash, D_dash = fit_markov_chain([sequence for _, sequence in dashboard_pairs])
    B_probe, D_probe = probe.fit_markov_chain(
        [sequence for _, sequence in train + test], probe.N_WIDENED_SYMBOLS
    )
    assert np.array_equal(B_dash, B_probe), "the two consumers fit different chains"
    assert np.array_equal(D_dash, D_probe)


def test_the_probe_split_is_disjoint_and_covers_the_corpus():
    """A held-out set that leaks is a training set with a different name."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import probe_l5_widened_alphabet as probe

    from ailm.corpus.ud import document_stride_partition, load_ud_ewt_dev
    from ailm.levels.widened import extract_widened_sequences_documented

    train, test = probe.extract_widened_sequences()
    assert test, "the probe reports a held-out measurement; it must hold something out"
    assert train, "the probe fits on the training half; it must have one"

    sentences = load_ud_ewt_dev()
    document_ids, split = document_stride_partition(sentences, test_every=probe.TEST_EVERY)
    rows = extract_widened_sequences_documented(sentences)
    train_rows = [(stype, seq) for doc, stype, seq in rows if split[doc] == "train"]
    test_rows = [(stype, seq) for doc, stype, seq in rows if split[doc] == "test"]
    assert train == train_rows
    assert test == test_rows

    train_documents = {doc for doc, _, _ in rows if split[doc] == "train"}
    test_documents = {doc for doc, _, _ in rows if split[doc] == "test"}
    assert train_documents.isdisjoint(test_documents)
    raw_test_documents = sum(split[doc] == "test" for doc in document_ids)
    assert raw_test_documents / len(document_ids) == pytest.approx(1 / probe.TEST_EVERY, abs=0.01)


def test_neither_consumer_redeclares_what_it_imports():
    """A matching second copy is a divergence that has not happened yet.

    This checks the general property — no redeclaration of the alphabet or the
    fit anywhere in either consumer — rather than the specific copies that were
    removed, so a *new* transcription fails too.
    """
    for path in (PROBE, REPO_ROOT / "src" / "ailm" / "dashboard" / "engine.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assigned = {
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        assert not assigned & {"VERB_SYMBOL", "AUX_SYMBOL", "N_WIDENED_SYMBOLS"}, (
            f"{path.name} redeclares a widened-alphabet constant; it must import it"
        )
        defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        assert not defined & {"fit_markov_chain", "_fit_chain", "widened_symbols"}, (
            f"{path.name} defines its own copy of a shared routine"
        )


def test_the_probe_defines_everything_before_it_runs():
    """A def below the ``__main__`` guard does not exist when ``main()`` runs.

    Placement is not style here: the module body executes top to bottom, so a
    helper defined after the guard raises ``NameError`` for anyone running the
    script the documented way while remaining perfectly importable — which is
    to say invisible to every test that imports it. The check is positional and
    general, not a check for one function by name.
    """
    source = PROBE.read_text(encoding="utf-8")
    guard = re.search(r"^if __name__ == .__main__.:", source, re.M)
    assert guard, "the probe is documented as runnable; it needs a __main__ guard"
    guard_line = source[: guard.start()].count("\n") + 1
    late: list[str] = []
    for node in ast.parse(source).body:
        if node.lineno <= guard_line:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            late.append(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            late.extend(t.id for t in targets if isinstance(t, ast.Name))
    assert not late, f"defined after the __main__ guard, so unavailable to main(): {late}"
