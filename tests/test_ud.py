"""Tests for ailm.corpus.ud — CoNLL-U parsing and case oracle.

All tests use a tiny inline CoNLL-U sample — no network calls.
The case oracle is a RESEARCH CLAIM; these tests pin the oracle's output
for the specific deprel mappings, not ground truth.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import requests

from ailm.cases import ADPOSITION_TO_CASE, CaseRole
from ailm.corpus.ud import (
    CASE_ORACLE_PATHS,
    UD_EWT_DEV_URL,
    UD_EWT_RAW_SHA256,
    UD_EWT_RAW_SIZE_BYTES,
    UD_EWT_REVISION,
    UD_EWT_SOURCE_METADATA,
    Sentence,
    Token,
    UDSourceIntegrityError,
    _apply_case_oracle,
    _parse_feats,
    _parse_misc,
    case_oracle_scope_audit,
    case_oracle_stats,
    document_identity_sha256,
    document_stride_partition,
    ensure_ud_ewt_dev_path,
    load_ud_ewt_dev,
    parse_conllu,
    read_ud_ewt_dev_bytes,
    validate_ud_ewt_source_identity,
    validate_ud_ewt_source_metadata,
    verify_ud_ewt_bytes,
)

SAMPLE_CONLLU = """\
# newdoc id = weblog-document-a
# sent_id = weblog-test-0001
# text = She saw him .
1\tShe\tshe\tPRON\tPRP\tCase=Nom|Gender=Fem|Number=Sing|Person=3|PronType=Prs\t2\tnsubj\t_\t_
2\tsaw\tsee\tVERB\tVBD\tMood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin\t0\troot\t_\t_
3\thim\the\tPRON\tPRP\tCase=Acc|Gender=Masc|Number=Sing|Person=3|PronType=Prs\t2\tobj\t_\t_
4\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\tSpaceAfter=No

# newdoc id = weblog-document-b
# sent_id = weblog-test-0002
# text = I gave the book to her .
1\tI\tI\tPRON\tPRP\tCase=Nom|Number=Sing|Person=1|PronType=Prs\t2\tnsubj\t_\t_
2\tgave\tgive\tVERB\tVBD\tMood=Ind|Number=Sing|Person=1|Tense=Past|VerbForm=Fin\t0\troot\t_\t_
3\tthe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t4\tdet\t_\t_
4\tbook\tbook\tNOUN\tNN\tNumber=Sing\t2\tobj\t_\t_
5\tto\tto\tADP\tIN\t_\t6\tcase\t_\t_
6\ther\tshe\tPRON\tPRP\tCase=Acc|Gender=Fem|Number=Sing|Person=3|PronType=Prs\t2\tiobj\t_\t_
7\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\tSpaceAfter=No

"""

SAMPLE_WITH_POSSESSIVE = """\
# sent_id = weblog-test-0003
# text = the cat 's mat
1\tthe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t3\tdet\t_\t_
2\tcat\tcat\tNOUN\tNN\tNumber=Sing\t3\tnmod:poss\t_\t_
3\t's\t's\tPART\tPOS\t_\t2\tcase\t_\t_
4\tmat\tmat\tNOUN\tNN\tNumber=Sing\t0\troot\t_\tSpaceAfter=No

"""

FIXTURE_REVISION = "f" * 40


def _fixture_source(payload: bytes, *, url: str = "https://fixtures.invalid/ewt.conllu") -> dict:
    return {
        "url": url,
        "source_revision": FIXTURE_REVISION,
        "expected_sha256": hashlib.sha256(payload).hexdigest(),
        "expected_size_bytes": len(payload),
    }


class TestParseFeats:
    def test_empty_feats(self):
        assert _parse_feats("_") == {}

    def test_single_feat(self):
        assert _parse_feats("Case=Nom") == {"Case": "Nom"}

    def test_multiple_feats(self):
        result = _parse_feats("Case=Nom|Number=Sing")
        assert result["Case"] == "Nom"
        assert result["Number"] == "Sing"

    def test_returns_dict(self):
        assert isinstance(_parse_feats("Case=Acc|Tense=Past"), dict)


class TestParseMisc:
    def test_empty_misc(self):
        assert _parse_misc("_") == {}

    def test_space_after(self):
        result = _parse_misc("SpaceAfter=No")
        assert result["SpaceAfter"] == "No"


class TestParseConllu:
    def test_parses_two_sentences(self):
        sents = parse_conllu(SAMPLE_CONLLU)
        assert len(sents) == 2

    def test_sent_id_extracted(self):
        sents = parse_conllu(SAMPLE_CONLLU)
        assert sents[0].sent_id == "weblog-test-0001"
        assert sents[1].sent_id == "weblog-test-0002"

    def test_stateful_newdoc_identity_is_extracted(self):
        sents = parse_conllu(SAMPLE_CONLLU)
        assert [sentence.document_id for sentence in sents] == [
            "weblog-document-a",
            "weblog-document-b",
        ]

    def test_newdoc_identity_persists_across_multiple_sentences(self):
        source = SAMPLE_CONLLU.replace("# newdoc id = weblog-document-b\n", "")
        sents = parse_conllu(source)
        assert [sentence.document_id for sentence in sents] == [
            "weblog-document-a",
            "weblog-document-a",
        ]

    def test_text_extracted(self):
        sents = parse_conllu(SAMPLE_CONLLU)
        assert sents[0].text == "She saw him ."

    def test_token_count_first_sentence(self):
        sents = parse_conllu(SAMPLE_CONLLU)
        # "She saw him ." — 4 tokens
        assert len(sents[0].tokens) == 4

    def test_token_fields(self):
        sents = parse_conllu(SAMPLE_CONLLU)
        she = sents[0].tokens[0]
        assert she.form == "She"
        assert she.lemma == "she"
        assert she.upos == "PRON"
        assert she.deprel == "nsubj"

    def test_morphological_case_extracted(self):
        sents = parse_conllu(SAMPLE_CONLLU)
        she = sents[0].tokens[0]  # Case=Nom
        him = sents[0].tokens[2]  # Case=Acc
        assert she.case_morphological == "Nom"
        assert him.case_morphological == "Acc"

    def test_empty_input(self):
        sents = parse_conllu("")
        assert sents == []

    def test_multiword_token_skipped(self):
        conllu = """\
# sent_id = mwt
# text = don't
1-2\tdon't\t_\t_\t_\t_\t_\t_\t_\t_
1\tdo\tdo\tAUX\tVBP\t_\t0\troot\t_\t_
2\tn't\tnot\tPART\tRB\t_\t1\tadvmod\t_\tSpaceAfter=No

"""
        sents = parse_conllu(conllu)
        assert len(sents) == 1
        # Only tokens 1 and 2, not the 1-2 entry
        assert len(sents[0].tokens) == 2


class TestDocumentPartition:
    @staticmethod
    def _sentences(document_ids: list[str]) -> list[Sentence]:
        return [
            Sentence(str(i), "fixture", [], document_id)
            for i, document_id in enumerate(document_ids)
        ]

    def test_partition_is_by_first_seen_document_not_sentence(self):
        sentences = self._sentences(["doc-a", "doc-a", "doc-a", "doc-b", "doc-c"])
        ordered, split = document_stride_partition(sentences, test_every=2)
        assert ordered == ["doc-a", "doc-b", "doc-c"]
        assert split == {"doc-a": "test", "doc-b": "train", "doc-c": "test"}
        assert len({split[sentence.document_id] for sentence in sentences[:3]}) == 1

    def test_retired_sentence_stride_would_leak_a_multi_sentence_document(self):
        sentences = self._sentences(["doc-a", "doc-a", "doc-a", "doc-b", "doc-c"])
        old_train = {sentence.document_id for index, sentence in enumerate(sentences) if index % 2}
        old_test = {
            sentence.document_id for index, sentence in enumerate(sentences) if index % 2 == 0
        }
        assert old_train & old_test == {"doc-a"}, "negative control no longer exposes leakage"

        _ordered, split = document_stride_partition(sentences, test_every=2)
        new_train = {document_id for document_id, arm in split.items() if arm == "train"}
        new_test = {document_id for document_id, arm in split.items() if arm == "test"}
        assert new_train.isdisjoint(new_test)

    @pytest.mark.parametrize(
        "document_ids",
        [[], [""], ["doc-a", "doc-b", "doc-a"], ["doc-a\0evil", "doc-b"]],
    )
    def test_partition_rejects_missing_or_reappearing_document_identity(self, document_ids):
        with pytest.raises(ValueError):
            document_stride_partition(self._sentences(document_ids), test_every=2)

    @pytest.mark.parametrize("stride", [True, 0, 1, 2.5])
    def test_partition_rejects_invalid_or_single_arm_stride(self, stride):
        with pytest.raises(ValueError):
            document_stride_partition(self._sentences(["doc-a", "doc-b"]), test_every=stride)

    def test_document_identity_digest_is_ordered_and_duplicate_refusing(self):
        forward = document_identity_sha256(["doc-a", "doc-b"])
        assert forward != document_identity_sha256(["doc-b", "doc-a"])
        with pytest.raises(ValueError, match="duplicate"):
            document_identity_sha256(["doc-a", "doc-a"])


class TestCaseOracle:
    def test_nom_from_morphology(self):
        """Morphological Case=Nom → CaseRole.NOM."""
        sents = parse_conllu(SAMPLE_CONLLU)
        she = sents[0].tokens[0]  # She, Case=Nom
        assert she.case_derived == CaseRole.NOM

    def test_acc_from_morphology(self):
        """Morphological Case=Acc → CaseRole.ACC."""
        sents = parse_conllu(SAMPLE_CONLLU)
        him = sents[0].tokens[2]  # him, Case=Acc
        assert him.case_derived == CaseRole.ACC

    def test_nsubj_deprel_maps_to_nom(self):
        """nsubj deprel → NOM (fallback when no Case feat)."""
        # Construct a minimal token without morphological case
        conllu = """\
# sent_id = t1
# text = dogs bark
1\tdogs\tdog\tNOUN\tNNS\tNumber=Plur\t2\tnsubj\t_\t_
2\tbark\tbark\tVERB\tVBP\t_\t0\troot\t_\t_

"""
        sents = parse_conllu(conllu)
        dogs = sents[0].tokens[0]
        assert dogs.case_morphological is None  # no Case feat
        assert dogs.case_derived == CaseRole.NOM  # derived from nsubj

    def test_iobj_maps_to_dat(self):
        """iobj deprel → DAT when no morphological case is present.

        NOTE: In SAMPLE_CONLLU, "her" has Case=Acc morphologically, so the
        morphological case wins and produces CaseRole.ACC. To test the iobj
        path without morphological interference, use a nominal with iobj but
        no Case feat.
        """
        # A noun with iobj deprel, no Case= feature
        conllu = """\
# sent_id = iobj_test
# text = John gave Mary the book
1\tJohn\tJohn\tPROPN\tNNP\tNumber=Sing\t2\tnsubj\t_\t_
2\tgave\tgive\tVERB\tVBD\tMood=Ind|Tense=Past|VerbForm=Fin\t0\troot\t_\t_
3\tMary\tMary\tPROPN\tNNP\tNumber=Sing\t2\tiobj\t_\t_
4\tthe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t5\tdet\t_\t_
5\tbook\tbook\tNOUN\tNN\tNumber=Sing\t2\tobj\t_\tSpaceAfter=No

"""
        sents = parse_conllu(conllu)
        mary = sents[0].tokens[2]  # id=3, Mary, deprel=iobj, no Case feat
        assert mary.deprel == "iobj"
        assert mary.case_morphological is None
        assert mary.case_derived == CaseRole.DAT

    def test_nmod_poss_maps_to_gen(self):
        """nmod:poss deprel → GEN."""
        sents = parse_conllu(SAMPLE_WITH_POSSESSIVE)
        # "cat" has deprel nmod:poss
        cat = sents[0].tokens[1]  # token id=2, "cat"
        assert cat.deprel == "nmod:poss"
        assert cat.case_derived == CaseRole.GEN

    def test_no_case_for_punct(self):
        """Punctuation gets no case assignment."""
        sents = parse_conllu(SAMPLE_CONLLU)
        period = sents[0].tokens[3]
        assert period.upos == "PUNCT"
        assert period.case_derived is None

    def test_tokens_with_case_filters(self):
        """tokens_with_case() returns only assigned tokens."""
        sents = parse_conllu(SAMPLE_CONLLU)
        with_case = sents[0].tokens_with_case()
        assert len(with_case) < len(sents[0].tokens)
        assert all(t.case_derived is not None for t in with_case)

    def test_adposition_rule_labels_the_adp_token_not_its_head(self):
        """NEGATIVE CONTROL: ``to`` and its nominal keep separate roles.

        The ADP's own lemma maps the ADP token to DAT. Its nominal head already
        carries ACC from morphology and is neither consulted nor overwritten.
        A head-transfer implementation would make these equal and fail here.
        """
        sentence = parse_conllu(SAMPLE_CONLLU)[1]
        adposition = next(token for token in sentence.tokens if token.form == "to")
        head = next(token for token in sentence.tokens if token.id == adposition.head)

        assert _apply_case_oracle(adposition) is CaseRole.DAT
        assert adposition.case_derived is CaseRole.DAT
        assert head.case_derived is CaseRole.ACC
        assert adposition.case_derived is not head.case_derived


class TestCaseOracleScopeAudit:
    @staticmethod
    def _token(
        token_id: int,
        *,
        form: str,
        lemma: str,
        upos: str,
        head: int,
        deprel: str,
    ) -> Token:
        return Token(
            id=token_id,
            form=form,
            lemma=lemma,
            upos=upos,
            xpos="_",
            head=head,
            deprel=deprel,
        )

    def test_head_audit_resolves_conllu_ids_not_list_offsets(self):
        """Head ID lookup survives deliberately non-contiguous, shuffled IDs."""
        adposition = self._token(2, form="to", lemma="to", upos="ADP", head=17, deprel="case")
        decoy = self._token(
            41, form="someone", lemma="someone", upos="NOUN", head=0, deprel="nsubj"
        )
        nominal_head = self._token(
            17, form="store", lemma="store", upos="NOUN", head=0, deprel="obl"
        )
        sentence = Sentence("ids", "to store", [adposition, decoy, nominal_head])

        audit = case_oracle_scope_audit([sentence])["adposition_token"]
        assert audit == {
            "tokens": 1,
            "morphological_case_tokens": 0,
            "head_resolved_tokens": 1,
            "head_unresolved_tokens": 0,
            "head_also_labeled_tokens": 1,
            "head_role_agreements": 0,
            "head_role_disagreements": 1,
        }

    def test_missing_head_is_counted_instead_of_silently_reassigned(self):
        """POSITIVE CONTROL: the audit's unresolved-head branch is live."""
        adposition = self._token(2, form="to", lemma="to", upos="ADP", head=99, deprel="case")
        sentence = Sentence("missing", "to", [adposition])
        audit = case_oracle_scope_audit([sentence])["adposition_token"]
        assert audit["head_resolved_tokens"] == 0
        assert audit["head_unresolved_tokens"] == 1
        assert audit["head_also_labeled_tokens"] == 0

    def test_duplicate_token_ids_make_head_lookup_ambiguous(self):
        """NEGATIVE CONTROL: last-write-wins ID resolution is forbidden."""
        first = self._token(2, form="to", lemma="to", upos="ADP", head=17, deprel="case")
        second = self._token(2, form="store", lemma="store", upos="NOUN", head=0, deprel="obl")
        with pytest.raises(ValueError, match="duplicate token IDs"):
            case_oracle_scope_audit([Sentence("duplicate", "to store", [first, second])])

    def test_real_ewt_scope_counts_reconcile_and_exclude_adpositions_from_fidelity(self):
        """The real raw shard proves the fidelity and head populations differ."""
        sentences = load_ud_ewt_dev(download=False)
        audit = case_oracle_scope_audit(sentences)
        paths = audit["derived_path_tokens"]
        fidelity = audit["heuristic_vs_gold"]
        fidelity_paths = fidelity["source_path_tokens"]
        adposition = audit["adposition_token"]

        assert set(paths) == set(CASE_ORACLE_PATHS)
        assert sum(paths.values()) == audit["full_derived_tokens"]
        assert sum(fidelity_paths.values()) == fidelity["tokens_with_both"]
        assert fidelity_paths["adposition_token"] == 0
        assert adposition["tokens"] == paths["adposition_token"] > 0
        assert adposition["morphological_case_tokens"] == 0
        assert (
            adposition["head_resolved_tokens"] + adposition["head_unresolved_tokens"]
            == adposition["tokens"]
        )
        assert adposition["head_also_labeled_tokens"] > adposition["tokens"] / 2
        assert (
            adposition["head_role_agreements"] + adposition["head_role_disagreements"]
            == adposition["head_also_labeled_tokens"]
        )
        assert adposition["head_role_disagreements"] > adposition["head_also_labeled_tokens"] / 2

        # Independent raw-token census: the production audit must not be able to
        # make its own scope identities true by returning internally consistent
        # but unrelated counts.
        independently_mapped_adpositions = sum(
            1
            for sentence in sentences
            for token in sentence.tokens
            if token.upos == "ADP"
            and token.deprel == "case"
            and token.lemma.lower() in ADPOSITION_TO_CASE
        )
        assert independently_mapped_adpositions == adposition["tokens"]


class TestCaseOracleStats:
    def test_stats_structure(self):
        sents = parse_conllu(SAMPLE_CONLLU)
        stats = case_oracle_stats(sents)
        assert "total_tokens" in stats
        assert "morphological_cases" in stats
        assert "derived_cases" in stats

    def test_total_tokens_correct(self):
        sents = parse_conllu(SAMPLE_CONLLU)
        stats = case_oracle_stats(sents)
        # 4 tokens in sent 1, 7 tokens in sent 2 = 11
        assert stats["total_tokens"] == 11

    def test_morphological_cases_count(self):
        sents = parse_conllu(SAMPLE_CONLLU)
        stats = case_oracle_stats(sents)
        # "She" (Nom), "him" (Acc), "I" (Nom), "her" (Acc) = 4 morphological
        assert sum(stats["morphological_cases"].values()) == 4

    def test_derived_cases_nonempty(self):
        sents = parse_conllu(SAMPLE_CONLLU)
        stats = case_oracle_stats(sents)
        assert sum(stats["derived_cases"].values()) > 0

    def test_nom_in_derived_cases(self):
        sents = parse_conllu(SAMPLE_CONLLU)
        stats = case_oracle_stats(sents)
        assert "NOM" in stats["derived_cases"]

    def test_empty_sentences(self):
        stats = case_oracle_stats([])
        assert stats["total_tokens"] == 0


class TestLoadUdEwtDev:
    def test_load_from_path_no_download(self, tmp_path):
        """Write a small CoNLL-U file and load it with download=False."""
        conllu_file = tmp_path / "test.conllu"
        conllu_file.write_text(SAMPLE_CONLLU, encoding="utf-8")
        payload = SAMPLE_CONLLU.encode("utf-8")
        sents = load_ud_ewt_dev(path=conllu_file, download=False, **_fixture_source(payload))
        assert len(sents) == 2

    def test_verified_snapshot_survives_path_replacement(self, tmp_path):
        """A later path state cannot become the bytes that the parser sees."""
        payload = SAMPLE_CONLLU.encode("utf-8")
        source_path = tmp_path / "source.conllu"
        source_path.write_bytes(payload)
        snapshot = read_ud_ewt_dev_bytes(
            source_path,
            download=False,
            **_fixture_source(payload),
        )

        source_path.write_text("# sent_id = evil\n", encoding="utf-8")
        assert [sentence.sent_id for sentence in parse_conllu(snapshot.decode("utf-8"))] == [
            "weblog-test-0001",
            "weblog-test-0002",
        ]
        assert Path(source_path).read_text(encoding="utf-8") == "# sent_id = evil\n"

    def test_raises_when_missing_and_no_download(self, tmp_path):
        missing = tmp_path / "missing.conllu"
        with pytest.raises(FileNotFoundError, match="UD EWT"):
            load_ud_ewt_dev(path=missing, download=False)

    def test_real_http_download_uses_the_explicit_corpus_cache(
        self, tmp_path, monkeypatch, local_http_server
    ):
        cache = tmp_path / "corpus-cache"
        route = "/en_ewt-ud-dev.conllu"
        payload = SAMPLE_CONLLU.encode("utf-8")
        monkeypatch.setenv("AILM_CORPUS_CACHE_DIR", str(cache))
        local_http_server.enqueue_bytes(route, payload, content_type="text/plain")
        source = _fixture_source(payload, url=local_http_server.base_url + route)

        downloaded = load_ud_ewt_dev(**source)
        replayed = load_ud_ewt_dev(download=False, **source)

        assert len(downloaded) == len(replayed) == 2
        assert (cache / "en_ewt-ud-dev.conllu").read_bytes() == payload
        requests_seen = local_http_server.requests_for(route)
        assert len(requests_seen) == 1
        assert requests_seen[0].method == "GET"

    def test_failed_http_download_does_not_create_a_cache_entry(self, tmp_path, local_http_server):
        destination = tmp_path / "en_ewt-ud-dev.conllu"
        route = "/missing-ewt.conllu"
        local_http_server.enqueue_bytes(route, b"unavailable", status=503)

        with pytest.raises(requests.HTTPError):
            load_ud_ewt_dev(path=destination, url=local_http_server.base_url + route)

        assert not destination.exists()

    def test_existing_wrong_bytes_are_rejected_instead_of_silently_replaced(
        self, tmp_path, local_http_server
    ):
        destination = tmp_path / "en_ewt-ud-dev.conllu"
        destination.write_bytes(b"modified corpus")
        route = "/would-be-replacement.conllu"
        local_http_server.enqueue_bytes(route, SAMPLE_CONLLU.encode())

        with pytest.raises(UDSourceIntegrityError, match="declared|does not match"):
            load_ud_ewt_dev(path=destination, url=local_http_server.base_url + route)

        assert destination.read_bytes() == b"modified corpus"
        assert not local_http_server.requests_for(route)

    def test_wrong_download_bytes_never_reach_the_cache(self, tmp_path, local_http_server):
        destination = tmp_path / "en_ewt-ud-dev.conllu"
        route = "/wrong-ewt.conllu"
        local_http_server.enqueue_bytes(route, b"wrong", content_type="text/plain")

        with pytest.raises(UDSourceIntegrityError, match="declared|does not match"):
            load_ud_ewt_dev(path=destination, url=local_http_server.base_url + route)

        assert not destination.exists()
        assert not list(tmp_path.glob(".en_ewt-ud-dev.conllu.*.tmp"))

    def test_oversize_response_is_refused_before_cache_publication(
        self, tmp_path, local_http_server
    ):
        destination = tmp_path / "en_ewt-ud-dev.conllu"
        route = "/oversize-en_ewt-ud-dev.conllu"
        expected = SAMPLE_CONLLU.encode("utf-8")
        local_http_server.enqueue_bytes(route, expected + b"unbounded suffix")
        source = _fixture_source(expected, url=local_http_server.base_url + route)

        with pytest.raises(UDSourceIntegrityError, match=r"declared .* expected"):
            load_ud_ewt_dev(path=destination, **source)

        assert not destination.exists()
        assert not list(tmp_path.glob(".en_ewt-ud-dev.conllu.*.tmp"))

    def test_misleading_content_length_is_refused(self, tmp_path, local_http_server):
        destination = tmp_path / "en_ewt-ud-dev.conllu"
        route = "/misleading-length-en_ewt-ud-dev.conllu"
        payload = SAMPLE_CONLLU.encode("utf-8")
        local_http_server.enqueue_bytes(
            route,
            payload,
            declared_length=len(payload) - 1,
        )
        source = _fixture_source(payload, url=local_http_server.base_url + route)

        with pytest.raises(UDSourceIntegrityError, match=r"declared .* expected"):
            load_ud_ewt_dev(path=destination, **source)

        assert not destination.exists()

    def test_redirect_is_refused_without_contacting_its_target(self, tmp_path, local_http_server):
        destination = tmp_path / "en_ewt-ud-dev.conllu"
        redirect_route = "/redirect-en_ewt-ud-dev.conllu"
        target_route = "/redirect-target-en_ewt-ud-dev.conllu"
        payload = SAMPLE_CONLLU.encode("utf-8")
        local_http_server.enqueue_bytes(
            redirect_route,
            b"",
            status=307,
            headers={"Location": local_http_server.base_url + target_route},
        )
        local_http_server.enqueue_bytes(target_route, payload)
        source = _fixture_source(payload, url=local_http_server.base_url + redirect_route)

        with pytest.raises(UDSourceIntegrityError, match="refused HTTP redirect"):
            load_ud_ewt_dev(path=destination, **source)

        assert len(local_http_server.requests_for(redirect_route)) == 1
        assert not local_http_server.requests_for(target_route)
        assert not destination.exists()

    def test_interrupted_body_never_reaches_the_cache(self, tmp_path, local_http_server):
        destination = tmp_path / "en_ewt-ud-dev.conllu"
        route = "/truncated-en_ewt-ud-dev.conllu"
        payload = SAMPLE_CONLLU.encode("utf-8")
        local_http_server.enqueue_bytes(
            route,
            payload,
            declared_length=len(payload),
            truncate_after=len(payload) // 2,
        )
        source = _fixture_source(payload, url=local_http_server.base_url + route)

        with pytest.raises(requests.RequestException):
            load_ud_ewt_dev(path=destination, **source)

        assert not destination.exists()
        assert not list(tmp_path.glob(".en_ewt-ud-dev.conllu.*.tmp"))

    def test_moving_branch_and_missing_revision_fail_before_download(
        self, tmp_path, local_http_server
    ):
        destination = tmp_path / "en_ewt-ud-dev.conllu"
        branch_url = local_http_server.base_url + "/master/en_ewt-ud-dev.conllu"

        with pytest.raises(UDSourceIntegrityError, match="moving branch"):
            load_ud_ewt_dev(path=destination, url=branch_url)
        with pytest.raises(UDSourceIntegrityError, match="full Git commit"):
            load_ud_ewt_dev(
                path=destination,
                url=local_http_server.base_url + "/en_ewt-ud-dev.conllu",
                source_revision="",
            )

        assert not local_http_server.requests_for("/master/en_ewt-ud-dev.conllu")
        assert not local_http_server.requests_for("/en_ewt-ud-dev.conllu")
        assert not destination.exists()

    def test_source_constants_and_derived_metadata_are_complete_and_live(self):
        assert UD_EWT_REVISION in UD_EWT_DEV_URL
        assert "/master/" not in UD_EWT_DEV_URL
        assert "/main/" not in UD_EWT_DEV_URL
        validate_ud_ewt_source_identity(
            url=UD_EWT_DEV_URL,
            source_revision=UD_EWT_REVISION,
            expected_sha256=UD_EWT_RAW_SHA256,
            expected_size_bytes=UD_EWT_RAW_SIZE_BYTES,
        )
        validate_ud_ewt_source_metadata(dict(UD_EWT_SOURCE_METADATA))

        missing = dict(UD_EWT_SOURCE_METADATA)
        missing.pop("source_revision")
        with pytest.raises(UDSourceIntegrityError, match="lacks pinned"):
            validate_ud_ewt_source_metadata(missing)
        changed = dict(UD_EWT_SOURCE_METADATA)
        changed["source_sha256"] = "0" * 64
        with pytest.raises(UDSourceIntegrityError, match="disagrees"):
            validate_ud_ewt_source_metadata(changed)

    def test_byte_verifier_has_positive_and_negative_controls(self):
        payload = SAMPLE_CONLLU.encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        assert (
            verify_ud_ewt_bytes(
                payload,
                expected_sha256=digest,
                expected_size_bytes=len(payload),
            )
            == digest
        )
        with pytest.raises(UDSourceIntegrityError, match="does not match"):
            verify_ud_ewt_bytes(
                payload + b"\n",
                expected_sha256=digest,
                expected_size_bytes=len(payload),
            )

    def test_safe_writer_returns_the_verified_path(self, tmp_path, local_http_server):
        payload = SAMPLE_CONLLU.encode("utf-8")
        destination = tmp_path / "nested" / "en_ewt-ud-dev.conllu"
        route = "/pinned/en_ewt-ud-dev.conllu"
        local_http_server.enqueue_bytes(route, payload, content_type="text/plain")
        source = _fixture_source(payload, url=local_http_server.base_url + route)

        resolved = ensure_ud_ewt_dev_path(path=destination, **source)

        assert resolved == destination
        assert resolved.read_bytes() == payload
        assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))
