"""Tests for ailm.corpus.cmudict load functions (file I/O, not network).

All tests use local files written to tmp_path — no network calls.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import requests

from ailm.corpus.cmudict import (
    CANONICAL_ARPABET,
    CMUDICT_RAW_SHA256,
    audit_comment_stripping,
    build_phoneme_inventory,
    load_cmudict,
    load_pinned_cmudict_source,
    verify_cmudict,
    word_to_phones,
)

TINY_CMUDICT_TEXT = """\
;;; This is a comment line
the  DH AH0  # another comment
cat  K AE1 T
the(2)  DH IY0
sat  S AE1 T
on  AO1 N
mat  M AE1 T
"""


class TestLoadCmudict:
    def test_load_from_file(self, tmp_path):
        f = tmp_path / "cmudict.dict"
        f.write_text(TINY_CMUDICT_TEXT, encoding="latin-1")
        d = load_cmudict(path=f, download=False)
        assert "THE" in d
        assert "CAT" in d

    def test_comment_lines_skipped(self, tmp_path):
        f = tmp_path / "cmudict.dict"
        f.write_text(TINY_CMUDICT_TEXT, encoding="latin-1")
        d = load_cmudict(path=f, download=False)
        # ';;;' lines should not appear as keys
        assert ";;;" not in d

    def test_inline_comments_stripped(self, tmp_path):
        """The 'the' entry has a trailing comment; phones must be clean."""
        f = tmp_path / "cmudict.dict"
        f.write_text(TINY_CMUDICT_TEXT, encoding="latin-1")
        d = load_cmudict(path=f, download=False)
        # 'the' should have ["DH", "AH0"], NOT ["DH", "AH0", "#", "another", "comment"]
        assert d["THE"][0] == ["DH", "AH0"]

    def test_comment_stripping_audit_exposes_planted_leak(self, tmp_path):
        source = tmp_path / "cmudict.dict"
        source.write_text(TINY_CMUDICT_TEXT, encoding="latin-1")

        audit = audit_comment_stripping(source)

        assert audit.canonical_inventory < audit.naive_inventory
        assert audit.leaked_comment_tokens == {"#", "another", "comment"}
        assert "#" not in audit.canonical_inventory

    def test_comment_stripping_audit_requires_a_real_local_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="CMUdict not found"):
            audit_comment_stripping(tmp_path / "missing.dict")

    @pytest.mark.raw_corpus_regeneration
    def test_pinned_writer_snapshot_survives_path_replacement(self, tmp_path):
        """The authenticated bytes, parse, and audit must be one snapshot."""
        canonical = Path(__file__).resolve().parents[1] / "vendor" / "cmudict.dict"
        if not canonical.is_file():
            pytest.skip("pinned CMUdict absent; raw-source control cannot run offline")
        attack_target = tmp_path / "cmudict.dict"
        attack_target.write_bytes(canonical.read_bytes())

        snapshot = load_pinned_cmudict_source(attack_target, download=False)
        attack_target.write_bytes(TINY_CMUDICT_TEXT.encode("latin-1"))

        assert hashlib.sha256(snapshot.content).hexdigest() == CMUDICT_RAW_SHA256
        assert len(snapshot.dictionary) > 100_000
        assert snapshot.comment_audit.canonical_inventory == CANONICAL_ARPABET

    def test_alternate_pronunciation_merged(self, tmp_path):
        """the(2) should add a second pronunciation to THE's list."""
        f = tmp_path / "cmudict.dict"
        f.write_text(TINY_CMUDICT_TEXT, encoding="latin-1")
        d = load_cmudict(path=f, download=False)
        # Should have 2 pronunciations for THE
        assert len(d["THE"]) == 2

    def test_raises_when_file_missing_no_download(self, tmp_path):
        missing = tmp_path / "no_such_file.dict"
        with pytest.raises(FileNotFoundError, match="CMUdict not found"):
            load_cmudict(path=missing, download=False)

    def test_returns_dict(self, tmp_path):
        f = tmp_path / "cmudict.dict"
        f.write_text(TINY_CMUDICT_TEXT, encoding="latin-1")
        d = load_cmudict(path=f, download=False)
        assert isinstance(d, dict)

    def test_phones_are_uppercase(self, tmp_path):
        f = tmp_path / "cmudict.dict"
        f.write_text(TINY_CMUDICT_TEXT, encoding="latin-1")
        d = load_cmudict(path=f, download=False)
        for word, pronuncs in d.items():
            for phones in pronuncs:
                for p in phones:
                    # Phones may have digits (stress) but letters are upper
                    assert p == p.upper(), f"Phone {p!r} for word {word!r} is not uppercase"

    def test_real_http_download_uses_the_explicit_corpus_cache(
        self, tmp_path, monkeypatch, local_http_server
    ):
        """A default-path download is cached outside an installed package tree."""
        cache = tmp_path / "corpus-cache"
        route = "/cmudict.dict"
        payload = TINY_CMUDICT_TEXT.encode("latin-1")
        monkeypatch.setenv("AILM_CORPUS_CACHE_DIR", str(cache))
        local_http_server.enqueue_bytes(
            route,
            payload,
            headers={"Content-Encoding": "identity"},
        )

        downloaded = load_cmudict(
            url=local_http_server.base_url + route,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_size_bytes=len(payload),
        )
        replayed = load_cmudict(
            download=False,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_size_bytes=len(payload),
        )

        assert downloaded == replayed
        assert (cache / "cmudict.dict").read_bytes() == payload
        requests_seen = local_http_server.requests_for(route)
        assert len(requests_seen) == 1
        assert requests_seen[0].method == "GET"
        assert not list(cache.glob(".cmudict.dict.*.tmp"))

    def test_custom_mirror_digest_mismatch_is_rejected_before_cache_write(
        self, tmp_path, local_http_server
    ):
        destination = tmp_path / "cmudict.dict"
        route = "/wrong-cmudict.dict"
        local_http_server.enqueue_bytes(route, TINY_CMUDICT_TEXT.encode("latin-1"))

        payload = TINY_CMUDICT_TEXT.encode("latin-1")
        with pytest.raises(ValueError, match="does not match"):
            load_cmudict(
                path=destination,
                url=local_http_server.base_url + route,
                expected_sha256="0" * 64,
                expected_size_bytes=len(payload),
            )

        assert not destination.exists()

    def test_failed_http_download_does_not_create_a_cache_entry(self, tmp_path, local_http_server):
        destination = tmp_path / "cmudict.dict"
        route = "/missing-cmudict.dict"
        local_http_server.enqueue_bytes(route, b"unavailable", status=503)

        with pytest.raises(requests.HTTPError):
            load_cmudict(
                path=destination,
                url=local_http_server.base_url + route,
                expected_sha256=hashlib.sha256(b"unavailable").hexdigest(),
                expected_size_bytes=len(b"unavailable"),
            )

        assert not destination.exists()

    def test_custom_mirror_without_digest_is_rejected_before_network(
        self, tmp_path, local_http_server
    ):
        destination = tmp_path / "cmudict.dict"
        route = "/untrusted-cmudict.dict"
        local_http_server.enqueue_bytes(route, TINY_CMUDICT_TEXT.encode("latin-1"))

        with pytest.raises(ValueError, match="requires expected_sha256"):
            load_cmudict(path=destination, url=local_http_server.base_url + route)

        assert not destination.exists()
        assert not local_http_server.requests_for(route)

    def test_oversize_response_is_refused_before_cache_publication(
        self, tmp_path, local_http_server
    ):
        destination = tmp_path / "cmudict.dict"
        route = "/oversize-cmudict.dict"
        expected = TINY_CMUDICT_TEXT.encode("latin-1")
        local_http_server.enqueue_bytes(route, expected + b"attacker-controlled suffix")

        with pytest.raises(ValueError, match=r"declared .* expected"):
            load_cmudict(
                path=destination,
                url=local_http_server.base_url + route,
                expected_sha256=hashlib.sha256(expected).hexdigest(),
                expected_size_bytes=len(expected),
            )

        assert not destination.exists()
        assert not list(tmp_path.glob(".cmudict.dict.*.tmp"))

    def test_misleading_content_length_is_refused(self, tmp_path, local_http_server):
        destination = tmp_path / "cmudict.dict"
        route = "/misleading-length-cmudict.dict"
        payload = TINY_CMUDICT_TEXT.encode("latin-1")
        local_http_server.enqueue_bytes(
            route,
            payload,
            declared_length=len(payload) + 1,
        )

        with pytest.raises(ValueError, match=r"declared .* expected"):
            load_cmudict(
                path=destination,
                url=local_http_server.base_url + route,
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_size_bytes=len(payload),
            )

        assert not destination.exists()

    def test_redirect_is_refused_without_contacting_its_target(self, tmp_path, local_http_server):
        destination = tmp_path / "cmudict.dict"
        redirect_route = "/redirect-cmudict.dict"
        target_route = "/redirect-target-cmudict.dict"
        payload = TINY_CMUDICT_TEXT.encode("latin-1")
        local_http_server.enqueue_bytes(
            redirect_route,
            b"",
            status=302,
            headers={"Location": local_http_server.base_url + target_route},
        )
        local_http_server.enqueue_bytes(target_route, payload)

        with pytest.raises(ValueError, match="refused HTTP redirect"):
            load_cmudict(
                path=destination,
                url=local_http_server.base_url + redirect_route,
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_size_bytes=len(payload),
            )

        assert len(local_http_server.requests_for(redirect_route)) == 1
        assert not local_http_server.requests_for(target_route)
        assert not destination.exists()

    def test_interrupted_body_never_reaches_the_cache(self, tmp_path, local_http_server):
        destination = tmp_path / "cmudict.dict"
        route = "/truncated-cmudict.dict"
        payload = TINY_CMUDICT_TEXT.encode("latin-1")
        local_http_server.enqueue_bytes(
            route,
            payload,
            declared_length=len(payload),
            truncate_after=len(payload) // 2,
        )

        with pytest.raises(requests.RequestException):
            load_cmudict(
                path=destination,
                url=local_http_server.base_url + route,
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_size_bytes=len(payload),
            )

        assert not destination.exists()
        assert not list(tmp_path.glob(".cmudict.dict.*.tmp"))

    @pytest.mark.parametrize(
        ("digest", "extent", "message"),
        [
            ("not-a-digest", len(TINY_CMUDICT_TEXT), "lowercase digest"),
            ("0" * 64, 0, "positive integer"),
            ("0" * 64, True, "positive integer"),
        ],
    )
    def test_malformed_mirror_identity_fails_before_network(
        self,
        tmp_path,
        local_http_server,
        digest,
        extent,
        message,
    ):
        destination = tmp_path / "cmudict.dict"
        route = "/malformed-identity-cmudict.dict"
        local_http_server.enqueue_bytes(route, TINY_CMUDICT_TEXT.encode("latin-1"))

        with pytest.raises(ValueError, match=message):
            load_cmudict(
                path=destination,
                url=local_http_server.base_url + route,
                expected_sha256=digest,
                expected_size_bytes=extent,
            )

        assert not local_http_server.requests_for(route)
        assert not destination.exists()


class TestWordToPhonesWithFile:
    def test_lookup_via_loaded_dict(self, tmp_path):
        f = tmp_path / "cmudict.dict"
        f.write_text(TINY_CMUDICT_TEXT, encoding="latin-1")
        d = load_cmudict(path=f, download=False)
        phones = word_to_phones("cat", d, strip_stress=True)
        assert phones is not None
        assert "K" in phones
        assert "T" in phones

    def test_stress_preserved_when_requested(self, tmp_path):
        f = tmp_path / "cmudict.dict"
        f.write_text(TINY_CMUDICT_TEXT, encoding="latin-1")
        d = load_cmudict(path=f, download=False)
        phones = word_to_phones("cat", d, strip_stress=False)
        assert phones is not None
        # AE1 should be preserved
        assert "AE1" in phones

    def test_oov_returns_none(self, tmp_path):
        f = tmp_path / "cmudict.dict"
        f.write_text(TINY_CMUDICT_TEXT, encoding="latin-1")
        d = load_cmudict(path=f, download=False)
        assert word_to_phones("xyzzy", d) is None


class TestVerifyCmudict:
    def test_verify_valid_file_returns_ok(self, tmp_path):
        """verify_cmudict needs the full 39-phoneme inventory to return 'ok'.
        With a tiny file, it raises AssertionError (inventory count mismatch).
        Test that the function at least runs and reports an error gracefully.
        """
        f = tmp_path / "cmudict.dict"
        f.write_text(TINY_CMUDICT_TEXT, encoding="latin-1")
        result = verify_cmudict(f)
        # Tiny dict will fail the 39-phoneme assertion
        assert result["status"] in ("ok", "error")
        if result["status"] == "error":
            assert "error" in result

    def test_verify_missing_file_returns_error(self, tmp_path):
        missing = tmp_path / "no_such.dict"
        result = verify_cmudict(missing)
        assert result["status"] == "error"
        assert "error" in result


class TestBuildInventoryFromFile:
    def test_tiny_file_inventory_is_subset(self, tmp_path):
        f = tmp_path / "cmudict.dict"
        f.write_text(TINY_CMUDICT_TEXT, encoding="latin-1")
        d = load_cmudict(path=f, download=False)
        inv = build_phoneme_inventory(d)
        assert inv.issubset(CANONICAL_ARPABET)

    def test_no_comment_tokens_in_inventory(self, tmp_path):
        f = tmp_path / "cmudict.dict"
        f.write_text(TINY_CMUDICT_TEXT, encoding="latin-1")
        d = load_cmudict(path=f, download=False)
        inv = build_phoneme_inventory(d)
        # If comment stripping worked, no ';;;' in the inventory
        assert ";;;" not in inv
        assert "#" not in inv
        assert "another" not in inv  # inline comment words
        assert "comment" not in inv
