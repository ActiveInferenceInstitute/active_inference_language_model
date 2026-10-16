"""Dedicated controls for the stipulated aligned spelling-to-pronunciation arm.

Protocol P2 of ``docs/extension_protocols.md`` §3. Every control runs real
generation, real files, and real engine execution under ``tmp_path`` — no
mocks, no network, no Lean build, no claim writes. The negative controls plant
corruption into real artifact bytes and require the validator to refuse for
the intended reason; the collision control forces a real exhaustion of the
declared resample budget through a degenerate (length-1) word space rather
than stubbing anything.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ailm.benchmarks import aligned_g2p as ag
from ailm.levels.phoneme import PHONES
from ailm.levels.synthetic import SYNTHETIC_LETTERS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEDGER = PROJECT_ROOT / "data" / "claims.json"
LEDGER_SHA_AT_IMPORT = hashlib.sha256(LEDGER.read_bytes()).hexdigest()


def _miniature():
    """A fixture-sized but fully real artifact payload with its splits."""

    correspondence = ag.stipulated_correspondence()
    train, test, stats = ag.build_splits(correspondence, n_train=60, n_test=12, word_length=4)
    payload = ag.build_artifact_payload(correspondence, train, test, stats)
    return correspondence, train, test, payload


@pytest.fixture(scope="module")
def miniature():
    return _miniature()


@pytest.fixture(scope="module")
def english_train_pairs():
    from ailm.levels.phoneme import g2p_split, load_phoneme_counts

    train, _test = g2p_split(load_phoneme_counts())
    return train


# ── The stipulated correspondence ────────────────────────────────────────────


def test_the_correspondence_is_complete_balanced_and_deterministic():
    first = ag.stipulated_correspondence()
    second = ag.stipulated_correspondence()
    assert first == second
    assert sorted(first) == sorted(SYNTHETIC_LETTERS)
    codomain = set(first.values())
    assert codomain <= set(PHONES)
    assert len(codomain) == ag.ALIGNED_G2P_CODEDOMAIN_SIZE


def test_word_generation_is_deterministic_and_drops_the_fixed_start():
    assert ag.stipulated_word(20261300) == ag.stipulated_word(20261300)
    words = [ag.stipulated_word(seed) for seed in range(20261300, 20261320)]
    assert all(len(word) == ag.ALIGNED_G2P_WORD_LENGTH for word in words)
    assert all(set(word) <= set(SYNTHETIC_LETTERS) for word in words)
    # The chain starts at a fixed symbol; the window drops it, so the words are
    # not all pinned to one first letter.
    assert len({word[0] for word in words}) > 1


def test_seed_domains_are_disjoint_by_construction():
    assert (
        ag.ALIGNED_G2P_HELD_OUT_SEED_BASE
        == ag.ALIGNED_G2P_TRAIN_SEED_BASE + ag.ALIGNED_G2P_TRAIN_WORDS
    )
    assert ag.ALIGNED_G2P_RESAMPLE_ATTEMPTS_CAP == 64
    with pytest.raises(ValueError, match="held-out seed domain"):
        ag.build_splits(
            ag.stipulated_correspondence(),
            n_train=40,
            n_test=10,
            word_length=4,
            train_seed_base=100,
            held_out_seed_base=120,
        )


# ── Split construction ───────────────────────────────────────────────────────


def test_splits_are_disjoint_and_carry_aligned_gold(miniature):
    _correspondence, train, test, _payload = miniature
    train_words = [word for word, _ in train]
    test_words = [word for word, _ in test]
    assert len(train_words) == len(set(train_words)) == 60
    assert len(test_words) == len(set(test_words)) == 12
    assert not set(train_words) & set(test_words)
    correspondence = ag.stipulated_correspondence()
    for word, phones in train + test:
        assert len(word) == len(phones) == 4
        assert set(word) <= set(correspondence)
        assert set(phones) <= set(PHONES)
        assert phones == [correspondence[letter] for letter in word]


def test_collision_exhaustion_refuses_before_any_output():
    # Length-2 words have exactly 26**2 = 676 possible spellings.  Demanding
    # 700 distinct words exhausts the space through real generation: every
    # resample draws from the same finite space, so the declared budget runs
    # out and the builder refuses with nothing written.
    with pytest.raises(SystemExit, match="could not draw a fresh word"):
        ag.build_splits(
            ag.stipulated_correspondence(),
            n_train=700,
            n_test=2,
            word_length=2,
            train_seed_base=3_000_000,
            held_out_seed_base=4_000_000,
            attempts_cap=8,
        )


# ── Artifact validation: the refusals ────────────────────────────────────────


def test_artifact_payload_validates_and_round_trips_through_the_loader(tmp_path, miniature):
    _correspondence, _train, _test, payload = miniature
    ag.validate_artifact_payload(payload)
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / ag.ALIGNED_G2P_ARTIFACT_NAME).write_text(json.dumps(payload), encoding="utf-8")
    loaded = ag.load_aligned_g2p(vendor_dir=vendor)
    assert loaded == payload


def test_validation_refuses_a_corrupted_correspondence(miniature):
    payload = copy.deepcopy(miniature[3])
    tampered = copy.deepcopy(payload)
    letter = sorted(tampered["correspondence"])[0]
    tampered["correspondence"][letter] = next(
        phone for phone in PHONES if phone not in set(tampered["correspondence"].values())
    )
    with pytest.raises(ValueError, match="correspondence_table_digest"):
        ag.validate_artifact_payload(tampered)
    # If the digest is re-stamped too, the per-position gold check catches it.
    tampered["provenance"]["correspondence_table_digest"] = ag._canonical_digest(
        tampered["correspondence"]
    )
    with pytest.raises(ValueError, match="stipulated correspondence applied per position"):
        ag.validate_artifact_payload(tampered)


def test_validation_refuses_incomplete_coverage(miniature):
    payload = copy.deepcopy(miniature[3])
    tampered = copy.deepcopy(payload)
    dropped = sorted(tampered["correspondence"])[0]
    del tampered["correspondence"][dropped]
    with pytest.raises(ValueError, match="complete over exactly"):
        ag.validate_artifact_payload(tampered)


def test_validation_refuses_out_of_inventory_values(miniature):
    payload = copy.deepcopy(miniature[3])
    tampered = copy.deepcopy(payload)
    letter = sorted(tampered["correspondence"])[0]
    tampered["correspondence"][letter] = "ZZ"
    with pytest.raises(ValueError, match="canonical 39-phone inventory"):
        ag.validate_artifact_payload(tampered)


def test_validation_refuses_duplicates_and_cross_split_collisions(miniature):
    payload = copy.deepcopy(miniature[3])
    train = miniature[1]
    duplicated = copy.deepcopy(payload)
    duplicated["splits"]["train"][-1] = list(duplicated["splits"]["train"][0])
    with pytest.raises(ValueError, match="duplicate spellings"):
        ag.validate_artifact_payload(duplicated)

    collided = copy.deepcopy(payload)
    train_word, train_gold = train[0]
    collided["splits"]["test"][0] = [train_word, " ".join(train_gold)]
    with pytest.raises(ValueError, match="cross-split collision"):
        ag.validate_artifact_payload(collided)


def test_validation_refuses_misaligned_gold(miniature):
    payload = copy.deepcopy(miniature[3])
    tampered = copy.deepcopy(payload)
    spelling, phones = tampered["splits"]["test"][0]
    swapped = phones.split()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    tampered["splits"]["test"][0] = [spelling, " ".join(swapped)]
    with pytest.raises(ValueError, match="stipulated correspondence applied per position"):
        ag.validate_artifact_payload(tampered)


def test_validation_refuses_wrong_length_and_out_of_inventory_letters(miniature):
    payload = copy.deepcopy(miniature[3])
    short = copy.deepcopy(payload)
    short["splits"]["test"][0][0] = short["splits"]["test"][0][0][:3]
    with pytest.raises(ValueError, match="word_length_L"):
        ag.validate_artifact_payload(short)

    outside = copy.deepcopy(payload)
    outside["splits"]["test"][0][0] = "a1c" + outside["splits"]["test"][0][0][3:]
    with pytest.raises(ValueError, match="outside the stipulated"):
        ag.validate_artifact_payload(outside)


def test_digests_refuse_tampering(miniature):
    payload = copy.deepcopy(miniature[3])
    tampered = copy.deepcopy(payload)
    original_word = tampered["splits"]["train"][0][0]
    taken = {row[0] for row in tampered["splits"]["train"]}
    replacement = next(
        candidate
        for candidate in (
            letter + original_word[1:] for letter in SYNTHETIC_LETTERS if letter != original_word[0]
        )
        if candidate not in taken
    )
    tampered["splits"]["train"][0][0] = replacement
    tampered["splits"]["train"][0][1] = " ".join(
        tampered["correspondence"][letter] for letter in replacement
    )
    with pytest.raises(ValueError, match="train_words_digest"):
        ag.validate_artifact_payload(tampered)


def test_validation_refuses_overlapping_seed_domains(miniature):
    payload = copy.deepcopy(miniature[3])
    tampered = copy.deepcopy(payload)
    tampered["provenance"]["held_out_seed"] = tampered["provenance"]["corpus_seed"]
    with pytest.raises(ValueError, match="held-out seed domain overlaps"):
        ag.validate_artifact_payload(tampered)


# ── The native model and its controls ────────────────────────────────────────


def test_the_native_model_executes_and_the_trivial_lookup_is_exact(miniature, english_train_pairs):
    correspondence, train, test, _payload = miniature
    report = ag.evaluate_aligned_g2p(
        correspondence, train, test, english_train_pairs=english_train_pairs
    )
    arms = report["arms"]
    native = arms["native_stipulated_engine"]
    lookup = arms["t_lookup_baseline"]
    floor = arms["a3_uniform_floor"]["position_floor"]
    # The trivial map is exactly right on T-generated data, by construction.
    assert lookup["position_accuracy"] == 1.0
    assert lookup["exact_accuracy"] == 1.0
    # The shipped engine actually ran: real predictions, valid shapes, above floor.
    assert 0.0 <= native["position_accuracy"] <= 1.0
    assert native["position_accuracy"] > floor
    assert all(len(p) == 4 for p in native["predictions"])
    assert all(set(p) <= set(PHONES) for p in native["predictions"])
    # The A1 English-transfer control ran on the identical positions.
    english = arms["a1_english_transfer"]
    assert 0.0 <= english["position_accuracy"] <= 1.0
    assert len(english["predictions"]) == len(test)
    # The A2 broken-map control is a live negative: it never outperforms native.
    broken = arms["a2_broken_correspondence"]
    assert broken["position_accuracy"] < native["position_accuracy"]
    assert report["pre_registration"]["a2_clears_bar"] is False


def test_the_broken_map_is_a_total_derangement_of_the_codomain(miniature):
    correspondence, _train, _test, _payload = miniature
    broken = ag.broken_correspondence(correspondence)
    assert sorted(broken) == sorted(SYNTHETIC_LETTERS)
    assert set(broken.values()) == set(correspondence.values())
    assert all(broken[letter] != phone for letter, phone in correspondence.items())
    assert ag.broken_correspondence(correspondence) == broken


# ── L1 / L2: prompt purity and gold isolation ────────────────────────────────


def test_prompt_purity_l1(miniature):
    _correspondence, _train, test, _payload = miniature
    words = [word for word, _ in test]
    golds = [" ".join(phones) for _, phones in test]
    prompts = [ag.probe_prompt(word) for word in words]
    assert prompts == [ag.probe_prompt(word) for word in words]  # pure function
    for prompt, word, gold in zip(prompts, words, golds, strict=True):
        assert word in prompt
        assert gold not in prompt
        assert prompt == ag.build_prompt("g2p", word)
        # Quoted response-schema tokens are the placeholders only; no real
        # phone symbol is quoted anywhere in the prompt.
        quoted = set(re.findall(r'"([A-Za-z0-9]+)"', prompt))
        assert quoted <= {"answer", "PHONE1", "PHONE2"}, quoted


def test_gold_isolation_l2(miniature):
    _correspondence, _train, test, _payload = miniature
    words = [word for word, _ in test]
    payloads = ag.probe_request_payloads(words)
    assert len(payloads) == len(words)
    for payload, (_word, phones) in zip(payloads, test, strict=True):
        blob = json.dumps(payload)
        assert " ".join(phones) not in blob
        quoted = set(re.findall(r'"([A-Za-z0-9]+)"', payload["prompt"]))
        assert quoted <= {"answer", "PHONE1", "PHONE2"}, quoted
        assert "prompt" in payload and "task" in payload


# ── Train-only demonstrations and the A4 control ─────────────────────────────


def test_demonstrations_are_train_only_and_the_a4_control_deranges(miniature):
    _correspondence, train, _test, _payload = miniature
    examples, deranged = ag.demonstration_pairs(train, k=3, position_index=0)
    train_gold = {word: phones for word, phones in train}
    for (word, phones), (dword, _dphones) in zip(examples, deranged, strict=True):
        assert word == dword  # the spellings stay; only the gold assignment moves
        assert phones == train_gold[word]  # drawn exclusively from train pairs
    assert [tuple(p) for _, p in deranged] != [tuple(p) for _, p in examples]
    assert sorted(tuple(p) for _, p in deranged) == sorted(tuple(p) for _, p in examples)
    assert all(dphones != train_gold[dword] for dword, dphones in deranged)


def test_demonstration_draw_refuses_an_undecomposable_pool():
    train, _test, _stats = ag.build_splits(
        ag.stipulated_correspondence(), n_train=4, n_test=2, word_length=4
    )
    with pytest.raises(ValueError, match="demonstration refusal"):
        ag.demonstration_pairs(train, k=1, position_index=0, attempts_cap=8)


# ── The registry's fourth member ─────────────────────────────────────────────


def test_the_registry_carries_the_fourth_member():
    from ailm.benchmarks.synthetic_protocol import (
        SYNTHETIC_ALIGNED_G2P_PROTOCOL,
        SYNTHETIC_INCONTEXT_PROTOCOL,
        SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL,
        SYNTHETIC_ZERO_SHOT_PROTOCOL,
    )

    protocol = SYNTHETIC_ALIGNED_G2P_PROTOCOL
    assert protocol.version == "synthetic-aligned-g2p-v1"
    assert protocol.word_length == ag.ALIGNED_G2P_WORD_LENGTH == 6
    assert protocol.resample_attempts_cap == 64
    assert protocol.train_words == ag.ALIGNED_G2P_TRAIN_WORDS
    assert protocol.test_words == ag.ALIGNED_G2P_TEST_WORDS
    assert protocol.held_out_seed_base == protocol.train_seed_base + protocol.train_words
    assert protocol.response_inventory == tuple(PHONES)
    cache_keys = {
        SYNTHETIC_ZERO_SHOT_PROTOCOL.cache_protocol,
        SYNTHETIC_INCONTEXT_PROTOCOL.cache_protocol,
        SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL.cache_protocol,
        protocol.cache_protocol,
    }
    assert len(cache_keys) == 4
    with pytest.raises(FrozenInstanceError):
        protocol.word_length = 7  # type: ignore[misc]


# ── The thin CLIs (real subprocesses, true exits) ────────────────────────────


@pytest.mark.timeout(600)
def test_generator_cli_is_deterministic_and_refuses_collisions(tmp_path):
    generator = PROJECT_ROOT / "scripts" / "generate_synthetic_aligned_g2p.py"
    out1 = tmp_path / "artifact_a.json"
    out2 = tmp_path / "artifact_b.json"
    fixture_args = ["--train-words", "40", "--test-words", "8", "--word-length", "4"]
    for output in (out1, out2):
        result = subprocess.run(
            [sys.executable, str(generator), "--output", str(output), *fixture_args],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, result.stderr
    assert out1.read_bytes() == out2.read_bytes()
    assert json.loads(out1.read_text(encoding="utf-8"))["schema"] == ag.ALIGNED_G2P_SCHEMA

    refused = tmp_path / "refused.json"
    collision = subprocess.run(
        [
            sys.executable,
            str(generator),
            "--output",
            str(refused),
            "--word-length",
            "1",
            "--train-words",
            "40",
            "--test-words",
            "8",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(PROJECT_ROOT),
    )
    assert collision.returncode != 0
    assert "word_length must be at least 2" in collision.stderr
    assert not refused.exists()


@pytest.mark.timeout(600)
def test_evaluator_cli_reports_offline_arms(tmp_path, miniature):
    _correspondence, _train, _test, payload = miniature
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / ag.ALIGNED_G2P_ARTIFACT_NAME).write_text(json.dumps(payload), encoding="utf-8")
    evaluator = PROJECT_ROOT / "scripts" / "evaluate_synthetic_aligned_g2p.py"
    report_path = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(evaluator),
            "--artifact",
            str(vendor / ag.ALIGNED_G2P_ARTIFACT_NAME),
            "--report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert 0.0 <= report["arms"]["a2_broken_correspondence"]["position_accuracy"] <= 1.0
    assert set(report["arms"]) == {
        "native_stipulated_engine",
        "t_lookup_baseline",
        "a1_english_transfer",
        "a2_broken_correspondence",
        "a3_uniform_floor",
    }
    assert report["arms"]["t_lookup_baseline"]["position_accuracy"] == 1.0
    assert report["policies"]["smoothing"]
    assert report["pre_registration"]["bar_value"] is not None


@pytest.mark.timeout(600)
def test_evaluator_cli_refuses_an_absent_artifact(tmp_path):
    evaluator = PROJECT_ROOT / "scripts" / "evaluate_synthetic_aligned_g2p.py"
    missing = tmp_path / "vendor"
    missing.mkdir()
    result = subprocess.run(
        [sys.executable, str(evaluator), "--artifact", str(missing / ag.ALIGNED_G2P_ARTIFACT_NAME)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode != 0
    assert "generate_synthetic_aligned_g2p.py" in result.stderr


def test_no_claim_was_written_by_any_control():
    """The whole unit is offline: the ledger's bytes must be exactly as inherited.

    The baseline is snapshotted when this module imports, not pinned to an
    absolute hash: the pin re-broke on every ledger commit, and the test's
    actual claim is that the control runs themselves write nothing.
    """

    assert hashlib.sha256(LEDGER.read_bytes()).hexdigest() == LEDGER_SHA_AT_IMPORT
