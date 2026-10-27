"""Tests for the AILM multimodal interactive dashboard engine and server.

The shape assertions below came first and passed throughout a period in which
the dashboard's Level 5 classifier scored **below** the majority-class
baseline: the widened alphabet's AUX symbol was unreachable, punctuation was
injected as a case observation, and the Level 6 phase was pinned to "opening"
for every input. Structural checks cannot see any of that. The behavioral
tests that follow exist for exactly that reason, and each carries a negative
control — a variant that must NOT fire — so a test cannot pass by asserting
something that is true of any model at all.
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from functools import partial
from pathlib import Path
from socketserver import BaseRequestHandler
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from ailm.dashboard.config import DashboardConfig
from ailm.dashboard.engine import (
    AUX_SYMBOL,
    MAX_CONTINUATION_WORDS,
    MAX_HISTORY_TURNS,
    N_WIDENED_SYMBOLS,
    VERB_SYMBOL,
    generate_next_tokens,
    generate_topdown_continuation,
    get_dashboard_models,
    infer_sentence_hierarchy,
)


def test_dashboard_models_preload():
    """Verify preloaded models contain all level matrices."""
    models = get_dashboard_models()
    assert "A_letter" in models
    assert "A_g2p" in models
    assert "B_g2p" in models
    assert "word_artifact" in models
    assert "A_case" in models
    assert "B_case" in models
    assert "clause_models" in models
    assert models["A_letter"].shape == (30, 30)
    assert models["B_g2p"].shape == (39, 39)


def test_dashboard_infer_sentence_hierarchy():
    """Verify hierarchical inference execution over a test sentence."""
    text = "The quick brown fox jumps over the lazy dog."
    result = infer_sentence_hierarchy(text)

    assert "text" in result
    assert "levels" in result
    assert "topdown_continuation" in result
    assert len(result["topdown_continuation"]["continuation_text"]) > 0

    # Check Level 1
    l1 = result["levels"]["l1_letter"]
    assert len(l1["steps"]) > 0
    first_step = l1["steps"][0]
    assert "prior" in first_step
    assert "posterior" in first_step
    assert "entropy" in first_step
    assert "top_predictions" in first_step
    assert len(first_step["top_predictions"]) == 5

    # Check Level 3
    l3 = result["levels"]["l3_word"]
    assert len(l3["words"]) > 0
    first_word = l3["words"][0]
    assert "raw_word" in first_word
    assert "decoded_phones" in first_word
    assert "phone_steps" in first_word
    assert "case_posterior" in first_word
    assert "top_case" in first_word

    # Check Level 5 & 6
    l5 = result["levels"]["l5_clause"]
    assert l5["predicted_type"] in ["declarative", "interrogative"]
    assert len(l5["posterior"]) == 2

    l6 = result["levels"]["l6_narrative"]
    assert l6["current_phase"] in ["opening", "middle", "closing"]


def test_dashboard_generate_next_tokens():
    """Verify top-down EFE active inference token rollout."""
    prefix = list("the quick")
    rollout = generate_next_tokens(prefix, max_len=10)
    assert len(rollout) > 0
    assert all(isinstance(c, str) for c in rollout)


def test_dashboard_static_files_exist():
    """Verify dashboard static index.html exists."""
    static_html = (
        Path(__file__).resolve().parents[1] / "src" / "ailm" / "dashboard" / "static" / "index.html"
    )
    assert static_html.exists()
    content = _dashboard_html()
    assert "AILM Level Stack Inspector" in content
    assert "SpeechRecognition" in content or "webkitSpeechRecognition" in content
    assert "speechSynthesis" in content
    assert "canvas" in content


def test_dashboard_edge_cases_and_robustness():
    """Verify engine edge-cases: empty inputs, punctuation, long text, OOV tokens."""
    from ailm.dashboard.engine import generate_topdown_continuation

    edge_inputs = [
        "",
        "   ",
        "???!!!...",
        "1234567890",
        "xyzzy qwertyuiop asdfghjkl",
        "A" * 100,
        "The " * 30,
    ]
    for text in edge_inputs:
        res = infer_sentence_hierarchy(text, n_continuation_words=5)
        assert "text" in res
        assert "levels" in res
        assert "topdown_continuation" in res
        assert len(res["levels"]["l1_letter"]["steps"]) > 0

    # Top-down continuation edge cases
    c_res1 = generate_topdown_continuation("", n_words=5)
    assert len(c_res1["continuation_text"]) > 0
    assert len(c_res1["steps"]) > 0

    c_res2 = generate_topdown_continuation("unknownword123", n_words=3)
    assert len(c_res2["continuation_text"]) > 0


# ── BEHAVIOR: the widened Level 5 alphabet ──────────────────────────────────


def test_the_widened_alphabet_has_exactly_one_definition():
    """The dashboard and the probe import the alphabet; neither declares it.

    This test used to compare two copies of the symbol indices, because there
    were two. Comparing transcriptions is the second-best outcome; having one is
    the first. `ailm.levels.widened` now owns the indices, the extraction and
    the chain fit, and both consumers import them.

    So the check is no longer "do the copies agree" but "is there still only
    one" — a redeclaration anywhere is the defect, and a matching redeclaration
    is just a defect that has not diverged yet.
    """
    from ailm.levels import widened

    assert (widened.VERB_SYMBOL, widened.AUX_SYMBOL, widened.N_WIDENED_SYMBOLS) == (8, 9, 10)
    assert (VERB_SYMBOL, AUX_SYMBOL, N_WIDENED_SYMBOLS) == (
        widened.VERB_SYMBOL,
        widened.AUX_SYMBOL,
        widened.N_WIDENED_SYMBOLS,
    ), "the dashboard re-exports something other than the shared constants"

    root = Path(__file__).resolve().parents[1]
    for path in (
        root / "src" / "ailm" / "dashboard" / "engine.py",
        root / "scripts" / "probe_l5_widened_alphabet.py",
    ):
        source = path.read_text(encoding="utf-8")
        for name in ("VERB_SYMBOL", "AUX_SYMBOL", "N_WIDENED_SYMBOLS"):
            assert not re.search(rf"^{name}\s*=\s*\d", source, re.M), (
                f"{path.name} redeclares {name}; it must import it"
            )
        assert not re.search(r"^def fit_markov_chain|def _fit_chain", source, re.M), (
            f"{path.name} defines its own chain fit; it must import one"
        )


def test_aux_symbol_is_reachable_at_inference():
    """POSITIVE: an auxiliary emits symbol 9. NEGATIVE: a sentence without one does not.

    The regression this pins: the part-of-speech tag was rewritten to the
    case-emission default *before* the widened-symbol decision, and AUX has no
    column in that emission. Every auxiliary became a NOUN, symbol 9 became
    unreachable, and the widened alphabet silently collapsed to nine symbols —
    losing subject-auxiliary inversion, the strongest interrogative cue in
    English. The negative control is what makes this test informative: without
    it, an engine that emitted AUX for every token would pass.
    """
    with_aux = infer_sentence_hierarchy("is he going to the store ?")
    assert AUX_SYMBOL in with_aux["levels"]["l5_clause"]["widened_symbols_seq"]

    # Negative control: no auxiliary in the sentence, so no symbol 9.
    without_aux = infer_sentence_hierarchy("the dog chased the cat .")
    assert AUX_SYMBOL not in without_aux["levels"]["l5_clause"]["widened_symbols_seq"]


def test_punctuation_contributes_no_widened_symbol():
    """POSITIVE: trailing punctuation adds no symbol. NEGATIVE: a real word does.

    The widened chains are fitted with UD PUNCT tokens skipped. Inference
    stripped the punctuation to an empty string, substituted the literal word
    "word", tagged that NOUN, and fed a case observation into a position the
    training distribution never had one in.
    """
    bare = infer_sentence_hierarchy("the dog runs")
    punctuated = infer_sentence_hierarchy("the dog runs .")
    assert (
        punctuated["levels"]["l5_clause"]["widened_symbols_seq"]
        == bare["levels"]["l5_clause"]["widened_symbols_seq"]
    )

    # Negative control: appending a *word* must change the sequence, or the
    # assertion above would hold for an engine that ignores its input.
    extended = infer_sentence_hierarchy("the dog runs home")
    assert (
        extended["levels"]["l5_clause"]["widened_symbols_seq"]
        != bare["levels"]["l5_clause"]["widened_symbols_seq"]
    )


def test_punctuation_only_input_reports_no_evidence():
    """No case-bearing token means no evidence — the type prior, not a crash."""
    res = infer_sentence_hierarchy("!!! ??? ...")
    l5 = res["levels"]["l5_clause"]
    assert l5["widened_symbols_seq"] == []
    assert l5["has_evidence"] is False
    # Falls back to the type prior rather than manufacturing a posterior.
    assert l5["posterior"] == pytest.approx(l5["prior"], abs=1e-9)


def test_l5_beats_chance_on_held_out_ud_sentences():
    """The dashboard must rank interrogatives above declaratives, with a control.

    Accuracy is the wrong statistic here — the corpus is ~89% declarative, so
    a constant predictor scores well. AUC on the type log-ratio is the
    statistic ``scripts/probe_l5_widened_alphabet.py`` reports, so it is the
    one that compares. The probe clears 0.90 on ORACLE case roles; the
    dashboard runs on case roles *inferred* by Level 4 from a lexicon
    part-of-speech tagger, a strictly harder setting, and scores lower. The
    threshold is set from that measured setting, not from the probe's.

    NEGATIVE CONTROL: the same scores against shuffled labels must sit at
    chance. Without it, a bug that inflated every score identically would
    still show a high AUC.
    """
    from ailm.corpus.ud import load_ud_ewt_dev

    models = get_dashboard_models()
    type_by_final = {".": "declarative", "?": "interrogative"}

    cases: list[tuple[int, str]] = []
    for sentence in load_ud_ewt_dev():
        forms = [t.form for t in sentence.tokens]
        final = next((f for f in reversed(forms) if f in {".", "?", "!"}), None)
        if final in type_by_final:
            cases.append((int(type_by_final[final] == "interrogative"), sentence.text))
        if len(cases) >= 250:
            break

    labels = np.array([c[0] for c in cases])
    assert labels.sum() > 10, "too few interrogatives to measure ranking"

    p_dec, p_int = models["clause_models"]["type_prior"][:2]
    scores = []
    for _, text in cases:
        seq = infer_sentence_hierarchy(text, n_continuation_words=1)["levels"]["l5_clause"][
            "widened_symbols_seq"
        ]
        lp_d = float(np.log(models["D_dec_widened"][seq[0]])) if seq else 0.0
        lp_i = float(np.log(models["D_int_widened"][seq[0]])) if seq else 0.0
        for t in range(len(seq) - 1):
            lp_d += float(np.log(models["B_dec_widened"][seq[t + 1], seq[t]]))
            lp_i += float(np.log(models["B_int_widened"][seq[t + 1], seq[t]]))
        scores.append((lp_i + np.log(p_int)) - (lp_d + np.log(p_dec)))
    scores_arr = np.asarray(scores)

    def _auc(labels_in: np.ndarray) -> float:
        pos, neg = scores_arr[labels_in == 1], scores_arr[labels_in == 0]
        return float(np.mean([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos]))

    auc = _auc(labels)
    assert auc > 0.65, f"L5 ranking collapsed to {auc:.3f}; the widened alphabet is degraded"

    # NEGATIVE CONTROL: shuffled labels must not rank.
    rng = np.random.default_rng(0)
    shuffled_aucs = [_auc(rng.permutation(labels)) for _ in range(15)]
    assert max(shuffled_aucs) < auc, (
        f"a shuffled-label control reached AUC {max(shuffled_aucs):.3f} against the "
        f"model's {auc:.3f} — the statistic is not measuring type"
    )


# ── BEHAVIOR: the Level 6 top-down channel ──────────────────────────────────


def test_narrative_phase_is_an_input_and_the_panel_responds():
    """POSITIVE: the phase changes the reported phase and P(type | phase).

    The regression: the phase was ``tercile(min(len(words) - 1, 0), doc_len)``,
    and ``min(n - 1, 0)`` is 0 for every non-empty input. Level 6 reported
    "opening" for all text of any length, so the level the panel exists to
    demonstrate never moved.
    """
    text = "the dog chased the cat ."
    seen_phases, seen_emissions = [], []
    for phase in range(3):
        l6 = infer_sentence_hierarchy(text, narrative_phase=phase)["levels"]["l6_narrative"]
        assert l6["phase_idx"] == phase
        seen_phases.append(l6["current_phase"])
        seen_emissions.append(tuple(l6["phase_type_emission"]))

    assert len(set(seen_phases)) == 3, f"phase pinned to {set(seen_phases)}"
    assert len(set(seen_emissions)) > 1, "P(type | phase) is constant across phases"

    # Out-of-range input is clamped, not crashed on.
    assert (
        infer_sentence_hierarchy(text, narrative_phase=99)["levels"]["l6_narrative"]["phase_idx"]
        == 2
    )


def test_channel_alpha_zero_is_the_exact_identity_anchor():
    """NEGATIVE CONTROL for the channel: at alpha = 0 the phase cannot matter.

    A channel that "worked" at alpha = 0 would be reading the phase through
    some path other than the blend — the anchor is what rules that out. Paired
    with the positive case below, which requires alpha = 1 to move the prior.
    """
    text = "the dog chased the cat ."
    at_zero = [
        infer_sentence_hierarchy(text, narrative_phase=p, narrative_alpha=0.0)["levels"][
            "l6_narrative"
        ]["blended_prior_sample"]
        for p in range(3)
    ]
    assert at_zero[0] == at_zero[1] == at_zero[2], "alpha = 0 leaked phase information"

    # POSITIVE: at alpha = 1 the phase must move the prior, or the channel is dead.
    at_one = [
        infer_sentence_hierarchy(text, narrative_phase=p, narrative_alpha=1.0)["levels"][
            "l6_narrative"
        ]["blended_prior_sample"]
        for p in range(3)
    ]
    assert at_one[0] != at_one[1], "alpha = 1 changes nothing — the channel is inert"


def test_channel_sweep_reports_the_control_and_its_fixed_point():
    """The sweep must carry the scrambled control, whose middle phase is fixed.

    The scramble is the reversal ``p -> 2 - p``, which fixes the middle phase
    for any corpus. Asserting per-phase difference would be permanently
    unsatisfiable; asserting difference *somewhere* is the honest form.
    """
    sweep = infer_sentence_hierarchy("the dog chased the cat .")["levels"]["l6_narrative"][
        "channel_sweep"
    ]
    at_zero = next(r for r in sweep if r["alpha"] == 0.0)
    assert at_zero["phases"] == at_zero["scrambled"]

    at_one = next(r for r in sweep if r["alpha"] == 1.0)
    assert at_one["phases"]["middle"] == pytest.approx(at_one["scrambled"]["middle"], abs=1e-12)
    assert any(
        abs(at_one["phases"][p] - at_one["scrambled"][p]) > 1e-9 for p in ("opening", "closing")
    ), "the scramble is invisible at every phase — it is not a control"


# ── ROBUSTNESS: bounds on caller-supplied quantities ─────────────────────────


def test_continuation_length_is_clamped():
    """A rollout count off the wire must not become an unbounded loop.

    The rollout costs one lexicon scan per emitted character and runs inside
    the request handler, so an unclamped count is a stall, not just a slow
    response.
    """
    huge = generate_topdown_continuation("the dog", n_words=10_000_000)
    assert len(huge["continuation_text"].split()) <= MAX_CONTINUATION_WORDS

    # Negative counts must degrade to empty, not run backwards or hang.
    assert generate_topdown_continuation("the dog", n_words=-5)["words_generated"] == 0


def test_rollout_returns_the_number_of_words_requested():
    """POSITIVE: N words asked for, N words back — from a partial word or a whole one.

    The rollout resumes inside the caller's in-flight word, and the space that
    closed *that* word was counted against the request. So N words asked for
    returned N - 1, and a request for a single word returned an empty string:
    the entire budget went to finishing a word the caller already had.
    """
    for requested in (1, 2, 5, 12):
        # Mid-word resume: "dog" is still in flight.
        assert (
            generate_topdown_continuation("the dog", n_words=requested)["words_generated"]
            == requested
        )
        # Fresh start: the trailing space closes the caller's word already.
        assert (
            generate_topdown_continuation("the dog ", n_words=requested)["words_generated"]
            == requested
        )

    # NEGATIVE CONTROL: zero must produce nothing, not "one because the loop ran".
    empty = generate_topdown_continuation("the dog", n_words=0)
    assert empty["words_generated"] == 0
    assert empty["continuation_text"] == ""

    res = infer_sentence_hierarchy("the dog", n_continuation_words=10**9)
    assert len(res["topdown_continuation"]["continuation_text"].split()) <= MAX_CONTINUATION_WORDS


def test_upos_provenance_is_reported_not_hidden():
    """A defaulted tag must be distinguishable from an observed one.

    Out-of-lexicon words silently became NOUN. The tag is now reported with
    its source so the interface can mark it, rather than presenting a guess
    with the same authority as a lookup.
    """
    res = infer_sentence_hierarchy("the zzzqqq blorptastic .")
    sources = {w["upos_source"] for w in res["levels"]["l3_word"]["words"]}
    assert "fallback" in sources, "an out-of-lexicon word was not marked as defaulted"
    assert "punctuation" in sources, "punctuation was not marked as skipped"

    # Negative control: a fully in-lexicon sentence reports no fallbacks.
    known = infer_sentence_hierarchy("the dog is here")
    assert all(w["upos_source"] == "lexicon" for w in known["levels"]["l3_word"]["words"]), (
        "an in-lexicon sentence reported a defaulted tag"
    )


def test_static_html_has_no_dead_control_references():
    """Every element the script writes to must exist, and vice versa.

    Three controls shipped permanently inert because the script addressed
    payload keys the engine never emitted (``rollout``) or never addressed
    elements the markup declared (``l5-widened-obs``, ``l6-phase-bias``).
    Nothing failed; the panels just read "N/A" and "-" forever.
    """
    html = _dashboard_html()

    # Generalized: EVERY id the script looks up must be declared in the markup.
    # The earlier version listed the ids it knew about, which is why it missed
    # `rollout-len-input` — a control the script READ but never wrote, so it was
    # not on the list. `getElementById` returned null, `.value` threw a
    # TypeError while building the request body, and runInference() failed on
    # every call before reaching the network. Nothing errored visibly; the
    # panels just stayed on their placeholder text.
    code_only = "\n".join(line.split("//", 1)[0] for line in html.splitlines())
    looked_up = set(re.findall(r"getElementById\(\s*['\"]([A-Za-z0-9_-]+)['\"]", code_only))
    declared = set(re.findall(r"id=\"([A-Za-z0-9_-]+)\"", html))
    undeclared = sorted(looked_up - declared)
    assert not undeclared, (
        f"the script looks up ids the markup never declares: {undeclared}. "
        f"getElementById returns null and the next property access throws."
    )
    assert len(looked_up) > 10, "the id scan found almost nothing; the regex has drifted"

    # No script line may read a payload key the engine does not emit. Checked
    # against code with `//` comments stripped, so the prose recording *why*
    # a key was removed does not itself trip the guard. The rule is general —
    # any read of an absent top-level key, not the one that was fixed — so a
    # different key going stale later is caught the same way.
    code = "\n".join(line.split("//", 1)[0] for line in html.splitlines())
    # The page talks to two endpoints with different payload shapes, so the
    # union of both is what a script line may legally read. Checking against
    # /api/infer alone would flag every legitimate chat key.
    from ailm.dashboard.engine import converse

    emitted = set(infer_sentence_hierarchy("the dog runs .")) | set(
        converse("the dog runs .", n_words=2)
    )
    for match in re.finditer(r"(?:currentInference|data)\.([A-Za-z_][A-Za-z0-9_]*)", code):
        key = match.group(1)
        if key in {"error", "keys"}:
            continue
        assert key in emitted, f"the script reads .{key}, which the engine payload has no key for"


# ── ENTRY POINT ──────────────────────────────────────────────────────────────


def _unused_loopback_port() -> int:
    """Ask the kernel for a currently unused loopback port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _clean_subprocess_env() -> dict[str, str]:
    """An override-free environment for one package subprocess."""
    env = dict(os.environ)
    for name in tuple(env):
        if name.startswith("AILM_DASHBOARD_"):
            env.pop(name)
    source = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _fresh_subprocess(code: str) -> subprocess.CompletedProcess[str]:
    """Run an assertion script in a clean interpreter over the real tree.

    Import-boundary behavior cannot be verified in-process: the pytest run
    has already imported the numerical engine, so a lazy-import regression
    would be invisible to it. Every probe here must observe a fresh start.
    """
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=_clean_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=480,
        check=False,
    )


def _dashboard_environment(tmp_path: Path, *, port: int) -> dict[str, str]:
    """A deterministic environment for one real dashboard subprocess."""
    env = _clean_subprocess_env()
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    env["AILM_DASHBOARD_HOST"] = "127.0.0.1"
    env["AILM_DASHBOARD_PORT"] = str(port)
    env["AILM_DASHBOARD_MAX_SPOKEN_CHARS"] = "321"
    return env


def _launch_dashboard(env: dict[str, str], *arguments: str) -> subprocess.Popen[str]:
    """Start the real package CLI; callers must close it with `_interrupt_dashboard`."""
    return subprocess.Popen(
        [sys.executable, "-m", "ailm.dashboard", *arguments],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_dashboard(process: subprocess.Popen[str], port: int) -> dict:
    """Wait until the real HTTP endpoint answers or expose an early process exit."""
    deadline = time.monotonic() + 30
    root_url = f"http://127.0.0.1:{port}/"
    config_url = f"http://127.0.0.1:{port}/api/config"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"dashboard exited {process.returncode} before binding\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )
        try:
            with urlopen(root_url, timeout=0.25) as landing:
                cookie = landing.headers.get("Set-Cookie", "").split(";", 1)[0]
            request = Request(config_url, headers={"Cookie": cookie})
            with urlopen(request, timeout=0.25) as response:
                assert response.status == 200
                return json.loads(response.read())
        except (OSError, URLError):
            time.sleep(0.05)
    raise AssertionError(f"dashboard did not answer {config_url} within 30 seconds")


def _interrupt_dashboard(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Deliver a real Ctrl-C and require the process to settle promptly."""
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    try:
        return process.communicate(timeout=120)
    except subprocess.TimeoutExpired as error:
        process.terminate()
        process.communicate(timeout=5)
        raise AssertionError("dashboard ignored SIGINT for 120 seconds") from error


def test_dashboard_entry_point_validates_its_arguments():
    """POSITIVE: bad ports are rejected with a status, not a traceback.

    NEGATIVE CONTROL: a valid port must NOT be rejected, or the guard would pass
    by refusing everything.
    """
    from ailm.dashboard.__main__ import main, resolve_bind
    from ailm.dashboard.config import DashboardConfig

    assert main(["notaport"]) == 2
    assert main(["0"]) == 2
    assert main(["99999"]) == 2
    assert main(["-1"]) == 2

    # A valid port reaches the exact bind resolver that `main` uses. Exercising
    # it separately avoids opening a long-lived server merely to inspect a pure
    # precedence decision; the real process tests below prove the resolved bind
    # reaches a socket.
    shipped = DashboardConfig()
    assert resolve_bind(["9001"], shipped) == (shipped.host, 9001)
    assert resolve_bind([], shipped) == (shipped.host, shipped.port)
    with pytest.raises(ValueError, match="not loopback"):
        resolve_bind(["9002", "0.0.0.0"], shipped)
    assert resolve_bind(["9002", "127.0.0.3"], shipped) == ("127.0.0.3", 9002)


@pytest.mark.timeout(240)
def test_the_environment_moves_the_port_the_server_actually_binds(tmp_path):
    """POSITIVE: `AILM_DASHBOARD_PORT` reaches the socket, not just the parser.

    The README documents the variable as the way to move the dashboard. The
    entry point used to read it — through `load_config`, validated and all — and
    then pass its own hardcoded 8080 to the server anyway. Nothing raised: the
    server came up on the default port while the operator believed they had
    moved it, and the only symptom was a bind failure if something already held
    8080.

    NEGATIVE CONTROL: an explicit argument must still win, or this would pass by
    letting the environment override everything.
    """
    from ailm.dashboard.__main__ import resolve_bind
    from ailm.dashboard.config import config_from_env

    port = _unused_loopback_port()
    env = _dashboard_environment(tmp_path, port=port)
    process = _launch_dashboard(env)
    try:
        payload = _wait_for_dashboard(process, port)
    finally:
        stdout, stderr = _interrupt_dashboard(process)

    assert process.returncode == 0, stderr
    assert payload["max_spoken_chars"] == 321
    assert f"http://127.0.0.1:{port}/" in stdout
    assert "Shutting down dashboard server" in stdout

    config = config_from_env(environ=env)
    assert resolve_bind([], config) == ("127.0.0.1", port)
    # An argument is more specific than the environment and must win.
    assert resolve_bind(["9100"], config) == ("127.0.0.1", 9100)
    assert resolve_bind(["9100", "127.0.0.3"], config) == ("127.0.0.3", 9100)


@pytest.mark.timeout(240)
def test_an_unparseable_override_stops_the_launch(tmp_path):
    """A refused setting exits with a status; it does not start on the default.

    Starting anyway would be the quiet fallback `config.py` refuses on purpose:
    the operator asked for a specific limit and would get a working server that
    ignores it.
    """
    # No socket should be opened on this path, so a fixed legal port is enough;
    # requiring a free-port probe would add a bind before the refusal under test.
    env = _dashboard_environment(tmp_path, port=8123)
    env["AILM_DASHBOARD_MAX_SPOKEN_CHARS"] = "not-a-number"
    result = subprocess.run(
        [sys.executable, "-m", "ailm.dashboard"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 2
    assert "not-a-number" in result.stderr
    assert "live at" not in result.stdout


def test_dashboard_is_launched_as_a_package_not_a_submodule():
    """The package has exactly one launch surface, `python -m ailm.dashboard`.

    `__main__.py` gives the launch a module of its own. The lazy `__init__`
    also removed the old reason for this guard — `__init__` used to import
    `server`, so invoking the submodule as `__main__` initialized it twice and
    warned on every launch — but a second launch path must still not exist:
    the server module refuses direct execution (next test), and the README
    keeps naming the package form.
    """
    entry = Path(__file__).resolve().parents[1] / "src" / "ailm" / "dashboard" / "__main__.py"
    assert entry.exists(), "the package entry point was removed"
    assert "run_dashboard" in entry.read_text(encoding="utf-8")

    readme = entry.parent / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert "-m ailm.dashboard\n" in text or "-m ailm.dashboard " in text


@pytest.mark.timeout(240)
def test_the_server_module_refuses_to_be_run_directly():
    """`python -m ailm.dashboard.server` must not serve anything.

    It used to be runnable with a hardcoded 8080 and the environment
    discarded — the exact bug the package entry point exists to prevent. With
    the stale block merely deleted it would instead exit 0 having served
    nothing, which is its own trap: the operator believes the dashboard is
    up. It refuses with guidance instead, naming the real launch path.
    """
    result = subprocess.run(
        [sys.executable, "-m", "ailm.dashboard.server"],
        cwd=Path(__file__).resolve().parents[1],
        env=_clean_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0
    assert "-m ailm.dashboard" in result.stderr, "the refusal must name the real entry point"
    assert "live at" not in result.stdout


# ── LAZY STARTUP BOUNDARY ────────────────────────────────────────────────────


@pytest.mark.timeout(960)
def test_importing_the_entry_point_loads_no_numerical_engine():
    """`import ailm.dashboard.__main__` must not pull numpy, jax or pymdp.

    The old package eagerly imported the engine, so merely launching — before
    any configuration was validated or socket bound — paid a full numerical
    import. The boundary now lives in the service adapters. The probe runs in
    a fresh interpreter because the pytest process has already imported the
    engine and could never see a regression.

    NEGATIVE CONTROL: the same probe must see the modules when the engine is
    imported for real, so the assertion cannot pass by blindness.
    """
    result = _fresh_subprocess(
        textwrap.dedent(
            """
            import sys

            import ailm.dashboard.__main__ as entry  # noqa: F401

            heavy = [m for m in ("numpy", "jax", "pymdp") if m in sys.modules]
            assert not heavy, f"numerical engine loaded at import time: {heavy}"

            import ailm.dashboard.engine  # noqa: F401  (the deliberate control)

            assert "numpy" in sys.modules, "probe is blind; a regression would be invisible"
            print("entry-point boundary holds")
            """
        )
    )
    assert result.returncode == 0, result.stderr
    assert "entry-point boundary holds" in result.stdout


@pytest.mark.timeout(960)
def test_public_api_resolves_the_original_callables():
    """The lazy facade must return the real function objects, nothing else.

    A re-export shim can quietly substitute a wrapper; embedders comparing
    `ailm.dashboard.run_dashboard` to `server.run_dashboard` would then see
    two functions. Resolution must also fail like a missing attribute, list
    itself in `dir()`, and never list a refused name.

    NEGATIVE CONTROL: an unknown attribute raises and is absent from `dir()`.
    """
    result = _fresh_subprocess(
        textwrap.dedent(
            """
            import ailm.dashboard
            import ailm.dashboard.engine
            import ailm.dashboard.server

            assert (
                ailm.dashboard.infer_sentence_hierarchy
                is ailm.dashboard.engine.infer_sentence_hierarchy
            )
            assert ailm.dashboard.run_dashboard is ailm.dashboard.server.run_dashboard

            try:
                ailm.dashboard.no_such_export
            except AttributeError as error:
                assert "no_such_export" in str(error)
            else:
                raise SystemExit("an unknown attribute resolved instead of failing")

            listing = dir(ailm.dashboard)
            assert {"infer_sentence_hierarchy", "run_dashboard"} <= set(listing)
            assert "no_such_export" not in listing
            print("public api resolves originals")
            """
        )
    )
    assert result.returncode == 0, result.stderr
    assert "public api resolves originals" in result.stdout


@pytest.mark.timeout(960)
def test_static_config_and_auth_serve_before_any_numerical_import():
    """A fresh process answers page, credential and config without the engine.

    This is the contract the service adapters exist for: launch cost is
    stdlib-only, and only a computation request pays for the numerical
    engine.

    NEGATIVE CONTROL: /api/config without the credential must be refused, so
    a test cannot pass against a server that serves everything.
    """
    result = _fresh_subprocess(
        textwrap.dedent(
            """
            import json
            import sys
            import threading
            import urllib.error
            import urllib.request
            from functools import partial

            from ailm.dashboard import server as dashboard_server

            heavy = [m for m in ("numpy", "jax", "pymdp") if m in sys.modules]
            assert not heavy, f"engine loaded before serving: {heavy}"

            httpd = dashboard_server.ReusableThreadingServer(
                ("127.0.0.1", 0),
                partial(dashboard_server.DashboardHandler, config=dashboard_server.CONFIG),
            )
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = httpd.server_address[:2]
                base = f"http://{host}:{port}"
                with urllib.request.urlopen(base + "/", timeout=5) as landing:
                    assert landing.status == 200
                    cookie = landing.headers["Set-Cookie"].split(";", 1)[0]
                request = urllib.request.Request(
                    base + "/api/config", headers={"Cookie": cookie}
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    assert response.status == 200
                    assert json.loads(response.read())["max_request_bytes"] > 0
                try:
                    urllib.request.urlopen(base + "/api/config", timeout=5)
                except urllib.error.HTTPError as error:
                    assert error.code == 401
                else:
                    raise SystemExit("config served without the credential")
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=5)

            still_absent = [m for m in ("numpy", "jax", "pymdp") if m in sys.modules]
            assert not still_absent, f"serving pulled in the engine: {still_absent}"
            print("served with no numerical imports")
            """
        )
    )
    assert result.returncode == 0, result.stderr
    assert "served with no numerical imports" in result.stdout


def test_default_service_adapters_forward_to_the_engine():
    """`DEFAULT_SERVICES` forwards wire requests to the production engine.

    The adapters exist so that importing `server` does not import the engine;
    an adapter that dropped or renamed an argument would change request
    semantics with no structural test noticing, so identical calls through
    the boundary and into the engine must return identical results. The
    engine is deterministic over fixed arguments (the chat rollout seed is
    derived from the text), so equality here is a real check, not a tautology.
    """
    from ailm.dashboard.engine import (
        converse,
        get_dashboard_models,
        infer_sentence_hierarchy,
    )
    from ailm.dashboard.server import (
        DEFAULT_SERVICES,
        _converse,
        _get_dashboard_models,
        _infer_sentence_hierarchy,
    )

    # The defaults must be the lazy adapters themselves, not the engine.
    assert DEFAULT_SERVICES.infer is _infer_sentence_hierarchy
    assert DEFAULT_SERVICES.chat is _converse
    assert DEFAULT_SERVICES.models is _get_dashboard_models

    infer_kwargs = dict(
        n_continuation_words=3,
        narrative_phase=2,
        narrative_alpha=0.5,
        max_continuation_words=8,
    )
    assert DEFAULT_SERVICES.infer("did she go home ?", **infer_kwargs) == (
        infer_sentence_hierarchy("did she go home ?", **infer_kwargs)
    )

    chat_kwargs = dict(
        history=[{"role": "user", "text": "hi"}],
        n_words=3,
        narrative_phase=None,
        narrative_alpha=0.5,
        max_history_turns=5,
        max_continuation_words=8,
    )
    assert DEFAULT_SERVICES.chat("hello there", **chat_kwargs) == converse(
        "hello there", **chat_kwargs
    )

    # `get_dashboard_models` is lru_cached, so the boundary must hand back
    # the very object the engine caches, not a copy.
    assert DEFAULT_SERVICES.models() is get_dashboard_models()


# ── BIDIRECTIONAL CHAT ───────────────────────────────────────────────────────


def test_conversation_advances_the_narrative_phase():
    """POSITIVE: the phase moves with the turn index, unset by the caller.

    A narrative phase is a document-position tercile. A single sentence has no
    position, which is why the inspector takes the phase as an input — but a
    turn in a transcript does have one, and this is the setting where Level 6
    finally has something real to condition on.
    """
    from ailm.dashboard.engine import converse

    history: list[dict] = []
    phases = []
    for message in ["hello there", "is he going ?", "the dog ran .", "what now ?"]:
        turn = converse(message, history=history, n_words=4)
        phases.append(turn["read_as"]["narrative_phase"])
        assert turn["phase_source"] == "derived_from_turn"
        history += [
            {"role": "user", "text": message},
            {"role": "model", "text": turn["reply"]},
        ]

    assert phases[0] == "opening"
    assert len(set(phases)) > 1, f"the phase never moved across turns: {phases}"

    # NEGATIVE CONTROL: an explicit phase must override the derivation, or the
    # inspector's own phase control would silently stop working in chat.
    forced = converse("hello there", history=history, n_words=4, narrative_phase=2)
    assert forced["read_as"]["narrative_phase"] == "closing"
    assert forced["phase_source"] == "explicit"


def test_replies_vary_across_turns_but_reproduce_exactly():
    """POSITIVE: successive turns differ. NEGATIVE CONTROL: a repeat is identical.

    A fixed rollout seed made every turn emit the same words, which reads as a
    broken interface. The seed is derived from the turn index and the message,
    with a stable digest rather than `hash()` — Python salts string hashing per
    process, so `hash()` would give a different transcript after every server
    restart and quietly break the reproducibility this project claims.
    """
    from ailm.dashboard.engine import converse

    history: list[dict] = []
    replies = []
    for message in ["hello there", "hello there", "hello there"]:
        turn = converse(message, history=history, n_words=6)
        replies.append(turn["reply"])
        history += [
            {"role": "user", "text": message},
            {"role": "model", "text": turn["reply"]},
        ]
    assert len(set(replies)) > 1, f"the same message repeated verbatim every turn: {replies}"

    # Same message at the same point in the conversation reproduces exactly.
    again = converse("hello there", history=[], n_words=6)
    assert again["reply"] == replies[0]
    assert again["rollout_seed"] == converse("hello there", history=[], n_words=6)["rollout_seed"]


def test_chat_reports_what_it_inferred_about_the_user_turn():
    """The transcript's per-turn line must carry measured facts, not decoration."""
    from ailm.dashboard.engine import converse

    turn = converse("is he going to the store ?", n_words=4)
    read = turn["read_as"]
    assert read["clause_type"] in ("declarative", "interrogative")
    assert 0.0 <= read["clause_confidence"] <= 1.0
    assert read["tokens"] > 0
    assert 0 <= read["in_lexicon"] <= read["tokens"]
    assert "AUX" in read["surface"], "the auxiliary is not reaching the chat summary"
    assert turn["hierarchy"]["levels"]["l5_clause"]["predicted_type"] == read["clause_type"]


def test_chat_survives_hostile_and_empty_history():
    """A transcript arrives off the wire; nothing in it may be trusted."""
    from ailm.dashboard.engine import converse

    assert converse("hello", history=None, n_words=3)["turn_index"] == 0
    assert converse("hello", history=[], n_words=3)["turn_index"] == 0

    # An over-long transcript is truncated rather than replayed in full.
    flood = [{"role": "user", "text": f"turn {i}"} for i in range(500)]
    bounded = converse("hello", history=flood, n_words=3)
    assert bounded["turn_index"] <= MAX_HISTORY_TURNS

    # Empty message still produces a well-formed turn rather than raising.
    blank = converse("", n_words=3)
    assert isinstance(blank["reply"], str)
    assert blank["read_as"]["tokens"] > 0


def test_chat_history_normalizer_drops_malformed_turns():
    """POSITIVE: valid turns survive. NEGATIVE CONTROL: junk does not."""
    from ailm.dashboard.server import _as_history

    assert _as_history(None, max_turns=2) == []
    assert _as_history("not a list", max_turns=2) == []
    assert _as_history([{"role": "user", "text": "hi"}], max_turns=2) == [
        {"role": "user", "text": "hi"}
    ]
    assert _as_history([{"role": "system", "text": "x"}], max_turns=2) == []
    assert _as_history([{"text": "no role"}], max_turns=2) == []
    assert _as_history(["bare string", 42, None], max_turns=2) == []
    assert _as_history([{"role": "model", "text": 7}], max_turns=2) == [
        {"role": "model", "text": "7"}
    ]


def test_chat_ui_is_wired_and_states_what_the_reply_is():
    """Every chat control the script addresses must exist, and the honesty note stays.

    The disclaimer is load-bearing: this model has a few thousand free
    parameters and no semantics, and an interface that presents its rollout as
    an answer would be the single most misleading thing in the repository.
    """
    html = _dashboard_html()

    for element_id in (
        "chat-log",
        "chat-input",
        "chat-send-btn",
        "chat-mic-btn",
        "chat-turn-count",
        "chat-export-btn",
    ):
        assert f'id="{element_id}"' in html, f"{element_id} is addressed but not declared"

    for function_name in ("sendChat", "renderChat", "resetChat", "toggleChatSTT", "exportChat"):
        assert f"function {function_name}" in html, f"{function_name} is called but not defined"

    assert "/api/chat" in html
    assert "no semantics" in html, "the honesty note was removed from the chat panel"
    assert "word salad" in html, "the expectation-setting line was removed"


def _transcript_export_defects(source: str) -> list[str]:
    """Static contract for the transcript export: a local JSON download, nothing more.

    The function must build the file from the page's own transcript store, type
    it as JSON, hand it to the browser as an object URL that it then revokes,
    and never talk to the server: an export that fetched would be a request the
    README's endpoint table does not list.
    """
    match = re.search(r"function exportChat\(\)\s*\{(?P<body>.*?)\n\}", source, re.S)
    if match is None:
        return ["exportChat is not defined"]
    body = match.group("body")
    defects = []
    if "store.chatHistory" not in body:
        defects.append("export does not read the transcript store")
    if "new Blob(" not in body or "application/json" not in body:
        defects.append("export is not a JSON blob")
    if "URL.createObjectURL(" not in body or "URL.revokeObjectURL(" not in body:
        defects.append("object URL is not created and revoked")
    if ".download =" not in body:
        defects.append("export does not trigger a download")
    if "fetch(" in body:
        defects.append("export talks to the server")
    return defects


def test_chat_export_is_a_local_json_download_of_the_transcript():
    source = (STATIC / "js" / "chat.js").read_text(encoding="utf-8")
    assert _transcript_export_defects(source) == []
    # The reply line counts turns from one, as the header does, and discloses
    # defaulted tags instead of hiding them.
    assert "turn ${data.turn_index + 1}" in source
    assert "read.defaulted_tags" in source
    # NEGATIVE CONTROLS: a leaking or server-bound export must be refused.
    leaking = source.replace("    URL.revokeObjectURL(url);\n", "")
    assert "object URL is not created and revoked" in _transcript_export_defects(leaking)
    remote = source.replace("const blob = new Blob(", "fetch('/api/chat'); const blob = new Blob(")
    assert "export talks to the server" in _transcript_export_defects(remote)
    assert _transcript_export_defects("function other() {}") == ["exportChat is not defined"]


def _opening_tag_attributes(markup: str, element_id: str) -> dict[str, str] | None:
    match = re.search(rf'<[A-Za-z][^>]*\bid="{re.escape(element_id)}"[^>]*>', markup)
    if match is None:
        return None
    return dict(re.findall(r'([A-Za-z_:][-A-Za-z0-9_:.]*)="([^"]*)"', match.group(0)))


def _control_name_defects(markup: str, element_ids: tuple[str, ...]) -> list[str]:
    """Return requested controls with no label-derived accessible name."""
    defects: list[str] = []
    declared_ids = set(re.findall(r'\bid="([^"]+)"', markup))
    for element_id in element_ids:
        attrs = _opening_tag_attributes(markup, element_id)
        if attrs is None:
            defects.append(f"{element_id}: missing")
            continue
        has_label = bool(attrs.get("aria-label", "").strip())
        referenced = attrs.get("aria-labelledby", "").split()
        # "aria-labelledby" is the fixed W3C ARIA attribute name; only the local
        # variable below is renamed to satisfy the American-spelling gate.
        has_label = has_label or bool(referenced and all(ref in declared_ids for ref in referenced))
        has_label = has_label or bool(
            re.search(rf'<label\b[^>]*\bfor="{re.escape(element_id)}"[^>]*>', markup)
        )
        if not has_label:
            defects.append(f"{element_id}: no accessible name")
    return defects


def _canvas_accessibility_defects(markup: str) -> list[str]:
    defects: list[str] = []
    canvases = list(re.finditer(r"<canvas\b(?P<attrs>[^>]*)>(?P<body>.*?)</canvas>", markup, re.S))
    if not canvases:
        return ["no canvas elements found"]
    for canvas in canvases:
        attrs = dict(re.findall(r'([A-Za-z_:][-A-Za-z0-9_:.]*)="([^"]*)"', canvas.group("attrs")))
        element_id = attrs.get("id", "unnamed-canvas")
        if attrs.get("role") != "img":
            defects.append(f"{element_id}: no image role")
        if not attrs.get("aria-label", "").strip() and not attrs.get("aria-labelledby", "").strip():
            defects.append(f"{element_id}: no accessible name")
        if not " ".join(canvas.group("body").split()):
            defects.append(f"{element_id}: no fallback text")
    return defects


def _live_region_defects(markup: str) -> list[str]:
    expected_roles = {
        "status-line": "status",
        "chat-log": "log",
        "continuation-banner": "status",
    }
    defects: list[str] = []
    for element_id, role in expected_roles.items():
        attrs = _opening_tag_attributes(markup, element_id)
        if attrs is None or attrs.get("role") != role or attrs.get("aria-live") != "polite":
            defects.append(f"{element_id}: not a polite {role} region")
    return defects


def test_primary_inputs_and_dynamic_results_are_accessible_with_planted_controls():
    markup = (STATIC / "index.html").read_text(encoding="utf-8")
    controls = ("sentence-input", "chat-input", "time-slider")
    assert _control_name_defects(markup, controls) == []
    assert _canvas_accessibility_defects(markup) == []
    assert _live_region_defects(markup) == []

    unnamed = (
        markup.replace('aria-label="Sentence to analyze"', "", 1)
        .replace('aria-label="Chat message"', "", 1)
        .replace('for="time-slider"', 'for="wrong-slider"', 1)
        .replace('aria-labelledby="time-slider-label slider-step-label"', "", 1)
    )
    assert _control_name_defects(unnamed, controls) == [
        "sentence-input: no accessible name",
        "chat-input: no accessible name",
        "time-slider: no accessible name",
    ]

    assert _canvas_accessibility_defects('<canvas id="planted"></canvas>') == [
        "planted: no image role",
        "planted: no accessible name",
        "planted: no fallback text",
    ]

    silent = markup.replace(
        'id="chat-log" role="log" aria-live="polite"', 'id="chat-log" role="log"', 1
    )
    assert "chat-log: not a polite log region" in _live_region_defects(silent)


def _narrow_layout_defects(stylesheet: str) -> list[str]:
    required = {
        "missing narrow breakpoint": "@media (max-width: 720px)",
        "level cards retain a fixed minimum width": (
            ".grid-levels, .matrix-grid-view, .stat-grid { grid-template-columns: minmax(0, 1fr); }"
        ),
        "header actions cannot wrap": ".header-actions { flex-wrap: wrap; width: 100%; }",
        "sentence controls cannot wrap": ".input-row { flex-wrap: wrap; }",
    }
    return [message for message, fragment in required.items() if fragment not in stylesheet]


def test_narrow_layout_has_no_fixed_width_trap_with_a_planted_control():
    stylesheet = (STATIC / "dashboard.css").read_text(encoding="utf-8")
    assert _narrow_layout_defects(stylesheet) == []

    planted = stylesheet.replace(
        ".grid-levels, .matrix-grid-view, .stat-grid { grid-template-columns: minmax(0, 1fr); }",
        ".grid-levels { grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); }",
        1,
    )
    assert _narrow_layout_defects(planted) == ["level cards retain a fixed minimum width"]


def _causal_honesty_defects(text: str) -> list[str]:
    normalized = " ".join(re.sub(r"<[^>]+>", " ", text).split())
    defects: list[str] = []
    if not re.search(r"only the L3→L1g policy rollout produces (?:a|the) reply", normalized, re.I):
        defects.append("the sole reply path is not named")
    if "L2, L4, L5, and L6" not in normalized or "do not feed" not in normalized:
        defects.append("analysis-only levels are not excluded from the reply path")
    if "belief state that produced" in normalized:
        defects.append("analysis panels are falsely presented as reply causes")
    return defects


def test_dashboard_states_the_reply_path_and_rejects_the_old_cascade_claim():
    markup = (STATIC / "index.html").read_text(encoding="utf-8")
    assert _causal_honesty_defects(markup) == []

    planted = (
        "The belief state that produced the model's reply is shown at every level. "
        "The model runs a six-level cascade."
    )
    assert _causal_honesty_defects(planted) == [
        "the sole reply path is not named",
        "analysis-only levels are not excluded from the reply path",
        "analysis panels are falsely presented as reply causes",
    ]

    for path in (STATIC.parent / "README.md", STATIC.parent / "engine.py"):
        text = path.read_text(encoding="utf-8")
        assert "L3→L1g" in text, f"{path.name} does not name the reply path"
        assert "belief state that produced" not in text, f"{path.name} retains the false cascade"


def test_no_raw_latex_reaches_the_page():
    """The dashboard ships no math renderer, so TeX source renders as TeX source.

    The page is deliberately self-contained — no CDN, no bundler, no external
    assets — so loading KaTeX or MathJax is not available to it, and every
    ``$...$`` in the markup was shown to the user literally: the Level 5 panel
    read ``P(\\text{type} \\mid \\text{cases}_{1:t})``. Unicode renders
    everywhere and costs nothing.

    NEGATIVE CONTROL: the scan must still find JavaScript template literals, or
    a regex that matched nothing would pass this trivially.
    """
    html = _dashboard_html()

    # `${...}` is a JS template literal, not TeX; everything else between
    # dollar signs on one line is math the page cannot render.
    latex = [m for m in re.findall(r"\$[^${<>\n]{1,60}\$", html) if not m.startswith("${")]
    assert not latex, f"raw TeX reaches the rendered page: {sorted(set(latex))}"

    assert "${" in html, "the scan found no template literals; the pattern has drifted"


# ── SPEECH LIFECYCLE ─────────────────────────────────────────────────────────


def _dashboard_html() -> str:
    """Markup plus every asset it loads, as one string.

    The page is modular now: markup in `index.html`, styles in `dashboard.css`,
    behavior across `js/*.js`. These assertions are about the page as the
    browser assembles it, so they read what the browser reads. Scanning
    `index.html` alone would silently stop checking the JavaScript the moment it
    moved out of the document, which is the failure this helper exists to avoid.
    """
    static = Path(__file__).resolve().parents[1] / "src" / "ailm" / "dashboard" / "static"
    sources = [(static / "index.html").read_text(encoding="utf-8")]
    sources += [p.read_text(encoding="utf-8") for p in sorted(static.glob("*.css"))]
    sources += [p.read_text(encoding="utf-8") for p in sorted((static / "js").glob("*.js"))]
    return "\n".join(sources)


def test_every_function_the_page_calls_is_defined():
    """No handler may call a function that does not exist.

    `speakDialogicResponse()` was wired to the inspector's voice input and had
    never been defined anywhere. Every spoken input raised a ReferenceError
    inside an unhandled promise rejection, so the failure was invisible without
    the console open. The earlier dead-control guard could not see it: that one
    checks `getElementById` ids, and this was a bare call.

    Stated over the general class — every identifier invoked from an `onclick`,
    an `onchange`, or a `.then(...)` body — so the next one is caught too.
    """
    html = _dashboard_html()
    code = "\n".join(line.split("//", 1)[0] for line in html.splitlines())

    defined = set(re.findall(r"function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code))
    defined |= set(
        re.findall(r"(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?\(", code)
    )
    assert len(defined) > 10, "the function scan found almost nothing; the pattern has drifted"

    # Inline handlers are the ones a user can actually trigger.
    handlers = set(re.findall(r'on(?:click|change|input)="([A-Za-z_][A-Za-z0-9_]*)\(', html))
    assert handlers, "no inline handlers found; the scan is not looking at the page"

    browser_builtins = {"runInference", "sendChat"}
    undefined = sorted(h for h in handlers if h not in defined and h not in browser_builtins)
    assert not undefined, f"inline handlers call undefined functions: {undefined}"

    # And the specific ghost, by name, so it cannot come back.
    assert "speakDialogicResponse(" not in code, "the undefined speech handler is back"


def test_speech_has_exactly_one_entry_point_and_it_stops_first():
    """All speech goes through `speak`, and `speak` cancels before it starts.

    `speechSynthesis` is a queue, not a channel: `speak()` appends and `cancel()`
    only clears what is pending at that instant. Two turns in quick succession,
    or one turn speaking from two code paths, leave the queue draining after the
    turn is over. One entry point that stops first is what makes a turn
    terminate.
    """
    code = "\n".join(line.split("//", 1)[0] for line in _dashboard_html().splitlines())

    # Only the controller may touch the synthesis API directly.
    controller = code[
        code.index("function stopSpeaking") : code.index("function updateSpeakingIndicator")
    ]
    speak_fn = code[code.index("function speak(") : code.index("function speakCurrentSentence")]
    elsewhere = code.replace(controller, "").replace(speak_fn, "")
    assert "speechSynthesis.speak(" not in elsewhere, (
        "something outside the controller queues an utterance directly"
    )

    # speak() must stop before it starts.
    assert speak_fn.index("stopSpeaking()") < speak_fn.index("speechSynthesis.speak("), (
        "speak() queues an utterance before cancelling the previous one"
    )


def test_speech_is_stopped_at_every_boundary_that_ends_a_turn():
    """A turn, a microphone, a hidden tab and an unload each terminate speech.

    Each is a way the page can stop being the thing the user is attending to
    while an utterance is still in flight.
    """
    code = "\n".join(line.split("//", 1)[0] for line in _dashboard_html().splitlines())

    send_chat = code[code.index("async function sendChat") : code.index("chatRecognition.onresult")]
    assert "stopSpeaking()" in send_chat, "a new chat turn does not silence the previous one"

    for start in ("recognition.start();", "chatRecognition.start();"):
        prefix = code[: code.index(start)]
        assert prefix.rstrip().endswith("stopSpeaking();"), (
            f"{start} is not preceded by stopSpeaking; the page would hear itself"
        )

    for event in ("visibilitychange", "pagehide", "beforeunload"):
        assert event in code, f"no {event} handler; a backgrounded tab keeps talking"

    # A user-reachable stop, and an indicator that something is speaking.
    html = _dashboard_html()
    assert 'id="stop-speaking-btn"' in html
    assert 'id="speaking-indicator"' in html


def test_utterance_length_and_watchdog_are_bounded():
    """A long rollout is the utterance most likely to outlive its turn."""
    from ailm.dashboard.config import DashboardConfig

    code = _dashboard_html()
    # The bounds moved to the server, so the page must READ them rather than
    # declare them. Asserting the literals here would re-create the duplication
    # the config module exists to remove.
    assert "cfg.max_spoken_chars" in code
    assert "cfg.speech_watchdog_ms" in code

    config = DashboardConfig()
    assert 50 <= config.max_spoken_chars <= 2000, "implausible spoken-length cap"
    assert 5000 <= config.speech_watchdog_ms <= 120000, "implausible watchdog"


def test_self_cancellation_is_not_reported_as_an_error():
    """`interrupted` and `canceled` are what our own stop produces.

    Reporting them would make every normal turn end with an error banner, which
    trains the reader to ignore the banner.
    """
    code = _dashboard_html()
    assert "'interrupted'" in code and "'canceled'" in code, (
        "self-cancellation is not distinguished from a real synthesis error"
    )


# ── STRUCTURE: the page is modular, not one file ─────────────────────────────


STATIC = Path(__file__).resolve().parents[1] / "src" / "ailm" / "dashboard" / "static"


def test_the_page_is_markup_only():
    """No inline `<style>` or `<script>` body. `index.html` was 1,306 lines.

    A single file holding 339 lines of CSS, 671 of JS and 296 of markup cannot
    be reviewed a concern at a time, and a change to any one of them shows up as
    a diff against all three.
    """
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "<style>" not in html, "CSS is inline again"
    assert re.search(r"<script(?![^>]*\bsrc=)[^>]*>\s*\S", html) is None, "JS is inline again"
    assert '<link rel="stylesheet" href="dashboard.css">' in html
    assert '<script type="module" src="js/main.js"></script>' in html
    assert len(html.splitlines()) < 500, "index.html is growing back into a monolith"


def test_the_markup_carries_no_inline_style_attributes():
    """A `style=` attribute is CSS in the markup that does not announce itself.

    The `<style>` check above does not catch it, and 48 of them had accumulated:
    the same three button backgrounds transcribed across eight buttons, the same
    twelve-pixel muted label repeated at every chart. They cannot be reused,
    cannot be overridden without `!important`, and do not read as a stylesheet
    change in review. `dashboard.css` owns appearance.

    NEGATIVE CONTROL: the stylesheet must actually define the classes the markup
    now names, or this would pass on a page with no styling at all.
    """
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    inline = re.findall(r'\sstyle="[^"]*"', html)
    assert not inline, f"{len(inline)} inline style attributes: {inline[:3]}"

    css = (STATIC / "dashboard.css").read_text(encoding="utf-8")
    used = set(re.findall(r'class="([^"]+)"', html))
    names = {name for group in used for name in group.split()}
    defined = set(re.findall(r"\.([A-Za-z][A-Za-z0-9_-]*)", css))
    undefined = sorted(name for name in names if name not in defined)
    assert not undefined, f"markup names classes the stylesheet never defines: {undefined}"


def test_no_module_opens_a_modal_dialog():
    """`alert`, `confirm` and `prompt` block the page's own event loop.

    A modal freezes every other control until someone dismisses it — including
    the control that would retry the failed call — and it wedges any automation
    driving the page, which is how a browser-verified check turns into a hang
    with no error. The status line reports failures instead.

    Comments naming the functions are allowed; this looks for calls.
    """
    for path in sorted((STATIC / "js").glob("*.js")):
        source = path.read_text(encoding="utf-8")
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith(("//", "*", "/*"))
        )
        for banned in ("alert(", "confirm(", "prompt("):
            assert banned not in code, f"{path.name} calls {banned[:-1]}(); use setStatus"


def test_no_two_elements_show_the_same_thing_under_the_same_heading():
    """The rollout appeared twice, identically titled, from both render paths.

    A reader saw the same generated sentence twice under the same words and had
    to decide which was authoritative. Neither was — they were copies, and a
    change to one render path would have left them disagreeing, which is worse
    than the duplication.

    The general property: no heading text appears twice in the markup.
    """
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    headings = re.findall(r"<(?:strong|h[1-4])[^>]*>([^<]{12,})</(?:strong|h[1-4])>", html)
    normalized = [" ".join(h.split()) for h in headings]
    duplicates = {h for h in normalized if normalized.count(h) > 1}
    assert not duplicates, f"the same heading titles two elements: {sorted(duplicates)}"


def test_the_page_prints_no_measured_number_of_its_own():
    """The Level 5 tag read "AUC 0.93" — a probe result, hand-typed, on a panel
    that runs a harder setting than the probe measured.

    The probe scores the widened alphabet on *gold* case roles; this panel infers
    them, and the README's own comparison says the dashboard's ranking is
    materially lower. Printing the probe's number beside the dashboard's output
    attributes an easier setting's result to a harder one. It was also a manuscript
    number typed into an interface, which is the thing `{{TOKEN}}` hydration exists
    to prevent — except no hydration reaches static HTML, so nothing would ever
    have corrected it.

    NEGATIVE CONTROL: structural counts the page owns — alphabet sizes, lexicon
    size, matrix shapes — are not measurements and must remain.
    """
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    tags = re.findall(r'class="level-tag">([^<]+)<', html)
    assert tags, "no level tags found; this test would be vacuous"
    for tag in tags:
        assert not re.search(r"\b(AUC|accuracy|perplexity|nats|p\s*[=<])\b", tag, re.I), (
            f"level tag quotes a measured result: {tag!r}"
        )
    # Negative control: the structural counts are still there.
    assert any("30 Symbols" in tag for tag in tags)
    assert any("2,000-Word Lexicon" in tag for tag in tags)


def test_every_javascript_module_is_reachable_from_main():
    """No orphan modules, and no module the entry point cannot reach.

    A file in `js/` that nothing imports is dead code the browser never loads,
    which is the same failure as a dead control: it looks like it works because
    nothing exercises it.
    """
    modules = {p.stem for p in (STATIC / "js").glob("*.js")}
    assert modules, "no JavaScript modules found"
    assert "main" in modules

    reachable, frontier = {"main"}, ["main"]
    while frontier:
        current = frontier.pop()
        source = (STATIC / "js" / f"{current}.js").read_text(encoding="utf-8")
        for imported in re.findall(r"(?:from\s+|import\()'\./([A-Za-z0-9_]+)\.js'", source):
            if imported not in reachable:
                reachable.add(imported)
                frontier.append(imported)

    orphans = sorted(modules - reachable)
    assert not orphans, f"modules nothing imports: {orphans}"

    missing = sorted(reachable - modules)
    assert not missing, f"main imports modules that do not exist: {missing}"


def test_no_module_is_a_monolith():
    """Each module stays reviewable on its own."""
    for path in sorted((STATIC / "js").glob("*.js")):
        length = len(path.read_text(encoding="utf-8").splitlines())
        assert length < 400, f"{path.name} is {length} lines; split it"


def test_shared_state_lives_in_one_module():
    """Cross-module values go through `store`, not another module's globals.

    Reaching into a sibling's top-level `let` would make the import graph a lie:
    the dependency would be real and invisible.
    """
    for name in ("inference", "chat", "matrices", "speech"):
        source = (STATIC / "js" / f"{name}.js").read_text(encoding="utf-8")
        for shared in ("currentInference", "staticMatrices", "chatHistory", "chatBusy"):
            bare = re.search(rf"^\s*(?:let|var)\s+{shared}\b", source, re.M)
            assert bare is None, f"{name}.js redeclares shared state {shared}"


def test_inline_handlers_are_exported_to_the_global_scope():
    """ES modules do not populate `window`; inline handlers resolve against it.

    Every function the markup names must be attached in `main.js`, or the
    control is dead the moment the page loads.
    """
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    main = (STATIC / "js" / "main.js").read_text(encoding="utf-8")

    handlers = set(re.findall(r'on(?:click|change|input)="([A-Za-z_][A-Za-z0-9_]*)\(', html))
    assert handlers, "no inline handlers found; the scan is not looking at the page"

    assigned = set(re.findall(r"^\s{4}([A-Za-z_][A-Za-z0-9_]*),\s*$", main, re.M))
    missing = sorted(handlers - assigned)
    assert not missing, f"inline handlers not attached to window in main.js: {missing}"


# ── CONFIGURATION ────────────────────────────────────────────────────────────


def test_config_defaults_are_internally_consistent():
    from ailm.dashboard.config import DashboardConfig

    config = DashboardConfig()
    assert config.default_reply_words <= config.max_continuation_words
    assert 0.0 in config.alpha_grid, "the identity anchor is missing"
    assert list(config.alpha_grid) == sorted(config.alpha_grid)
    assert config.host == "127.0.0.1", "the default bind address is not loopback"
    assert len(config.api_token.encode("utf-8")) >= 32
    assert config.max_concurrent_requests > 0
    assert config.max_requests_per_minute > 0
    assert config.request_timeout_seconds > 0


@pytest.mark.parametrize(
    "field,value,fragment",
    [
        ("port", 0, "outside 1-65535"),
        ("port", 70000, "outside 1-65535"),
        ("max_spoken_chars", 0, "must be positive"),
        ("max_history_turns", -1, "must be positive"),
        ("default_reply_words", 500, "exceeds"),
        ("default_narrative_alpha", 1.5, "outside"),
        ("alpha_grid", (0.1, 0.5), "identity anchor"),
        ("alpha_grid", (0.0, 0.5, 0.1), "ascending"),
        ("host", "0.0.0.0", "not loopback"),
        ("host", "192.0.2.10", "not loopback"),
        ("max_concurrent_requests", 0, "must be positive"),
        ("max_requests_per_minute", 0, "must be positive"),
        ("request_timeout_seconds", 0, "must be positive"),
        ("request_timeout_seconds", float("nan"), "must be finite"),
        ("request_timeout_seconds", float("inf"), "must be finite"),
        ("default_narrative_alpha", float("nan"), "must be finite"),
        ("default_narrative_alpha", float("inf"), "must be finite"),
        ("alpha_grid", (0.0, float("inf")), "finite weights"),
        ("alpha_grid", (0.0, -0.1), "outside \\[0, 1\\]"),
        ("alpha_grid", (0.0, 1.1), "outside \\[0, 1\\]"),
        ("api_token", "too-short", "at least 32"),
    ],
)
def test_an_unusable_configuration_is_refused(field, value, fragment):
    """NEGATIVE CONTROL: a typo in a limit must not be defaulted away.

    A quiet fallback is exactly wrong here: the operator believes they changed a
    setting and the process runs on something else.
    """
    from ailm.dashboard.config import DashboardConfig

    with pytest.raises(ValueError, match=fragment):
        DashboardConfig(**{field: value})


def test_environment_overrides_are_applied_and_bad_ones_refused():
    """POSITIVE: an override takes effect. NEGATIVE: an unparseable one raises."""
    from ailm.dashboard.config import ENV_PREFIX, DashboardConfig, config_from_env

    overridden = config_from_env(environ={f"{ENV_PREFIX}MAX_SPOKEN_CHARS": "120"})
    assert overridden.max_spoken_chars == 120
    assert overridden.port == DashboardConfig().port, "an unset field changed"

    assert config_from_env(environ={f"{ENV_PREFIX}ALPHA_GRID": "0,0.2,0.4"}).alpha_grid == (
        0.0,
        0.2,
        0.4,
    )

    with pytest.raises(ValueError, match="not a int"):
        config_from_env(environ={f"{ENV_PREFIX}PORT": "not-a-port"})

    for name, raw, fragment in (
        ("REQUEST_TIMEOUT_SECONDS", "nan", "must be finite"),
        ("REQUEST_TIMEOUT_SECONDS", "inf", "must be finite"),
        ("DEFAULT_NARRATIVE_ALPHA", "-inf", "must be finite"),
        ("ALPHA_GRID", "0,inf", "finite weights"),
        ("ALPHA_GRID", "-0.1,0", "outside"),
        ("ALPHA_GRID", "0,1.1", "outside"),
    ):
        with pytest.raises(ValueError, match=fragment):
            config_from_env(environ={f"{ENV_PREFIX}{name}": raw})

    # An empty value is not an override; it is an unset variable.
    assert config_from_env(environ={f"{ENV_PREFIX}PORT": "  "}).port == DashboardConfig().port


def test_the_public_config_omits_the_bind_address():
    """The page knows where it was served from; the response need not say."""
    from ailm.dashboard.config import DashboardConfig

    public = DashboardConfig().to_public_dict()
    assert "host" not in public and "port" not in public
    assert "api_token" not in public
    assert public["max_spoken_chars"] == DashboardConfig().max_spoken_chars
    assert isinstance(public["alpha_grid"], list), "tuples are not JSON"


def test_server_json_serialization_rejects_nonstandard_nan_tokens():
    source = (Path(__file__).resolve().parents[1] / "src/ailm/dashboard/server.py").read_text(
        encoding="utf-8"
    )
    required = "json.dumps(data, allow_nan=False)"
    assert required in source

    # NEGATIVE CONTROL: the historical permissive call must fail this guard.
    planted = source.replace(required, "json.dumps(data)", 1)
    assert required not in planted


def test_the_page_reads_its_bounds_from_the_server():
    """No limit is transcribed into the front end by hand.

    The reply-length input carried `max="200"` and the speech cap was a
    `const` in a script block, both copied from constants they had no link to. A
    limit written twice is a limit that will disagree with itself.
    """
    config_js = (STATIC / "js" / "config.js").read_text(encoding="utf-8")
    assert "/api/config" in config_js, "the page never fetches its configuration"
    assert "applyToControls" in config_js
    initializer = re.search(r"export const cfg\s*=\s*\{(?P<body>.*?)\};", config_js, re.S)
    assert initializer is not None
    assert not initializer.group("body").strip(), (
        "browser configuration contains transcribed defaults instead of failing closed"
    )
    main_js = (STATIC / "js" / "main.js").read_text(encoding="utf-8")
    failed_branch = main_js[main_js.index("if (config.loadFailed)") :]
    failed_branch = failed_branch[: failed_branch.index("}\n")]
    assert "return;" in failed_branch, (
        "a failed configuration fetch still starts requests with unknown limits"
    )

    speech_js = (STATIC / "js" / "speech.js").read_text(encoding="utf-8")
    assert "cfg.max_spoken_chars" in speech_js
    assert "cfg.speech_watchdog_ms" in speech_js
    assert not re.search(r"const MAX_SPOKEN_CHARS\s*=\s*\d", speech_js), (
        "the spoken-length cap is hardcoded in the page again"
    )


def test_static_assets_are_served_uncacheable():
    """A cached stylesheet or module makes verification read the wrong file.

    `SimpleHTTPRequestHandler` sends `Last-Modified` and honors
    `If-Modified-Since` — correct for a file server, wrong for this one. It was
    measured live: an edited `dashboard.css` was served byte-identical to disk
    while the loaded page's CSSOM still held the previous rules, surviving a
    reload and a cache-busted URL (the query string busts the document, not its
    subresources). A fix then looks broken and a break looks fixed, and no test
    can see it, because the stale copy lives in the browser and every test
    reads the file.

    NEGATIVE CONTROL: the header must reach the JSON path too — a cached
    `/api/config` would pin the page's bounds to a previous launch — and it must
    be set in exactly one place, since two would eventually disagree.
    """
    from ailm.dashboard import server as dashboard_server

    class _Probe(dashboard_server.DashboardHandler):
        """A handler with no socket, recording only what headers it sends."""

        def __init__(self):
            self.sent: list[tuple[str, str]] = []
            self.request_version = "HTTP/1.1"
            self._headers_buffer: list[bytes] = []

        def send_header(self, key, value):
            self.sent.append((key, value))

        def flush_headers(self):
            pass

    probe = _Probe()
    probe.end_headers()

    directives = [value for key, value in probe.sent if key.lower() == "cache-control"]
    assert directives, "no Cache-Control header is sent"
    assert "no-store" in directives[0], f"caching is allowed: {directives[0]}"

    # One implementation covers the static tree and the JSON responses, because
    # `_send_json` ends its headers through the same override.
    source = Path(dashboard_server.__file__).read_text(encoding="utf-8")
    assert source.count("Cache-Control") == 1, (
        "the header is set in more than one place; they will disagree"
    )
    assert "def end_headers" in source


def test_a_button_variant_can_actually_override_the_base():
    """The base rule is `button.btn-primary`; a bare `.btn-x` cannot beat it.

    Specificity 0,1,1 versus 0,1,0 — source order never gets a say, so every
    variant renders the default green no matter where its rule sits. This is
    exactly what happened when the button colors moved out of inline `style=`
    attributes, and no DOM test could see it: the classes were present, the
    markup was correct, and the page was the wrong color.

    The check is the general property — any bare-class rule setting a property
    the qualified base rule also sets is outranked — not a check for the four
    variants that exist today.
    """
    css = (STATIC / "dashboard.css").read_text(encoding="utf-8")
    rules = re.findall(r"([^{}]+)\{([^{}]*)\}", css)

    def declared(selector_test):
        properties = set()
        for selector, body in rules:
            if selector_test(" ".join(selector.split())):
                properties |= {
                    line.split(":", 1)[0].strip() for line in body.split(";") if ":" in line
                }
        return properties

    base = declared(lambda s: s == "button.btn-primary")
    assert base, "the base button rule is gone; this test would be vacuous"

    outranked = {}
    for selector, body in rules:
        name = " ".join(selector.split())
        if not re.fullmatch(r"\.btn-[a-z-]+", name):
            continue
        properties = {line.split(":", 1)[0].strip() for line in body.split(";") if ":" in line}
        clash = properties & base
        if clash:
            outranked[name] = sorted(clash)
    assert not outranked, (
        f"bare-class rules lose to button.btn-primary and will not apply: {outranked}. "
        f"Qualify them with `button` so source order decides."
    )


def test_rebuilding_the_timeline_also_renders_it():
    """A timeline that is not showing a step is not in a valid state.

    `setupTimeline` rebuilt the slider and the character pills and left the
    rendering of step 0 to its caller. The inspector path called it; the chat
    path did not. So after a chat turn the page showed the new sentence's
    characters above the *previous* turn's posteriors — a stale Level 1
    observation, a stale case distribution, a stale clause posterior, all of
    them well-formed and confidently wrong.

    Worse, the chat disclaimer promises that every analysis panel updates to the
    current turn. That made the defect a claim the page printed and did not keep,
    which is the failure mode this project's whole claims discipline exists to
    prevent — it just happened in HTML rather than in a manuscript.

    The fix is structural: building the timeline and rendering its first step is
    one operation, so no caller can perform half of it.
    """
    inference = (STATIC / "js" / "inference.js").read_text(encoding="utf-8")

    body = inference[inference.index("function setupTimeline(") :]
    body = body[: body.index("\nfunction ")]
    assert "onSliderChange(0)" in body, (
        "setupTimeline no longer renders its first step; a caller has to remember to"
    )

    # NEGATIVE CONTROL: no caller may render it a second time. Two renders is
    # how the responsibility drifted back to the callers last time — one of
    # them had it, so the other looked optional.
    for path in sorted((STATIC / "js").glob("*.js")):
        source = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(source.splitlines(), 1):
            if "onSliderChange(0)" not in line:
                continue
            in_setup = path.name == "inference.js" and "onSliderChange(0)" in body
            assert in_setup, f"{path.name}:{line_number} renders step 0 outside setupTimeline"

    chat = (STATIC / "js" / "chat.js").read_text(encoding="utf-8")
    assert "setupTimeline(" in chat, "the chat path no longer refreshes the panels at all"


# ── SERVER: real loopback HTTP requests ──────────────────────────────────────


def _request(
    raw: str,
    *,
    services=None,
    authenticated: bool = True,
    config: DashboardConfig | None = None,
    explorer_page: Path | None = None,
) -> tuple[int, dict, bytes]:
    """Drive one wire-format request through a real loopback HTTP server."""
    from ailm.dashboard import server as dashboard_server

    active_config = dashboard_server.CONFIG if config is None else config
    handler = partial(
        dashboard_server.DashboardHandler,
        services=dashboard_server.DEFAULT_SERVICES if services is None else services,
        config=active_config,
        **({} if explorer_page is None else {"explorer_page": explorer_page}),
    )
    httpd = dashboard_server.ReusableThreadingServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    chunks: list[bytes] = []
    try:
        raw = raw.replace("{PORT}", str(httpd.server_address[1]))
        if authenticated and " /api/" in raw and "\r\nAuthorization:" not in raw:
            raw = raw.replace(
                "\r\n\r\n",
                f"\r\nAuthorization: Bearer {active_config.api_token}\r\n\r\n",
                1,
            )
        with socket.create_connection(httpd.server_address, timeout=2) as connection:
            # The 2-second budget above is a connect deadline, not a response
            # deadline: a real engine call (cold numerical import, or the slow
            # path this test exercises) can answer correctly after it. This
            # helper's wait for a response is a finite test allowance sized
            # above the server's `request_timeout_seconds` — which bounds the
            # server's socket operations, not its engine computation — so a
            # slow-but-correct reply is read rather than raced.
            connection.settimeout(active_config.request_timeout_seconds + 5)
            connection.sendall(raw.encode("utf-8"))
            connection.shutdown(socket.SHUT_WR)
            while chunk := connection.recv(65_536):
                chunks.append(chunk)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    response = b"".join(chunks)
    head, _, body = response.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").splitlines()
    status = int(lines[0].split()[1])
    headers = {}
    for line in lines[1:]:
        key, _, value = line.partition(":")
        headers[key.strip().lower()] = value.strip()
    return status, headers, body


def _post(
    path: str,
    body: str,
    *,
    length: str | None = None,
    services=None,
    authenticated: bool = True,
    content_type: str = "application/json",
    extra_headers: str = "",
    host: str = "x",
    config: DashboardConfig | None = None,
) -> tuple[int, dict, bytes]:
    declared = str(len(body.encode("utf-8"))) if length is None else length
    return _request(
        f"POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: {content_type}\r\n"
        f"Content-Length: {declared}\r\n{extra_headers}\r\n{body}",
        services=services,
        authenticated=authenticated,
        config=config,
    )


def test_the_config_endpoint_serves_what_the_server_enforces():
    """POSITIVE: `GET /api/config` returns the live configuration.

    NEGATIVE CONTROL: the bind address must NOT be in it. Echoing a host into a
    response body tells a reader something about the machine they did not ask
    for, and `to_public_dict` drops it on purpose.
    """
    status, headers, body = _request("GET /api/config HTTP/1.1\r\nHost: x\r\n\r\n")
    assert status == 200
    payload = json.loads(body)
    assert payload["max_continuation_words"] == DashboardConfig().max_continuation_words
    assert "host" not in payload and "port" not in payload
    assert "no-store" in headers.get("cache-control", "")
    assert "access-control-allow-origin" not in headers


def test_configured_rollout_and_history_limits_are_enforced_end_to_end():
    """Real HTTP controls prove override values constrain the production engine.

    The default control is deliberately larger: a test that merely observed the
    engine's historical fixed constants would fail to distinguish the two
    configurations.
    """
    token = "configured-limit-test-token-000000000000000000000000"
    constrained = DashboardConfig(
        max_continuation_words=1,
        default_reply_words=1,
        max_history_turns=2,
        api_token=token,
    )
    history = [{"role": "user", "text": f"turn {index}"} for index in range(8)]

    infer_status, _, infer_body = _post(
        "/api/infer",
        json.dumps({"text": "active inference", "n_words": 7}),
        config=constrained,
    )
    chat_status, _, chat_body = _post(
        "/api/chat",
        json.dumps({"message": "continue", "history": history, "n_words": 7}),
        config=constrained,
    )
    assert (infer_status, chat_status) == (200, 200)
    assert json.loads(infer_body)["topdown_continuation"]["requested_words"] == 1
    assert json.loads(chat_body)["turn_index"] == 2

    default_status, _, default_body = _post(
        "/api/infer",
        json.dumps({"text": "active inference", "n_words": 7}),
    )
    default_chat_status, _, default_chat_body = _post(
        "/api/chat",
        json.dumps({"message": "continue", "history": history, "n_words": 1}),
    )
    assert (default_status, default_chat_status) == (200, 200)
    assert json.loads(default_body)["topdown_continuation"]["requested_words"] == 7
    assert json.loads(default_chat_body)["turn_index"] == len(history)


def test_every_api_endpoint_requires_the_process_token():
    """Missing and wrong credentials fail; the real process token succeeds."""
    missing, _, _ = _request("GET /api/config HTTP/1.1\r\nHost: x\r\n\r\n", authenticated=False)
    wrong, _, _ = _request(
        "GET /api/config HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer wrong\r\n\r\n",
        authenticated=False,
    )
    valid, _, body = _request("GET /api/config HTTP/1.1\r\nHost: x\r\n\r\n")
    assert (missing, wrong, valid) == (401, 401, 200)
    assert "api_token" not in json.loads(body)


def test_static_entry_issues_a_strict_httponly_auth_cookie():
    status, headers, _ = _request("GET / HTTP/1.1\r\nHost: x\r\n\r\n", authenticated=False)
    cookie = headers.get("set-cookie", "")
    assert status == 200
    assert "ailm_dashboard_token=" in cookie
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie and "Path=/" in cookie


def test_hostile_origin_simple_post_is_refused_even_with_the_token():
    """Reproduce the browser-simple CSRF path reported by the security audit."""
    status, headers, body = _post(
        "/api/infer",
        "{}",
        content_type="text/plain",
        extra_headers="Origin: https://attacker.example\r\n",
    )
    assert status == 403
    assert json.loads(body)["error"] == "cross-origin request refused"
    assert "access-control-allow-origin" not in headers

    same_origin, _, _ = _post(
        "/api/infer",
        "{}",
        extra_headers="Origin: http://127.0.0.1:{PORT}\r\n",
        host="127.0.0.1:{PORT}",
    )
    assert same_origin == 200


def test_post_requires_json_and_preflight_obeys_the_same_policy():
    status, _, _ = _post("/api/infer", "{}", content_type="text/plain")
    assert status == 415

    hostile, _, _ = _request(
        "OPTIONS /api/infer HTTP/1.1\r\nHost: 127.0.0.1:{PORT}\r\n"
        "Origin: https://attacker.example\r\n\r\n"
    )
    allowed, _, _ = _request(
        "OPTIONS /api/infer HTTP/1.1\r\nHost: 127.0.0.1:{PORT}\r\n"
        "Origin: http://127.0.0.1:{PORT}\r\n\r\n"
    )
    assert hostile == 403
    assert allowed == 204


def test_an_unknown_endpoint_is_refused_on_both_verbs():
    assert _post("/api/nope", "{}")[0] == 404
    # GET falls through to the static tree, so an unknown path is a 404 from
    # the file server rather than from the API branch.
    assert _request("GET /api/nope HTTP/1.1\r\nHost: x\r\n\r\n")[0] == 404


def test_a_malformed_content_length_is_a_bad_request_not_a_traceback():
    """The parse was once outside the try, so a bad header raised out of the handler."""
    status, _, body = _post("/api/infer", "{}", length="banana")
    assert status == 400
    assert "malformed" in json.loads(body)["error"]


def test_an_oversized_body_is_refused_before_it_is_read():
    """An unbounded client-declared length is an allocation, not a request."""
    over = DashboardConfig().max_request_bytes + 1
    status, _, body = _post("/api/infer", "{}", length=str(over))
    assert status == 413
    assert str(DashboardConfig().max_request_bytes) in json.loads(body)["error"]


@pytest.mark.parametrize("payload", ["not json at all", "[1, 2, 3]"])
def test_a_body_that_is_not_a_json_object_is_a_bad_request(payload: str):
    """NEGATIVE CONTROL for the object check: a JSON *array* parses and is still wrong."""
    status, _, body = _post("/api/infer", payload)
    assert status == 400
    assert "bad request" in json.loads(body)["error"]


def test_the_infer_endpoint_returns_the_hierarchy():
    status, _, body = _post("/api/infer", json.dumps({"text": "did she go home ?", "n_words": 3}))
    assert status == 200
    payload = json.loads(body)
    assert set(payload["levels"]) >= {"l1_letter", "l5_clause", "l6_narrative"}
    assert payload["topdown_continuation"]["words_generated"] <= 3


def test_the_chat_endpoint_derives_the_phase_when_none_is_given():
    """Omitting `narrative_phase` is the interesting case.

    The engine derives it from the turn index, which is the whole reason Level 6
    means anything in a conversation. NEGATIVE CONTROL: an explicit phase must
    override the derivation, or the parameter would be decorative.
    """
    history = [{"role": "user", "text": "hello"}, {"role": "model", "text": "the the"}]
    _, _, derived = _post(
        "/api/chat", json.dumps({"message": "is it working ?", "history": history})
    )
    payload = json.loads(derived)
    assert payload["turn_index"] == 1
    assert payload["phase_source"] == "derived_from_turn"

    _, _, explicit = _post(
        "/api/chat",
        json.dumps({"message": "is it working ?", "history": history, "narrative_phase": 2}),
    )
    assert json.loads(explicit)["read_as"]["narrative_phase"] == "closing"


def test_wire_values_that_do_not_parse_fall_back_rather_than_raising():
    """A chat client can send anything; the endpoint must not 500 on it."""
    status, _, body = _post(
        "/api/chat",
        json.dumps(
            {"message": "hi", "n_words": "lots", "narrative_alpha": "loud", "history": "no"}
        ),
    )
    assert status == 200
    assert json.loads(body)["reply"]


def test_wire_float_coercion_rejects_nan_and_keeps_a_real_number():
    """POSITIVE: a parseable alpha is used. NEGATIVE: NaN falls back.

    ``float("nan")`` parses, so the try/except cannot catch it, and a NaN alpha
    would propagate through every blend as NaN — the panels would render "NaN"
    in place of every probability with no error anywhere. The self-inequality
    check is the only line standing between the wire and that.
    """
    from ailm.dashboard.server import _as_float

    assert _as_float("0.5", 0.1) == 0.5
    assert _as_float(0.25, 0.1) == 0.25
    assert _as_float("nan", 0.1) == 0.1
    assert _as_float("NaN", 0.1) == 0.1
    assert _as_float(float("nan"), 0.1) == 0.1
    assert _as_float(None, 0.1) == 0.1


def test_an_engine_failure_is_a_generic_500_with_detail_only_in_the_log(caplog):
    """The handler's last resort: an unexpected engine exception must still
    produce a well-formed JSON response and a log line, because a handler that
    raises tears down the connection and the page sees only a network error.

    NEGATIVE CONTROL: the same request against the real engine is a 200, so
    this cannot pass by a handler that answers 500 to everything.
    """
    import logging

    from ailm.dashboard.server import DashboardServices

    body = json.dumps({"message": "hello", "n_words": 2})
    status, _, _ = _post("/api/chat", body)
    assert status == 200

    def _explode(*args, **kwargs):
        raise ValueError("engine secret: /private/model/path")

    services = DashboardServices(chat=_explode)
    with caplog.at_level(logging.ERROR, logger="ailm.dashboard.server"):
        status, _, failed_body = _post("/api/chat", body, services=services)
    assert status == 500
    assert json.loads(failed_body)["error"] == "internal server error"
    assert b"engine secret" not in failed_body
    assert b"/private/model/path" not in failed_body
    assert any("Inference error" in record.message for record in caplog.records), (
        "the failure was answered but never logged; the operator cannot see it"
    )


def test_a_matrix_dump_failure_is_a_500_not_a_dropped_connection():
    """Same last resort on the GET side, where the models load lazily."""
    from ailm.dashboard.server import DashboardServices

    def _explode():
        raise RuntimeError("models unavailable")

    services = DashboardServices(models=_explode)
    status, _, body = _request(
        "GET /api/static_matrices HTTP/1.1\r\nHost: x\r\n\r\n", services=services
    )
    assert status == 500
    assert json.loads(body)["error"] == "internal server error"
    assert b"models unavailable" not in body


def test_server_never_runs_more_than_its_worker_ceiling():
    """Two handlers may run together; a third waits without creating a thread."""
    from ailm.dashboard import server as dashboard_server
    from ailm.dashboard.server import DashboardServices

    lock = threading.Lock()
    release = threading.Event()
    saturated = threading.Event()
    state = {"active": 0, "maximum": 0}

    def _blocking_infer(*args, **kwargs):
        with lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
            if state["active"] == 2:
                saturated.set()
        assert release.wait(5), "test did not release the bounded workers"
        with lock:
            state["active"] -= 1
        return {"ok": True}

    handler = partial(
        dashboard_server.DashboardHandler,
        services=DashboardServices(infer=_blocking_infer),
    )
    httpd = dashboard_server.ReusableThreadingServer(
        ("127.0.0.1", 0), handler, max_workers=2, request_timeout=5
    )
    serving = threading.Thread(target=httpd.serve_forever, daemon=True)
    serving.start()
    statuses: list[int] = []

    def _client() -> None:
        body = "{}"
        raw = (
            "POST /api/infer HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
            f"Authorization: Bearer {dashboard_server.CONFIG.api_token}\r\n"
            f"Content-Length: {len(body)}\r\n\r\n{body}"
        )
        with socket.create_connection(httpd.server_address, timeout=2) as connection:
            # The 2-second budget above is a connect deadline, not a response
            # deadline. What can delay a response here is the test's own
            # machinery — the server's 5-second socket ceiling, the `release`
            # gate holding a worker up to 5 s, and a queued third client — so
            # read for a bounded 15 s, above every bound this test enforces,
            # rather than for the connect budget.
            connection.settimeout(15)
            connection.sendall(raw.encode("utf-8"))
            connection.shutdown(socket.SHUT_WR)
            response = b""
            while chunk := connection.recv(65_536):
                response += chunk
        statuses.append(int(response.split(b" ", 2)[1]))

    clients = [threading.Thread(target=_client) for _ in range(3)]
    try:
        for client in clients:
            client.start()
        assert saturated.wait(5), "two workers never ran concurrently"
        assert state["maximum"] == 2
        assert state["active"] == 2
        release.set()
        for client in clients:
            client.join(timeout=10)
            assert not client.is_alive()
    finally:
        release.set()
        httpd.shutdown()
        httpd.server_close()
        serving.join(timeout=5)
    assert sorted(statuses) == [200, 200, 200]
    assert state["maximum"] == 2


def test_authenticated_api_rate_has_real_positive_and_negative_controls():
    """Two requests pass and the next is refused on the same live server."""
    from ailm.dashboard import server as dashboard_server

    httpd = dashboard_server.ReusableThreadingServer(
        ("127.0.0.1", 0),
        dashboard_server.DashboardHandler,
        max_requests_per_minute=2,
        request_timeout=5,
    )
    serving = threading.Thread(target=httpd.serve_forever, daemon=True)
    serving.start()

    def _status(token: str) -> int:
        raw = (
            "GET /api/config HTTP/1.1\r\nHost: x\r\n"
            f"Authorization: Bearer {token}\r\nConnection: close\r\n\r\n"
        )
        with socket.create_connection(httpd.server_address, timeout=2) as connection:
            # The 2-second budget above is a connect deadline, not a response
            # deadline; the server's own socket ceiling here is 5 s. Under
            # suite load a >2 s recv gap would fail this client thread — the
            # flake class R1 fixed in `_request` — so read for a bounded 15 s
            # instead of the connect budget.
            connection.settimeout(15)
            connection.sendall(raw.encode("utf-8"))
            response = b""
            while chunk := connection.recv(65_536):
                response += chunk
        return int(response.split(b" ", 2)[1])

    try:
        token = dashboard_server.CONFIG.api_token
        assert [_status(token), _status(token)] == [200, 200]
        assert _status(token) == 429
        # Authentication is still evaluated independently; the limiter is not
        # a blanket response that hides a missing credential.
        assert _status("wrong") == 401
    finally:
        httpd.shutdown()
        httpd.server_close()
        serving.join(timeout=5)


@pytest.mark.timeout(240)
def test_run_dashboard_binds_what_it_is_given_and_shuts_down_on_interrupt(tmp_path):
    """POSITIVE: overrides reach the socket; Ctrl-C closes the server.

    A real package subprocess opens the loopback socket, answers its public
    configuration endpoint, and receives SIGINT.  This proves argument
    precedence, the live handler configuration, the announced address and the
    shutdown path together at the process boundary.

    NEGATIVE CONTROL: with no overrides, the configured defaults must win; a
    run_dashboard that hardcoded its bind address passed exactly this way once
    (see `test_the_environment_moves_the_port_the_server_actually_binds`).
    """
    configured_port = _unused_loopback_port()
    explicit_port = _unused_loopback_port()
    while explicit_port == configured_port:
        explicit_port = _unused_loopback_port()
    env = _dashboard_environment(tmp_path, port=configured_port)
    process = _launch_dashboard(env, str(explicit_port), "127.0.0.1")
    try:
        payload = _wait_for_dashboard(process, explicit_port)
        with pytest.raises(URLError):
            urlopen(f"http://127.0.0.1:{configured_port}/api/config", timeout=0.2)
    finally:
        stdout, stderr = _interrupt_dashboard(process)

    assert process.returncode == 0, stderr
    assert payload["max_spoken_chars"] == 321
    assert f"http://127.0.0.1:{explicit_port}/" in stdout
    assert "Shutting down dashboard server" in stdout


@pytest.mark.timeout(45)
def test_dashboard_cli_handles_sigint_inherited_as_ignored(tmp_path):
    """A real launcher may ignore SIGINT; the dashboard CLI must still close."""
    port = _unused_loopback_port()
    environment = _dashboard_environment(tmp_path, port=port)
    launcher = (
        "import os, signal, sys\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "os.execv(sys.executable, [sys.executable, '-m', 'ailm.dashboard'])\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", launcher],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_dashboard(process, port)
        process.send_signal(signal.SIGINT)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pytest.fail("dashboard kept an inherited ignored SIGINT disposition")
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=5)
    assert process.returncode == 0, stderr
    assert "Shutting down dashboard server" in stdout


def test_interrupt_after_worker_completion_preserves_the_interrupt():
    """A completed worker and interrupted start must return the same slot once.

    The profile hook delivers a real interrupt at Thread.start's return, after
    the real worker has returned its slot. It does not replace Thread.start or
    the semaphore: this is the observed startup/completion race, synchronized.
    """
    from ailm.dashboard.server import ReusableThreadingServer

    class FinishedHandler(BaseRequestHandler):
        def handle(self):
            pass

    server = ReusableThreadingServer(("127.0.0.1", 0), FinishedHandler, max_workers=1)
    request, peer = socket.socketpair()
    previous_profile = sys.getprofile()
    previous_signal = signal.signal(signal.SIGINT, signal.default_int_handler)

    def interrupt_completed_start(frame, event, argument):
        if frame.f_code is threading.Thread.start.__code__ and event == "return":
            sys.setprofile(None)
            # Acquiring the slot proves the real worker's finally completed.
            assert server._worker_slots.acquire(timeout=5)
            server._worker_slots.release()
            signal.raise_signal(signal.SIGINT)

    try:
        sys.setprofile(interrupt_completed_start)
        with pytest.raises(KeyboardInterrupt):
            server.process_request(request, ("127.0.0.1", 0))
        assert server._worker_slots.acquire(blocking=False)
        assert not server._worker_slots.acquire(blocking=False)
        server._worker_slots.release()
    finally:
        sys.setprofile(previous_profile)
        signal.signal(signal.SIGINT, previous_signal)
        request.close()
        peer.close()
        server.server_close()


def test_the_static_matrices_endpoint_serves_every_declared_matrix():
    status, _, body = _request("GET /api/static_matrices HTTP/1.1\r\nHost: x\r\n\r\n")
    assert status == 200
    payload = json.loads(body)
    assert set(payload) == {
        "A_letter",
        "A_g2p",
        "B_g2p",
        "D_g2p",
        "A_case",
        "B_case",
        "D_case",
        "clause_prior",
    }
    assert len(payload["A_letter"]) == 30


def test_integrated_explorer_serves_exact_generated_bytes_and_no_output_directory():
    """The embedded view uses the current generator output, not another model copy."""
    from ailm.dashboard.server import EXPLORER_PAGE

    assert EXPLORER_PAGE.is_file(), "run the web exporter and builder before this test"
    status, headers, body = _request("GET /explorer/?view=matrix HTTP/1.0\r\n\r\n")
    assert status == 200
    assert body == EXPLORER_PAGE.read_bytes()
    assert headers["content-type"] == "text/html; charset=utf-8"
    assert "no-store" in headers["cache-control"]
    # Negative control: the fixed route must not expose neighboring model/output files.
    status, _, _ = _request("GET /explorer/ailm_model.json HTTP/1.0\r\n\r\n")
    assert status == 404


def test_missing_explorer_reports_rebuild_and_recovers_after_generation(tmp_path):
    page = tmp_path / "ailm.html"
    status, _, body = _request("GET /explorer/ HTTP/1.0\r\n\r\n", explorer_page=page)
    assert status == 503
    error = json.loads(body)["error"]
    assert "scripts/export_web_model.py" in error and "scripts/build_web_ailm.py" in error
    # Positive control: creating a real artifact restores serving without a server restart.
    page.write_text("<!doctype html><title>Generated explorer</title>", encoding="utf-8")
    status, _, body = _request("GET /explorer/ HTTP/1.0\r\n\r\n", explorer_page=page)
    assert status == 200 and body == page.read_bytes()


def test_unreadable_explorer_is_an_explicit_error(tmp_path, caplog):
    # A directory at the exact expected file path produces a real OS read error.
    status, _, body = _request("GET /explorer/ HTTP/1.0\r\n\r\n", explorer_page=tmp_path)
    assert status == 500
    assert "permissions" in json.loads(body)["error"]
    assert "Cannot read generated explorer" in caplog.text


def test_reply_speech_attention_guard_rejects_hidden_views_and_background_tabs():
    """Execute the production predicate for every attention/opt-in combination."""
    source = (STATIC / "js" / "chat.js").read_text(encoding="utf-8")
    predicate = re.search(r"export function replySpeechAllowed\([^)]*\) \{[^}]*\}", source)
    assert predicate is not None
    program = (
        predicate.group(0)
        + """
import assert from 'node:assert/strict';
for (const inspectorHidden of [false, true]) {
    for (const documentHidden of [false, true]) {
        for (const speakReplies of [false, true]) {
            assert.equal(replySpeechAllowed(inspectorHidden, documentHidden, speakReplies),
                inspectorHidden === false && documentHidden === false && speakReplies === true);
        }
    }
}
"""
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", program], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    # Negative control: retaining only opt-in recreates the delayed-response defect.
    broken = program.replace("!inspectorHidden && !documentHidden && speakReplies", "speakReplies")
    assert broken != program
    control = subprocess.run(
        ["node", "--input-type=module", "-e", broken], capture_output=True, text=True, timeout=10
    )
    assert control.returncode != 0 and "AssertionError" in control.stderr
    # The response-time call must consume actual view, visibility and opt-in state.
    response_branch = source[source.index("const data = await response.json()") :]
    assert re.search(
        r"if \(replySpeechAllowed\(\s*document.getElementById\('inspector-view'\).hidden,"
        r"\s*document.hidden,\s*document.getElementById\('chat-speak-replies'\).checked,"
        r"\s*\)\) \{\s*speak\(data.reply,",
        response_branch,
    )


def test_cold_module_transport_is_serial_over_real_dashboard_http(tmp_path):
    """Fetch and link real ES modules; DOM evaluation is verified in the browser.

    Returning linked real module namespaces lets Node exercise the entry import
    prologue without inventing a DOM. This is transport proof, not full-page QA.
    A deliberately static import fan-out must fail the same serial contract.
    """
    from ailm.dashboard import server as dashboard_server

    main = (STATIC / "js" / "main.js").read_text(encoding="utf-8")
    program = r"""
import { SourceTextModule } from 'node:vm';
import { readFileSync } from 'node:fs';
const root = process.argv[1];
const source = readFileSync(process.argv[2], 'utf8');
const imports = source.slice(0, source.indexOf('export async function showDashboardView'));
const pending = new Map();
let active = 0;
let peak = 0;
const requests = [];
async function load(specifier, owner) {
    const url = new URL(specifier, owner.identifier).href;
    if (!pending.has(url)) pending.set(url, (async () => {
        active++; peak = Math.max(peak, active);
        const response = await fetch(url);
        if (!response.ok) throw new Error(`${url}: ${response.status}`);
        const text = await response.text();
        active--; requests.push(url);
        return new SourceTextModule(text, {identifier: url});
    })());
    return pending.get(url);
}
const entry = new SourceTextModule(imports, {
    identifier: root + '/js/main.js',
    importModuleDynamically: async (specifier, owner) => {
        const module = await load(specifier, owner);
        if (module.status === 'unlinked') await module.link(load);
        return module.namespace;
    },
});
await entry.link(load);
// Refuse fan-out before evaluating any browser-only module body.
if (peak > 1) throw new Error('Nonserial or incomplete module transport');
await entry.evaluate();
console.log(JSON.stringify({peak, requests, modules: pending.size}));
if (peak !== 1 || pending.size !== 8) throw new Error('Nonserial or incomplete module transport');
"""
    httpd = dashboard_server.ReusableThreadingServer(
        ("127.0.0.1", 0), dashboard_server.DashboardHandler
    )
    serving = threading.Thread(target=httpd.serve_forever, daemon=True)
    serving.start()
    original_bound = dashboard_server.CONFIG.max_concurrent_requests
    assert httpd.max_workers == original_bound

    def run(source: str, name: str):
        path = tmp_path / name
        path.write_text(source, encoding="utf-8")
        return subprocess.run(
            [
                "node",
                "--experimental-vm-modules",
                "--input-type=module",
                "-e",
                program,
                f"http://127.0.0.1:{httpd.server_address[1]}",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

    try:
        result = run(main, "main.js")
        assert result.returncode == 0, result.stderr
        observed = json.loads(result.stdout)
        assert observed["peak"] == 1 and observed["modules"] == 8
        assert len(observed["requests"]) == len(set(observed["requests"])) == 8
        fanout = "\n".join(
            f"import './{path.name}';"
            for path in sorted((STATIC / "js").glob("*.js"))
            if path.stem != "main"
        )
        broken = run(fanout + "\n" + main, "parallel-main.js")
        assert broken.returncode != 0
        assert "Nonserial or incomplete" in broken.stderr or "fetch failed" in broken.stderr
    finally:
        httpd.shutdown()
        httpd.server_close()
        serving.join(timeout=5)
        assert not serving.is_alive()


def test_startup_keeps_live_config_failure_and_both_dom_ready_paths():
    source = (STATIC / "js" / "main.js").read_text(encoding="utf-8")
    assert "const config = await import('./config.js');" in source
    assert "if (config.loadFailed)" in source
    assert not re.search(r"const\s*\{[^}]*loadFailed", source)
    assert re.search(
        r"if \(document.readyState === 'loading'\) \{\s*"
        r"window.addEventListener\('DOMContentLoaded', startDashboard, \{ once: true \}\);"
        r"\s*\} else \{\s*startDashboard\(\);\s*\}",
        source,
    )
