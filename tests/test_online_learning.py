"""L1 online learning on non-stationary text — tests of decision entry 28.

What this file is for
---------------------
``docs/decisions.md`` entry 28 pre-registered the W2 increment where learning
could earn its way in: on a drifted (non-stationary) letter-bigram stream, an
agent that keeps learning should track the drift that a frozen (non-learning)
control cannot. The pre-registered predictions:

- **P1** — the drift is real: the two segments differ in held-out letter
  perplexity under the frozen matrix by more than tolerance.
- **P2** — frozen must lose: on held-out text from the second distribution,
  the online agent's perplexity is strictly lower than the frozen control's.
- **P3** — identity before the drift plus a re-fit control: the online agent
  at the drift point equals counting (lr=1 identity), and the frozen matrix
  re-estimated on the second segment scores at least as well as the online
  agent.

Two verification paths, matching how the recorded claims are produced:

- **Structural P1-P2-P3a checks** run the real pymdp online path on a bounded
  slice of the vendored stream (fast enough for a per-test timeout, real
  ``infer_parameters``). P1 (drift real) and P2 (frozen must lose) are
  robust at small slice sizes; so is P3a (lr=1 identity).
- **P3b (re-fit control) and ledger verification** are scale-dependent and
  are therefore checked where the recorded claim is made — on the *full*
  stream via the numpy closed form (fast, exact), within the float32
  conjugacy tolerance that decisions 22-23 license. The re-fit control needs
  enough post-drift data for the re-estimated matrix to be well-conditioned;
  on a tiny slice the online agent (which sees both segments) can beat a
  re-fit that saw only the small second segment, which is a sample-size
  artifact, not a violation of the pre-registration.

Claims discipline: this file **verifies**; ``scripts/record_online_learning_claims.py``
writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ailm.claims import load_ledger
from ailm.levels.online_learning import (
    closed_form_perplexity,
    letter_indices,
    online_letter_experiment_numpy,
    run_online_letter_experiment,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = PROJECT_ROOT / "vendor" / "brown_online_drift.json"
DRIFT_TOL = 0.01

#: float32 conjugacy agreement between pymdp's infer_parameters and the numpy
#: counting path (decisions 22-23): the two online matrices agree to ~1e-4
#: relative in perplexity, so ledger values are verified within this tolerance.
NUMPY_PYMDP_REL_TOL = 1e-3

#: The real pymdp path over the bounded slice is the expensive part of this
#: module, and it is paid once in the module-scoped ``result`` fixture. Fixture
#: setup is charged to the first test that requests it, so under the suite-wide
#: 60s default that test errors whenever the machine is busy — the module takes
#: about 32s to itself and rather more when the full suite is competing for the
#: same cores. The failure looked like four broken tests and was a clock. The
#: ceiling stays finite so a genuinely wedged run still fails, and the slice is
#: left alone: shrinking it to fit the default would buy runtime by weakening
#: the P1/P2/P3a checks the slice exists to support.
pytestmark = [
    pytest.mark.skipif(not ARTIFACT.exists(), reason="drift artifact absent"),
    pytest.mark.timeout(300),
]


@pytest.fixture(scope="module")
def streams():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    return payload["splits"]


@pytest.fixture(scope="module")
def bounded(streams):
    """A bounded slice: small enough for the real pymdp path to run in-test."""
    return {
        "pre_learn": streams["pre_learn"][:3000],
        "post_learn": streams["post_learn"][:3000],
        "pre_hold": streams["pre_hold"][:1000],
        "post_hold": streams["post_hold"][:1000],
    }


@pytest.fixture(scope="module")
def result(bounded):
    return run_online_letter_experiment(
        bounded["pre_learn"],
        bounded["post_learn"],
        bounded["pre_hold"],
        bounded["post_hold"],
    )


# ══ THE RESULT (structural, real pymdp on a bounded slice) ═══════════════════


def test_online_beats_frozen_on_post_drift_text(result):
    """(P2) The load-bearing clause: online beats frozen on the drifted eval."""
    assert result.ppl_online_post < result.ppl_frozen_post, (
        f"online {result.ppl_online_post:.4f} does not beat frozen "
        f"{result.ppl_frozen_post:.4f} on post-drift held-out text. Decision "
        f"28's claim that online conjugate updates recover the drift is "
        f"withdrawn if this no longer holds."
    )


def test_the_drift_is_real_p1(result):
    """(P1) The two segments genuinely differ under the frozen matrix."""
    assert result.max_drift > DRIFT_TOL, (
        f"frozen perplexity moves only {result.max_drift:.4f} between genres "
        f"(<= {DRIFT_TOL}); there is no drift to track."
    )


def test_online_pre_equals_frozen_before_drift(result):
    """(P3a) At lr=1 the online agent at the drift point equals counting."""
    assert float(np.max(np.abs(result.online_pre - result.frozen))) <= 1e-6


def test_result_carries_no_refusal(result):
    """P1, P2 and P3a pass on the bounded slice (refit control excluded)."""
    assert result.p1_passes and result.p2_passes, result.refuse_reason


# ══ P3b — RE-FIT CONTROL, WHERE THE RECORDED CLAIM IS MADE ════════════════════


def test_refit_control_on_the_full_stream(streams):
    """(P3b) On the full stream the re-fit scores at least as well as online.

    The re-fit control needs enough post-drift data to be well-conditioned,
    so it is verified on the full vendored stream (fast via the numpy path),
    not on the small pymdp slice — where the online agent, having seen both
    segments, can beat a re-fit that saw only a few thousand characters.
    """
    s = streams
    pre = letter_indices(s["pre_learn"])
    post = letter_indices(s["post_learn"])
    post_hold = letter_indices(s["post_hold"])

    def count(inds):
        m = np.full((30, 30), 1.0)
        for i in range(len(inds) - 1):
            m[inds[i + 1], inds[i]] += 1.0
        return m / m.sum(axis=0, keepdims=True)

    online = count(pre + post)
    refit = count(post)
    assert closed_form_perplexity(refit, post_hold) <= (
        closed_form_perplexity(online, post_hold) + 1e-9
    )


# ══ THE RESULT (ledger verification, numpy on the full stream) ═══════════════


def test_online_claims_verified(streams):
    """The recorded claims equal a re-derived numpy run on the same data.

    The numpy closed form is the exact identity-emission scorer; decisions
    22-23 prove pymdp's online conjugate path equals this counting at lr=1 to
    float32 rounding, and the recorder additionally cross-checks the P2 sign
    between the two paths before recording.
    """
    ledger = load_ledger(PROJECT_ROOT)
    r = online_letter_experiment_numpy(
        streams["pre_learn"],
        streams["post_learn"],
        streams["pre_hold"],
        streams["post_hold"],
    )
    expected = {
        "online_ppl_frozen_post": round(r["ppl_frozen_post"], 6),
        "online_ppl_online_post": round(r["ppl_online_post"], 6),
    }
    for label, value in expected.items():
        entry = ledger.get(label)
        assert entry is not None, (
            f"{label} is absent. Run scripts/record_online_learning_claims.py "
            f"— this test may not write it."
        )
        assert entry["source"].startswith("scripts/"), (
            f"{label} was last written by {entry['source']!r}."
        )
        assert entry["value"] == pytest.approx(value, rel=NUMPY_PYMDP_REL_TOL), (
            f"Ledger {label}={entry['value']!r} != numpy re-derivation "
            f"{value!r} (rel tol {NUMPY_PYMDP_REL_TOL})."
        )


def test_online_confirmed_flag_is_set(streams):
    """The recorded confirmation flag matches the re-derived P1-P3 outcome."""
    ledger = load_ledger(PROJECT_ROOT)
    r = online_letter_experiment_numpy(
        streams["pre_learn"],
        streams["post_learn"],
        streams["pre_hold"],
        streams["post_hold"],
    )
    entry = ledger.get("online_confirmed")
    assert entry is not None, "online_confirmed absent; run the recorder"
    assert int(entry["value"]) == 1, (
        "The recorded online_confirmed flag is not 1; the outcome entry in "
        "docs/decisions.md must match the measurement."
    )
    assert r["online_beats_frozen"], "numpy re-derivation contradicts P2"


# ══ ERROR PATHS ═══════════════════════════════════════════════════════════════


def test_letter_indices_rejects_a_degenerate_stream():
    with pytest.raises(ValueError, match="at least one transition"):
        letter_indices("a")  # fewer than two positions


def test_closed_form_perplexity_requires_two_positions():
    with pytest.raises(ValueError, match="at least two"):
        closed_form_perplexity(np.eye(30), [3])
