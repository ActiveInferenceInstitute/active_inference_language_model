#!/usr/bin/env python3
"""Probe pymdp sparse capabilities, blocking sites, and exact equivalence.

This probe establishes the exact relationship between pymdp's JAX backend and
sparse representations:

1. pymdp's JAX backend ALREADY contains sparse contraction machinery:
   `spm_dot_sparse` and `factor_dot_flex` in `maths.py`, `JAXSparse` handling in
   `inference.py`, and `sparse.todense` handling where dense arrays are needed.
2. BUT `Agent` construction and belief updating block sparse arrays through the
   public API:
   - `utils.validate_normalization` uses `jnp.sum(tensor, axis=axis)` which
     raises `TypeError: sum requires ndarray or scalar arguments, got BCOO`.
   - `control.compute_expected_state` returns BCOO arrays when contracting with
     sparse `B`, which causes downstream type mismatches in belief propagation.
   - If `B` has no leading batch dimension, `agent.py:263` raises
     `TypeError: broadcast_to requires ndarray or scalar arguments, got BCOO`.
3. After the generated three-site patch is applied to a throwaway copy of the
   installed package, a fresh Python process imports that copy and a sparse-B
   Stage 3 Agent is bit-identical to the dense Agent (max diff 0.000e+00), with
   identical trie-node posterior supports. No runtime function assignment is
   used to manufacture the result.

Usage:
    .venv/bin/python scripts/probe_pymdp_sparse.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pymdp  # noqa: E402
from pymdp import agent, utils  # noqa: E402

from ailm.levels.word import (  # noqa: E402
    END_INDEX,
    LETTERS,
    N_OBS,
    build_emission,
    build_position_transition,
    build_word_agent,
    build_word_prior,
    load_word_artifact,
)

PATCHED_WORKER_ARGUMENT = "--_ailm-patched-equivalence-worker"
RESULT_SENTINEL = "AILM_PATCHED_SPARSE_EQUIVALENCE "


def check_unpatched_failure_modes() -> list[dict[str, str]]:
    """Demonstrate the exact blocking call sites in unpatched pymdp."""
    failures = []
    V = 10
    data = jnp.ones((1, V), dtype=jnp.float32)
    idx = jnp.stack([jnp.arange(V), jnp.arange(V), jnp.zeros(V, dtype=int)], axis=1)[jnp.newaxis]
    bcoo_3d = sparse.BCOO((data[0], idx[0]), shape=(V, V, 1))
    bcoo_4d = sparse.BCOO((data, idx), shape=(1, V, V, 1))

    # Site 1: broadcast_to at agent.py:263 on unbatched BCOO
    try:
        jnp.broadcast_to(bcoo_3d, (1,) + bcoo_3d.shape)
    except Exception as e:
        failures.append(
            {
                "site": "pymdp/agent.py:263",
                "operation": "jnp.broadcast_to(b_f, (batch_size,) + b_f.shape)",
                "exception": f"{type(e).__name__}: {e}",
            }
        )

    # Site 2: validate_normalization at utils.py:109
    try:
        utils.validate_normalization(bcoo_4d, axis=1, tensor_name="B[0]")
    except Exception as e:
        failures.append(
            {
                "site": "pymdp/utils.py:109",
                "operation": "jnp.sum(tensor, axis=axis)",
                "exception": f"{type(e).__name__}: {e}",
            }
        )

    return failures


def _run_equivalence_in_current_environment() -> dict[str, float | bool]:
    """Run equivalence using the pymdp implementation already imported.

    This function is private because it is valid only inside the fresh worker
    whose import path points at an actually patched copy. The public function
    below constructs and authenticates that environment first.
    """
    artifact = load_word_artifact()
    V = len(artifact.vocab)

    # 4D BCOO with n_batch=1: (1, V, V, 1)
    data = jnp.ones((1, V), dtype=jnp.float32)
    idx = jnp.stack([jnp.arange(V), jnp.arange(V), jnp.zeros(V, dtype=int)], axis=1)[jnp.newaxis]
    bcoo_B0_4d = sparse.BCOO((data, idx), shape=(1, V, V, 1))

    B1_4d = jnp.array(build_position_transition(artifact), dtype=jnp.float32)[
        jnp.newaxis, :, :, jnp.newaxis
    ]

    B_sparse = [bcoo_B0_4d, B1_4d]
    emit = build_emission(artifact)
    A = [jnp.array(emit, dtype=jnp.float32)[jnp.newaxis]]
    position_start = np.zeros(artifact.n_positions, dtype=np.float32)
    position_start[0] = 1.0
    D = [
        jnp.array(build_word_prior(artifact), dtype=jnp.float32)[jnp.newaxis],
        jnp.array(position_start, dtype=jnp.float32)[jnp.newaxis],
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sparse_ag = agent.Agent(
            A=A,
            B=B_sparse,
            C=[jnp.zeros((1, N_OBS))],
            D=D,
            A_dependencies=[[0, 1]],
            B_dependencies=[[0], [1]],
        )
        dense_ag = build_word_agent(artifact)

    words_to_test = ["the", "and", "people", "government", "which", "state"]
    max_diff = 0.0

    for w in words_to_test:
        prior_d = [jnp.array(dense_ag.D[0]), jnp.array(position_start[np.newaxis])]
        prior_s = [jnp.array(sparse_ag.D[0]), jnp.array(position_start[np.newaxis])]

        letters = [LETTERS.index(c) for c in w] + [END_INDEX]
        for l_idx in letters:
            obs = [jnp.array([l_idx])]
            qs_d = dense_ag.infer_states(obs, prior_d)
            qs_s = sparse_ag.infer_states(obs, prior_s)

            diff0 = float(np.max(np.abs(np.array(qs_d[0]) - np.array(qs_s[0]))))
            diff1 = float(np.max(np.abs(np.array(qs_d[1]) - np.array(qs_s[1]))))
            max_diff = max(max_diff, diff0, diff1)

            prior_d = dense_ag.update_empirical_prior(jnp.array([[0, 0]]), qs_d)
            prior_s = sparse_ag.update_empirical_prior(jnp.array([[0, 0]]), qs_s)

    prefixes = ["th", "the", "co", "s"]
    all_supports_match = True
    for prefix in prefixes:
        prior_s = [jnp.array(sparse_ag.D[0]), jnp.array(position_start[np.newaxis])]
        qs = None
        for c in prefix:
            qs = sparse_ag.infer_states([jnp.array([ord(c) - 97])], prior_s)
            prior_s = sparse_ag.update_empirical_prior(jnp.array([[0, 0]]), qs)
        qs_w = np.array(qs[0]).ravel()
        support_words = [artifact.vocab[i] for i in np.where(qs_w > 1e-6)[0]]
        expected_words = [w for w in artifact.vocab if w.startswith(prefix)]
        if sorted(support_words) != sorted(expected_words):
            all_supports_match = False

    return {
        "max_diff": max_diff,
        "all_supports_match": all_supports_match,
    }


def _worker_result() -> dict[str, float | bool]:
    """Refuse an unpatched/wrong import before measuring in the child process."""

    expected_root = os.environ.get("AILM_PATCHED_PYMDP_ROOT", "")
    imported = Path(pymdp.__file__).resolve()
    if not expected_root:
        raise RuntimeError("patched sparse worker has no authenticated package root")
    expected_package = Path(expected_root).resolve(strict=True) / "pymdp"
    try:
        imported.relative_to(expected_package)
    except ValueError as error:
        raise RuntimeError(
            f"patched sparse worker imported {imported}, outside {expected_package}"
        ) from error
    utils_source = (expected_package / "utils.py").read_text(encoding="utf-8")
    control_source = (expected_package / "control.py").read_text(encoding="utf-8")
    agent_source = (expected_package / "agent.py").read_text(encoding="utf-8")
    required = (
        "sparse.sparsify(jnp.sum)",
        "maths._to_dense_if_sparse(qs_next_f)",
        "isinstance(b_f, sparse.BCOO)",
    )
    if not all(
        marker in source
        for marker, source in zip(
            required, (utils_source, control_source, agent_source), strict=True
        )
    ):
        raise RuntimeError("patched sparse worker package is missing a generated hunk")
    return _run_equivalence_in_current_environment()


def run_equivalence_probe(patch_text: str | None = None) -> dict[str, float | bool]:
    """Apply the generated patch, then measure it in a fresh real process.

    Args:
        patch_text: Exact generated unified diff. Tests may omit it; the
            generator is then imported only to derive the same canonical bytes.

    Returns:
        The worker's finite JSON result.
    """

    if patch_text is None:
        import generate_pymdp_sparse_patch

        patch_text = generate_pymdp_sparse_patch.build_patch()

    with tempfile.TemporaryDirectory() as temporary_directory:
        staged = Path(temporary_directory).resolve(strict=True)
        shutil.copytree(Path(pymdp.__file__).resolve().parent, staged / "pymdp")
        candidate = staged / "candidate.patch"
        candidate.write_text(patch_text, encoding="utf-8")
        subprocess.run(["git", "init", "-q", "."], cwd=staged, check=True)
        for arguments in (("apply", "--check"), ("apply",)):
            result = subprocess.run(
                ["git", *arguments, "-p1", str(candidate)],
                cwd=staged,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"generated sparse patch {' '.join(arguments)} failed: {result.stderr}"
                )

        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("PYTHON") and key != "AILM_PATCHED_PYMDP_ROOT"
        }
        environment.update(
            {
                "AILM_PATCHED_PYMDP_ROOT": str(staged),
                "PYTHONPATH": os.pathsep.join((str(staged), str(REPO_ROOT / "src"))),
                "PYTHONNOUSERSITE": "1",
                "PYTHONSAFEPATH": "1",
            }
        )
        worker = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), PATCHED_WORKER_ARGUMENT],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=240,
        )
        if worker.returncode != 0:
            raise RuntimeError(
                "patched sparse equivalence worker failed:\n"
                f"stdout:\n{worker.stdout}\nstderr:\n{worker.stderr}"
            )
        lines = [line for line in worker.stdout.splitlines() if line.startswith(RESULT_SENTINEL)]
        if len(lines) != 1:
            raise RuntimeError("patched sparse worker emitted no unique result sentinel")
        payload = json.loads(lines[0].removeprefix(RESULT_SENTINEL))
        if set(payload) != {"max_diff", "all_supports_match"}:
            raise RuntimeError(f"patched sparse worker returned unexpected fields: {payload}")
        return {
            "max_diff": float(payload["max_diff"]),
            "all_supports_match": bool(payload["all_supports_match"]),
        }


def main() -> None:
    if sys.argv[1:] == [PATCHED_WORKER_ARGUMENT]:
        print(RESULT_SENTINEL + json.dumps(_worker_result(), allow_nan=False, sort_keys=True))
        return
    print("=== Checking unpatched failure modes ===")
    failures = check_unpatched_failure_modes()
    for f in failures:
        print(f"Site: {f['site']}")
        print(f"  Op: {f['operation']}")
        print(f"  Error: {f['exception']}")

    print("\n=== Running Equivalence Probe ===")
    results = run_equivalence_probe()
    print(f"Max posterior difference (dense vs sparse): {results['max_diff']:.3e}")
    print(f"All prefix supports match trie nodes: {results['all_supports_match']}")

    if results["max_diff"] >= 1e-6:
        raise SystemExit(f"sparse/dense posteriors diverged: {results['max_diff']:.3e}")
    if not results["all_supports_match"]:
        raise SystemExit("sparse and dense supports diverged")
    print("\nSUCCESS: pymdp sparse capability confirmed and verified.")


if __name__ == "__main__":
    main()
