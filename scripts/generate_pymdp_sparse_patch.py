#!/usr/bin/env python3
"""Generate the candidate upstream pymdp patch from the installed source.

## Why this exists

The bundle under ``output/upstream_pymdp_pr/`` was hand-built and untracked.
``output/`` promises to be entirely regenerable, and this was the one thing in it
that was not: deleting the tree to prove the project rebuilds destroyed an
artifact no script could recreate, and only a copy in a scratch directory saved
it. An irreplaceable file living in the regenerable tree is protected by nothing.

So the patch is now derived. The three hunks are expressed as transformations of
the installed ``pymdp`` source rather than as stored context, which also fixes a
second problem: the previous patch's context had been written against a source
that does not exist, inventing an ``if tensor_name is None:`` block in
``validate_normalization`` and a loop body unlike the real
``compute_expected_state``. It failed ``git apply --check`` on both hunks while
its gate passed in 0.01 seconds. A patch generated from the source it targets
cannot have invented context.

## What the patch does

Each hunk is a place a sparse array is rejected, not a place the mathematics
fails. Decision 58 records the boundary and
``scripts/probe_pymdp_sparse.py`` measures the equivalence.

Usage:
    .venv/bin/python scripts/generate_pymdp_sparse_patch.py
    .venv/bin/python scripts/generate_pymdp_sparse_patch.py --check
"""

from __future__ import annotations

import argparse
import difflib
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _artifact_io import (  # noqa: E402
    AuthenticatedFileError,
    atomic_write_text,
    read_authenticated_regular_file,
)

BUNDLE_DIR = PROJECT_ROOT / "output" / "upstream_pymdp_pr"
PATCH_NAME = "0001-feat-sparse-B-support-in-Agent-and-FPI.patch"
DESCRIPTION_NAME = "PR_DESCRIPTION.md"


def _pymdp_root() -> Path:
    import pymdp

    return Path(pymdp.__file__).resolve().parent


#: One edit per blocking site: (file, exact text to find, replacement).
#: Written as exact-match rewrites so a pymdp release that changes any of these
#: lines fails loudly here rather than producing a patch that will not apply.
HUNKS: tuple[tuple[str, str, str], ...] = (
    (
        "utils.py",
        "    sums = jnp.sum(tensor, axis=axis)\n",
        "    # A BCOO tensor cannot be summed by `jnp.sum`, which raises rather than\n"
        "    # dispatching. `sparse.sparsify` lifts the reduction to the sparse array;\n"
        "    # the result is dense and small (one entry per distribution), so nothing\n"
        "    # downstream needs to change.\n"
        "    if isinstance(tensor, sparse.BCOO):\n"
        "        sums = sparse.sparsify(jnp.sum)(tensor, axis=axis).todense()\n"
        "    else:\n"
        "        sums = jnp.sum(tensor, axis=axis)\n",
    ),
    (
        "utils.py",
        "import equinox as eqx\n",
        "import equinox as eqx\nfrom jax.experimental import sparse\n",
    ),
    (
        "control.py",
        "        qs_next_f = factor_dot(B_f[...,u_f], relevant_factors, keep_dims=(0,))\n"
        "        qs_next.append(qs_next_f)\n",
        "        qs_next_f = factor_dot(B_f[...,u_f], relevant_factors, keep_dims=(0,))\n"
        "        # Contracting against a sparse B returns a BCOO marginal. A belief is a\n"
        "        # dense 1-D vector everywhere downstream, and feeding a BCOO back as the\n"
        "        # next prior fails dispatch in `factor_dot`, so densify here: the array\n"
        "        # is one factor wide and its sparsity buys nothing.\n"
        "        qs_next.append(maths._to_dense_if_sparse(qs_next_f))\n",
    ),
    (
        "control.py",
        "from jaxtyping import Array\n",
        "from jaxtyping import Array\nfrom pymdp import maths\n",
    ),
    (
        "agent.py",
        "jnp.broadcast_to(b_f, (batch_size,) + b_f.shape)",
        "b_f if isinstance(b_f, sparse.BCOO) else jnp.broadcast_to(b_f, (batch_size,) + b_f.shape)",
    ),
    (
        "agent.py",
        "import jax.numpy as jnp\n",
        "import jax.numpy as jnp\nfrom jax.experimental import sparse\n",
    ),
)


def build_patch() -> str:
    """Produce the unified diff against the installed pymdp.

    Returns:
        The patch text, with ``a/pymdp/...`` and ``b/pymdp/...`` headers and no
        timestamps, so the output is byte-stable across runs.

    Raises:
        SystemExit: If any hunk's target text is absent from the installed
            source. That means the release moved the line, and a patch built on
            a guess would not apply — the failure this script exists to prevent.
    """
    root = _pymdp_root()
    edited: dict[str, str] = {}
    for filename, find, replace in HUNKS:
        source = edited.get(filename)
        if source is None:
            source = (root / filename).read_text(encoding="utf-8")
        if find not in source:
            raise SystemExit(
                f"ERROR: pymdp/{filename} no longer contains the text this hunk "
                f"targets:\n  {find.strip()[:100]}\n"
                f"The installed release moved it. Re-derive the hunk against the "
                f"current source rather than shipping a patch that cannot apply."
            )
        edited[filename] = source.replace(find, replace, 1)

    chunks: list[str] = []
    for filename in dict.fromkeys(name for name, _, _ in HUNKS):
        original = (root / filename).read_text(encoding="utf-8").splitlines(keepends=True)
        modified = edited[filename].splitlines(keepends=True)
        chunks.extend(
            difflib.unified_diff(
                original,
                modified,
                fromfile=f"a/pymdp/{filename}",
                tofile=f"b/pymdp/{filename}",
                n=3,
            )
        )
    return "".join(chunks)


def build_description(measured: dict[str, object]) -> str:
    """The bundle's prose, with its measured figures injected.

    The description was hand-written and untracked alongside the patch, so it
    died with it. Its two load-bearing numbers come from the probe rather than
    from anyone's memory of what the probe said.

    Args:
        measured: The probe's own result mapping.

    Returns:
        Markdown for ``PR_DESCRIPTION.md``.
    """
    return f"""# Sparse `B` through the public `Agent` API

## What this is

A candidate upstream patch for `pymdp`'s JAX backend, generated by
`scripts/generate_pymdp_sparse_patch.py` and **not submitted**. It is evidence
for decision 58, which records that pymdp's sparse contraction machinery already
exists and that the boundary blocking it is the `Agent` API.

Regenerate and re-verify with:

```bash
.venv/bin/python scripts/generate_pymdp_sparse_patch.py
.venv/bin/python -m pytest tests/test_upstream_pymdp_bundle.py -q
```

The patch is derived from the installed source rather than stored with its own
context. An earlier hand-written version failed `git apply --check` on both
hunks while its test passed in 0.01 s, because that test checked only file size
and two substrings, and the context had been written against a source that does
not exist.

## The three sites

Each is a place a sparse array is rejected, not a place the mathematics fails.

| File | What blocks | Change |
|---|---|---|
| `pymdp/utils.py` | `jnp.sum` raises on `BCOO` in `validate_normalization` | sum through `sparse.sparsify` and densify the small result |
| `pymdp/control.py` | `compute_expected_state` returns a `BCOO` marginal that cannot be fed back as the next prior | densify the marginal, which is one factor wide |
| `pymdp/agent.py` | `jnp.broadcast_to` rejects `BCOO` outright | skip the broadcast when the array is already batched |

## What it achieves, measured

Applied to the installed pymdp, with **no runtime monkeypatches**, a
sparse-transition Stage 3 agent constructs, filters, and agrees with the dense
agent exactly:

```
max |dense - sparse| over six words: {measured["max_diff"]:.3e}
induced trie-node supports identical: {measured["all_supports_match"]}
```

## What it does not achieve

**The caller still has work to do, and the patch cannot do it.** The leading
batch dimension must be a genuine `BCOO` batch dimension (`n_batch=1`), not a
fourth sparse dimension. pymdp `vmap`s over axis 0 and refuses a `BCOO` with
`n_batch=0`. A caller who builds the obvious 4-D sparse array gets an agent that
constructs and then fails at the first inference step. That belongs in
documentation, not in these three hunks.

**`A` has no sparse path and should not get one.**
`compute_log_likelihood_single_modality` calls `jnp.clip(A_m, min=MINVAL)`, and
densification there is not an oversight: the log of a structural zero is not
zero, so a log-likelihood tensor has no structural zeros to exploit.

**Not adopted here.** Decision 58 records sparse `B` as considered, measured,
and declined for the shipped level: it saves 16.0 MB of a 19.5 MB agent that
nothing is memory-constrained by.
"""


def verify(patch_text: str) -> None:
    """Apply the patch to a throwaway copy and refuse to ship one that fails.

    A generated patch that does not apply is the same defect as a hand-written
    one that does not apply; generating it only removes the *cause*, not the
    need to check.

    Raises:
        SystemExit: If ``git apply --check`` rejects the patch.
    """
    import subprocess

    with tempfile.TemporaryDirectory() as tmp:
        # macOS returns /var/... here, where /var is a symlink to /private/var.
        # The descriptor-anchored artifact writer deliberately refuses symlink
        # path components, so resolve the OS-owned temporary directory before
        # asking it to publish the candidate patch.  This preserves the writer's
        # no-follow contract instead of weakening it for a platform alias.
        staged = Path(tmp).resolve(strict=True)
        shutil.copytree(_pymdp_root(), staged / "pymdp")
        subprocess.run(["git", "init", "-q", "."], cwd=staged, check=True)
        candidate = staged / "candidate.patch"
        atomic_write_text(candidate, patch_text)
        result = subprocess.run(
            ["git", "apply", "--check", "-p1", str(candidate)],
            cwd=staged,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SystemExit(f"ERROR: the generated patch does not apply:\n{result.stderr}")


def build_bundle() -> dict[str, str]:
    """Render and validate every file in the upstream-candidate bundle."""
    patch_text = build_patch()
    verify(patch_text)

    # The description's measured figures come from the probe, not from memory.
    import probe_pymdp_sparse

    measured = probe_pymdp_sparse.run_equivalence_probe(patch_text)
    return {
        PATCH_NAME: patch_text,
        DESCRIPTION_NAME: build_description(measured),
    }


def stale_bundle_entries(bundle_dir: Path, expected: dict[str, str]) -> list[str]:
    """Return missing, unsafe, or byte-stale bundle entries.

    Text decoding would make the check a comparison of interpretations rather
    than of the bytes publication actually ships.  Read each regular file from
    one authenticated descriptor and compare it with the generator's UTF-8
    bytes exactly.
    """
    stale: list[str] = []
    for name, text in expected.items():
        target = bundle_dir / name
        try:
            actual = read_authenticated_regular_file(
                target,
                description=f"generated upstream bundle entry {name}",
                max_bytes=1024 * 1024,
            ).content
        except (AuthenticatedFileError, FileNotFoundError, OSError):
            stale.append(name)
            continue
        if actual != text.encode("utf-8"):
            stale.append(name)
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the bundle on disk matches what this script generates, without writing",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=BUNDLE_DIR,
        help="alternate bundle directory for isolated generation or checking",
    )
    args = parser.parse_args()

    bundle_dir = args.bundle_dir.expanduser().resolve()
    rendered = build_bundle()
    target = bundle_dir / PATCH_NAME
    if args.check:
        stale = stale_bundle_entries(bundle_dir, rendered)
        if stale:
            print(
                "the upstream bundle differs from what this script generates "
                f"({', '.join(stale)}); "
                "re-run scripts/generate_pymdp_sparse_patch.py",
                file=sys.stderr,
            )
            return 1
        print("bundled patch and PR description are current; the patch applies cleanly.")
        return 0

    for name, text in rendered.items():
        atomic_write_text(bundle_dir / name, text)

    try:
        displayed_target = target.relative_to(PROJECT_ROOT)
    except ValueError:
        displayed_target = target
    print(f"wrote {displayed_target} ({len(rendered[PATCH_NAME].splitlines())} lines)")
    print(f"wrote {bundle_dir / DESCRIPTION_NAME}")
    print("verified: git apply --check succeeds against the installed pymdp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
