"""AILM — Active Inference Language Model.

Discrete language models from letter through document position. Every modeled
level has an explicit, inspectable A/B/C/D structure and a live pymdp Agent
builder, but the shipped comprehension scorers run levels independently and the
measured top-down seams are not one connected cascade.
No transformers, no learned embeddings, no neural networks.

Each level now has a falsifiable recorded gate with a baseline and negative
control. Several historical gates were repaired after their levels had already
been built, so this is not a claim of prospective sequential validation.

Grounding deposits (record identity and printed PDF identity are not collapsed):
  CEREBRUM — concept DOI 10.5281/zenodo.15170907; record version 1.4,
      while CEREBRUM_1-4_April-16-2025.pdf prints version 1.2. Use the complete
      deposited PDF, not the truncated full_text.md export.
  Compositional Approaches to Linguistic Case for Cognitive Modeling — concept
      DOI 10.5281/zenodo.19695259; the deposited file is titled Cognitive
      Diagrams: Reviewing Categorical Accounts of Linguistic Case. Exact extents,
      digests, and the optional companion revision: data/reference_sources.json.

Design decisions in docs/design/. What the papers do and do not provide in README.md.
"""

__version__ = "1.0.0"

import warnings as _warnings

# pymdp 1.0.3 emitted this UserWarning from every Agent() construction: it stored
# the JAX arrays passed as A/B/C/D on an equinox module field declared static.
# Verified against installed 1.0.4 (2026-08-29): Agent(A=..., B=..., C=..., D=...)
# with real jnp arrays raises no warnings, because Policies stores a hashable
# nested tuple statically and materializes policy_arr lazily. There are fourteen
# construction sites across the levels.
#
# pyproject.toml carries the identical filter, but that one applies only under
# pytest — so `scripts/generate_figures.py` and every other entry point printed
# the warning twice per run while the test suite was silent about it. Filtering
# here makes the two agree instead of depending on how the code was entered.
#
# Scoped to the exact message, never to the category: any OTHER UserWarning from
# pymdp or anywhere else still surfaces. Widening this to a bare
# `ignore::UserWarning` would hide the next real one.
_warnings.filterwarnings(
    "ignore",
    message="A JAX array is being set as static!",
    category=UserWarning,
)
