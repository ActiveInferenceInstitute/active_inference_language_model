"""Manuscript ``{{TOKEN}}`` generation for AILM.

Every quantitative statement in the manuscript is a ``{{TOKEN}}`` resolved from
``data/claims.json``. That is the whole point: the claims ledger is already the
project's single source of numeric truth (``ailm.claims``), so wiring the
manuscript to it makes a hand-typed number in prose *impossible* rather than
merely discouraged. A token with no ledger entry fails the render.

Ledger-token naming is mechanical — the ledger label uppercased:

    claim  stage1_letter_perplexity   ->   token  {{STAGE1_LETTER_PERPLEXITY}}

so there is no second mapping table to drift out of sync with the ledger. Other
tokens are typed, reviewable products of the literal registry, code-owned model
and channel registries, paper metadata, vendored-artifact provenance, or explicit
derivations from recorded claims. None is an unowned fallback.

Reads:
- ``data/claims.json``        — every recorded measurement (via ``ailm.claims``)
- ``data/literature_search.json`` — bounded priority-search protocol and disposition
- ``data/manuscript_literals.yaml`` — declared decision-log metadata
- ``data/reference_sources.json`` — deposited-source identity, extents, and locators
- ``docs/manuscript/config.yaml``  — paper metadata (title, authors, keywords)
- ``vendor/*.json``           — corpus artifact sizes and checksums

Returns a flat ``dict[str, str]`` consumed by
``scripts/z_generate_manuscript_variables.py``, which the renderer invokes
automatically before assembling the manuscript.

``src/`` must stay standalone — no ``infrastructure`` import (enforced by
``tests/test_figure_registry.py::test_nothing_under_src_imports_infrastructure``;
the template's drift gate carries the same check but only warns for this
project), so ``SOURCE_DATE_EPOCH`` is read directly for reproducible build
stamps.

The rationale for the recorder/verifier split, and for refusing to fall back to a
default value, is in ``docs/decisions.md``.
"""

from __future__ import annotations

from ailm.required_claims import REQUIRED_CLAIMS
from ailm.value_formatting import ManuscriptVariableError as ManuscriptVariableError
from ailm.value_formatting import format_ledger_value as _format_value

from ._channels import (
    LEVEL_QUALIFYING_VERDICT_FLAGS,
    canonical_stage5_passes,
    classify_cross_level_channels,
    level_six_qualification_limits,
    qualified_verdict_flags,
)
from ._claims import (
    _BENCHMARK_SIGNED_UNIT_INTERVAL_RE,
    _BENCHMARK_UNIT_INTERVAL_RE,
    _BINARY_LABEL_RE,
    _COUNT_LABEL_RE,
    _NONNEGATIVE_REAL_RE,
    _NONNEGATIVE_RELATIVE_PERCENTAGE_CLAIMS,
    _P_VALUE_LABEL_RE,
    _PERCENT_SCALE_STAGE4_RE,
    _POSITIVE_REAL_RE,
    _REQUIRED_BOOLEAN_CLAIMS,
    _REQUIRED_STRING_CLAIMS,
    _REQUIRED_STRING_LIST_CLAIMS,
    _SIGNED_RELATIVE_PERCENTAGE_CLAIMS,
    _SIGNED_UNIT_INTERVAL_CLAIMS,
    _UNIT_INTERVAL_COMPONENT_RE,
    _UNIT_INTERVAL_EXACT_CLAIMS,
    NumericClaimDomain,
    _claim_number,
    _ClaimReaders,
    _ContextReconciliation,
    _LexicalSweep,
    _numeric_claim_domain,
    _SyntheticSweep,
    _validate_numeric_claim_domain,
    _validate_required_claim_values,
    _validate_synthetic_interval,
    require_close,
)
from ._core import (
    _produce_build_provenance,
    _produce_counted_claims,
    _produce_derived_quantities,
    _produce_ledger_tokens,
    _produce_paper_metadata,
    generate_manuscript_variables,
)
from ._exceptions import (
    DuplicateInputKeyError,
    DuplicateManuscriptTokenError,
    InvalidLiteralVariableError,
    InvalidLiteratureSearchError,
    InvalidReferenceSourceError,
    MissingClaimError,
)
from ._formatting import (
    _COUNT_WORDS,
    _FAMILY_MEMBER_PHRASES,
    _ORDINAL_WORDS,
    _build_timestamp,
    _count_word,
    _join_prose,
    _ordinal_word,
    _quantified_clause,
    _separation_text,
)
from ._io import (
    DIGEST_SHORT_LENGTH,
    _construct_unique_yaml_mapping,
    _load_unique_yaml,
    _StrictVariableMap,
    _unique_json_object,
    _UniqueKeyLoader,
)
from ._stats import (
    AUTHORED_MANUSCRIPT_PROBE_KEYS,
    DERIVED_BOUND_ARMS,
    DERIVED_POWER_ARM_PREFIXES,
    PARAMETER_MODEL_DISPLAY_NAMES,
    SUBSAMPLE_BOUND_ARMS,
    _integer_success_count,
    _require_paired_table,
    _whole,
    paired_difference_bound_from_counts,
    paired_power_from_counts,
)
from .architecture import (
    _produce_architecture_tallies,
    _produce_calibration_rows,
    _produce_fitted_and_stage5,
    _produce_lexical_sweep,
)
from .context import (
    _produce_context_protocol,
    _produce_context_results,
)
from .literature import (
    _DECISION_LITERAL_NAMES,
    _ISO_DATE_RE,
    _LITERAL_VARIABLE_NAMES,
    _MONTH_YEAR_RE,
    _decision_ids,
    _load_claims,
    _load_config,
    _load_literal_variables,
    _load_literature_variables,
    _load_reference_variables,
    _validate_literal_value,
    _vendor_metadata,
)
from .null_bounds import (
    _produce_derived_arm_power,
    _produce_null_bound_conventions,
    _produce_null_bounds,
)
from .scaling import _produce_stage3_scaling_ladder
from .synthetic import (
    ALIGNED_G2P_COMPARATOR_ARMS,
    _g2p_bucket_variables,
    _produce_aligned_g2p_comparator,
    _produce_synthetic_context_results,
    _produce_synthetic_phone_results,
    _produce_synthetic_results,
)

__all__ = [
    "ALIGNED_G2P_COMPARATOR_ARMS",
    "AUTHORED_MANUSCRIPT_PROBE_KEYS",
    "DERIVED_BOUND_ARMS",
    "DERIVED_POWER_ARM_PREFIXES",
    "DIGEST_SHORT_LENGTH",
    "DuplicateInputKeyError",
    "DuplicateManuscriptTokenError",
    "InvalidLiteralVariableError",
    "InvalidLiteratureSearchError",
    "InvalidReferenceSourceError",
    "LEVEL_QUALIFYING_VERDICT_FLAGS",
    "ManuscriptVariableError",
    "MissingClaimError",
    "NumericClaimDomain",
    "PARAMETER_MODEL_DISPLAY_NAMES",
    "REQUIRED_CLAIMS",
    "SUBSAMPLE_BOUND_ARMS",
    "_BENCHMARK_SIGNED_UNIT_INTERVAL_RE",
    "_BENCHMARK_UNIT_INTERVAL_RE",
    "_BINARY_LABEL_RE",
    "_COUNT_LABEL_RE",
    "_COUNT_WORDS",
    "_ClaimReaders",
    "_ContextReconciliation",
    "_DECISION_LITERAL_NAMES",
    "_FAMILY_MEMBER_PHRASES",
    "_ISO_DATE_RE",
    "_LITERAL_VARIABLE_NAMES",
    "_LexicalSweep",
    "_MONTH_YEAR_RE",
    "_NONNEGATIVE_REAL_RE",
    "_NONNEGATIVE_RELATIVE_PERCENTAGE_CLAIMS",
    "_ORDINAL_WORDS",
    "_PERCENT_SCALE_STAGE4_RE",
    "_POSITIVE_REAL_RE",
    "_P_VALUE_LABEL_RE",
    "_REQUIRED_BOOLEAN_CLAIMS",
    "_REQUIRED_STRING_CLAIMS",
    "_REQUIRED_STRING_LIST_CLAIMS",
    "_SIGNED_RELATIVE_PERCENTAGE_CLAIMS",
    "_SIGNED_UNIT_INTERVAL_CLAIMS",
    "_StrictVariableMap",
    "_SyntheticSweep",
    "_UNIT_INTERVAL_COMPONENT_RE",
    "_UNIT_INTERVAL_EXACT_CLAIMS",
    "_UniqueKeyLoader",
    "_build_timestamp",
    "_claim_number",
    "_construct_unique_yaml_mapping",
    "_count_word",
    "_decision_ids",
    "_format_value",
    "_g2p_bucket_variables",
    "_integer_success_count",
    "_join_prose",
    "_load_claims",
    "_load_config",
    "_load_literal_variables",
    "_load_literature_variables",
    "_load_reference_variables",
    "_load_unique_yaml",
    "_numeric_claim_domain",
    "_ordinal_word",
    "_produce_aligned_g2p_comparator",
    "_produce_architecture_tallies",
    "_produce_build_provenance",
    "_produce_calibration_rows",
    "_produce_context_protocol",
    "_produce_context_results",
    "_produce_counted_claims",
    "_produce_derived_arm_power",
    "_produce_derived_quantities",
    "_produce_fitted_and_stage5",
    "_produce_ledger_tokens",
    "_produce_lexical_sweep",
    "_produce_null_bound_conventions",
    "_produce_null_bounds",
    "_produce_paper_metadata",
    "_produce_stage3_scaling_ladder",
    "_produce_synthetic_context_results",
    "_produce_synthetic_phone_results",
    "_produce_synthetic_results",
    "_quantified_clause",
    "_require_paired_table",
    "_separation_text",
    "_unique_json_object",
    "_validate_literal_value",
    "_validate_numeric_claim_domain",
    "_validate_required_claim_values",
    "_validate_synthetic_interval",
    "_vendor_metadata",
    "_whole",
    "canonical_stage5_passes",
    "classify_cross_level_channels",
    "generate_manuscript_variables",
    "level_six_qualification_limits",
    "paired_difference_bound_from_counts",
    "paired_power_from_counts",
    "qualified_verdict_flags",
    "require_close",
]
