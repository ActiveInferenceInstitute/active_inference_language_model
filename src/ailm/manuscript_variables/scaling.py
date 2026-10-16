"""Stage-3 scaling-ladder producer."""

from __future__ import annotations

import math
import re
from typing import (
    Any,
    Mapping,
)

from ailm.statistics import (
    CORRECTED_CONFIDENCE,
    REFERENCE_SHARE,
    minimum_discordant_pairs_for_equivalence,
)
from ailm.value_formatting import (
    ManuscriptVariableError,
)
from ailm.value_formatting import (
    format_ledger_value as _format_value,
)

from ._claims import (
    _ClaimReaders,
    require_close,
)
from ._io import _StrictVariableMap
from ._stats import _whole


def _produce_stage3_scaling_ladder(
    variables: _StrictVariableMap, claims: Mapping[str, Any], readers: _ClaimReaders
) -> None:
    """The Stage 3 scaling ladder: the dynamic claim family bound to the
    code-owned rung registry, every recorded identity replayed, and the span
    and equivalence-minima tokens derived.
    """
    num = readers.num

    # The scaling paragraph makes claims about the *whole ladder*, not only its
    # endpoints. Bind the dynamic claim family to the code-owned rung registry,
    # require every field at every rung, and replay every recorded identity.
    # This also makes deleting the complete non-REQUIRED family fail instead of
    # skipping the reconciliation block.
    from ailm.levels.word import SCALING_RUNGS

    _scale_fields = frozenset(
        {
            "coverage_pct",
            "word_perplexity",
            "bigram_baseline",
            "eval_positions",
            "gain_nats",
            "gain_shared",
        }
    )
    _scale_family: dict[int, set[str]] = {}
    for _label in claims:
        _match = re.fullmatch(r"stage3_scale_(\d+)_(.+)", _label)
        if _match is None:
            continue
        _size, _field = int(_match.group(1)), _match.group(2)
        if _field not in _scale_fields:
            raise ManuscriptVariableError(
                f"unknown Stage 3 scaling field {_field!r} in claim {_label!r}"
            )
        _scale_family.setdefault(_size, set()).add(_field)

    _full_vocab_size = _whole("stage3_full_vocab_size", num("stage3_full_vocab_size"))
    _expected_rung_sizes = tuple(
        sorted({size for size in SCALING_RUNGS if size < _full_vocab_size} | {_full_vocab_size})
    )
    _rung_sizes = tuple(sorted(_scale_family))
    if _rung_sizes != _expected_rung_sizes:
        raise ManuscriptVariableError(
            "Stage 3 scaling claim coordinates disagree with SCALING_RUNGS and "
            f"the recorded full vocabulary: found {_rung_sizes}, expected {_expected_rung_sizes}"
        )
    if len(_rung_sizes) < 2:
        raise ManuscriptVariableError("the Stage 3 scaling ladder requires at least two rungs")
    for _size in _rung_sizes:
        _missing_fields = _scale_fields - _scale_family[_size]
        if _missing_fields:
            raise ManuscriptVariableError(
                f"Stage 3 scaling rung {_size} is incomplete; missing {sorted(_missing_fields)}"
            )

    # Only the span is derived. The rung COUNT is recorded by the Stage 3
    # recorder and reconciled rather than replaced by the derived count.
    _recorded_rung_count = _whole("stage3_scale_rungs", num("stage3_scale_rungs"))
    if _recorded_rung_count != len(_rung_sizes):
        raise ManuscriptVariableError(
            "stage3_scale_rungs is inconsistent with the complete recorded "
            f"family: recorded {_recorded_rung_count}, found {len(_rung_sizes)}"
        )

    _top_positions = _whole(
        "top Stage 3 scale eval positions",
        num(f"stage3_scale_{_rung_sizes[-1]}_eval_positions"),
    )
    if _top_positions <= 0:
        raise ManuscriptVariableError("the top Stage 3 scaling rung must score positions")

    _scale_rows: list[dict[str, float]] = []
    for _size in _rung_sizes:
        _coverage = num(f"stage3_scale_{_size}_coverage_pct")
        _perplexity = num(f"stage3_scale_{_size}_word_perplexity")
        _baseline = num(f"stage3_scale_{_size}_bigram_baseline")
        _positions = _whole(
            f"stage3_scale_{_size}_eval_positions",
            num(f"stage3_scale_{_size}_eval_positions"),
        )
        _gain = num(f"stage3_scale_{_size}_gain_nats")
        _shared = num(f"stage3_scale_{_size}_gain_shared")
        if _positions <= 0 or _baseline <= 0.0 or _perplexity <= 0.0:
            raise ManuscriptVariableError(
                f"Stage 3 scaling rung {_size} must have positive positions and perplexities"
            )
        require_close(
            f"stage3_scale_{_size}_gain_nats",
            _gain,
            round(math.log(_baseline / _perplexity), 4),
        )
        require_close(
            f"stage3_scale_{_size}_gain_shared",
            _shared,
            round(_gain * _positions / _top_positions, 4),
        )
        _scale_rows.append(
            {
                "coverage": _coverage,
                "perplexity": _perplexity,
                "baseline": _baseline,
                "positions": float(_positions),
                "shared": _shared,
            }
        )

    for _field in ("coverage", "perplexity", "baseline", "positions", "shared"):
        _values = [row[_field] for row in _scale_rows]
        if any(
            current <= previous for previous, current in zip(_values[:-1], _values[1:], strict=True)
        ):
            raise ManuscriptVariableError(
                f"Stage 3 scaling {_field} must rise strictly across the complete ladder; "
                f"got {_values}"
            )

    _last_size = _rung_sizes[-1]
    for _full_label, _scale_field in (
        ("stage3_full_coverage_pct", "coverage_pct"),
        ("stage3_full_word_perplexity", "word_perplexity"),
        ("stage3_full_bigram_baseline", "bigram_baseline"),
        ("stage3_full_eval_positions", "eval_positions"),
        ("stage3_full_gain_nats", "gain_nats"),
        ("stage3_full_gain_shared", "gain_shared"),
    ):
        require_close(
            _full_label,
            num(_full_label),
            num(f"stage3_scale_{_last_size}_{_scale_field}"),
        )
    require_close(
        "stage3_full_excluded_pct",
        num("stage3_full_excluded_pct"),
        round(100.0 - num("stage3_full_coverage_pct"), 2),
    )
    variables["STAGE3_SCALE_SPAN_FACTOR"] = f"{max(_rung_sizes) / min(_rung_sizes):.0f}"
    _lo = num(f"stage3_scale_{_rung_sizes[0]}_gain_shared")
    _hi = num(f"stage3_scale_{_rung_sizes[-1]}_gain_shared")
    if _lo <= 0.0:
        raise ManuscriptVariableError(
            "the smallest Stage 3 shared-denominator gain must be positive "
            f"to define a relative rise, got {_lo!r}"
        )
    _scale_gain_rise_pct = 100.0 * (_hi - _lo) / _lo
    # The recorder owns this manuscript token. Recompute its defining identity
    # at the recorder's two-decimal precision without adding a second producer.
    require_close(
        "stage3_scale_gain_rise_pct",
        num("stage3_scale_gain_rise_pct"),
        round(_scale_gain_rise_pct, 2),
    )
    variables["EQUIVALENCE_MIN_DISCORDANT_PAIRS"] = str(minimum_discordant_pairs_for_equivalence())
    variables["EQUIVALENCE_MIN_DISCORDANT_PAIRS_CORRECTED"] = str(
        minimum_discordant_pairs_for_equivalence(confidence=CORRECTED_CONFIDENCE)
    )
    variables["EQUIVALENCE_REFERENCE_SHARE"] = _format_value(REFERENCE_SHARE)
