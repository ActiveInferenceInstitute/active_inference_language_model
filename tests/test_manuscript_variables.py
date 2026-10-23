"""Tests for the manuscript ``{{TOKEN}}`` generator.

The token map is the mechanism that makes a hand-typed number in the manuscript
structurally impossible: every quantitative statement resolves from
``data/claims.json``. That guarantee only holds if this module actually fails when
a required claim is missing, so the failure path is tested as carefully as the
success path.

No mocks: these tests use the real ledger and real temp files.
"""

from __future__ import annotations

import importlib
import json
import platform
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

from ailm.benchmarks.constants import SERVED_PROBES
from ailm.benchmarks.synthetic_protocol import SYNTHETIC_INCONTEXT_PROTOCOL
from ailm.manuscript_variables import (
    DERIVED_POWER_ARM_PREFIXES,
    DIGEST_SHORT_LENGTH,
    REQUIRED_CLAIMS,
    DuplicateInputKeyError,
    DuplicateManuscriptTokenError,
    InvalidLiteralVariableError,
    InvalidLiteratureSearchError,
    InvalidReferenceSourceError,
    ManuscriptVariableError,
    MissingClaimError,
    NumericClaimDomain,
    _format_value,
    _g2p_bucket_variables,
    _load_literature_variables,
    _load_reference_variables,
    _numeric_claim_domain,
    _require_paired_table,
    _validate_required_claim_values,
    _vendor_metadata,
    generate_manuscript_variables,
    paired_power_from_counts,
)
from ailm.statistics import (
    SUBSAMPLE_LARGER_ARM,
    SUBSAMPLE_LARGER_STRIDE,
    SUBSAMPLE_MATCHED_ARM,
    SUBSAMPLE_MATCHED_STRIDE,
    SUBSAMPLE_STRIDE_BY_ARM,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEDGER = PROJECT_ROOT / "data" / "claims.json"

_SKIP_NO_LEDGER = pytest.mark.skipif(
    not LEDGER.exists(),
    reason="data/claims.json absent; run the scripts/record_stage*_claims.py recorders",
)

#: The external-benchmark probes every arm was run against.
_BENCH_MODELS = tuple(probe.key for probe in SERVED_PROBES)

#: Arm prefixes whose power the recorder wrote into the ledger as
#: ``<prefix>_power_<model>``. These are the control group for the equivalence
#: proof below: the derived path must land on these recorded numbers exactly.
_RECORDED_POWER_ARMS = ("bench_letter", "bench_phoneme", "bench_g2p")


def test_paired_table_realizability_accepts_a_complete_binary_table():
    _require_paired_table(
        "control",
        first_successes=7,
        second_successes=5,
        b=3,
        c=1,
        total=10,
    )


@pytest.mark.parametrize(
    ("first", "second", "b", "c", "message"),
    (
        (30, 30, 100, 100, "negative concordant"),
        (400, 400, 1, 1, "negative concordant"),
        (30, 31, 2, 2, "incompatible paired marginals"),
    ),
)
def test_paired_table_realizability_rejects_impossible_concordant_cells(
    first, second, b, c, message
):
    with pytest.raises(ManuscriptVariableError, match=message):
        _require_paired_table(
            "mutated",
            first_successes=first,
            second_successes=second,
            b=b,
            c=c,
            total=400,
        )


def _real_claims() -> dict:
    """The shipped ledger, as a label -> entry mapping."""
    return json.loads(LEDGER.read_text(encoding="utf-8"))["claims"]


def _valid_config() -> dict:
    """Smallest configuration accepted by the production metadata contract."""
    return {
        "paper": {"title": "Test manuscript", "version": "1.0.0"},
        "authors": [{"name": "Test Author"}],
        "keywords": ["testing"],
        "rendering": {
            "body_font_pt": 9.5,
            "body_baseline_pt": 11.4,
            "toc_font_pt": 9.5,
            "toc_baseline_pt": 11.5,
            "margin_top_in": 0.45,
            "margin_bottom_in": 0.55,
            "margin_left_in": 0.55,
            "margin_right_in": 0.55,
        },
    }


def _write_project(tmp_path, claims: dict, config: dict | None = None):
    """Build a minimal project tree with the given ledger and config."""
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "claims.json").write_text(
        json.dumps({"_schema": "ailm-claims-v2", "claims": claims}), encoding="utf-8"
    )
    (tmp_path / "docs" / "manuscript").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "manuscript" / "config.yaml").write_text(
        yaml.safe_dump(_valid_config() if config is None else config), encoding="utf-8"
    )
    (tmp_path / "data" / "manuscript_literals.yaml").write_bytes(
        (PROJECT_ROOT / "data" / "manuscript_literals.yaml").read_bytes()
    )
    (tmp_path / "data" / "reference_sources.json").write_bytes(
        (PROJECT_ROOT / "data" / "reference_sources.json").read_bytes()
    )
    (tmp_path / "data" / "literature_search.json").write_bytes(
        (PROJECT_ROOT / "data" / "literature_search.json").read_bytes()
    )
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "decisions.md").write_text(
        "# Fixture decisions\n\n"
        "## 22 — fixture\n\n"
        "## 25 — fixture\n\n"
        "## 27 — fixture\n\n"
        "## 28 — fixture\n\n"
        "## 29 — fixture\n\n"
        "## 77 — fixture\n\n"
        "## 79 — fixture\n\n"
        "## 83 — fixture\n\n"
        "## 85 — fixture\n\n"
        "## 88 — fixture\n\n"
        "## 91 — fixture\n\n"
        "## 110 — fixture\n\n"
        "## 121 — fixture\n",
        encoding="utf-8",
    )
    vendor = tmp_path / "vendor"
    vendor.mkdir(parents=True, exist_ok=True)
    for name in (
        "brown_counts.json",
        "cmudict_counts.json",
        "brown_word_counts.json",
        "ud_ewt_case_counts.json",
    ):
        (vendor / name).write_text('{"fixture":true}\n', encoding="utf-8")
    return tmp_path


_OFFLINE_PROMPT_EVAL_TOKENS = 1


@lru_cache(maxsize=1)
def _synthetic_v2_fixture_records() -> tuple[tuple[str, object, str, str], ...]:
    """Build coherent v2 synthetic claims without reading their ledger values.

    The offline probe deterministically returns the first allowed response symbol
    for every request.  Recorder-owned pure helpers then derive every rate,
    interval, discordance table, family size, and protocol claim from those flags.
    This is a structural hydration fixture, never empirical benchmark evidence.
    """

    zero = importlib.import_module("scripts.record_synthetic_benchmark_claims")
    context = importlib.import_module("scripts.record_incontext_benchmark_claims")

    # One block per stipulated arm (letter and phone): each arm reconstructs its
    # own executed population from its protocol, scores AILM's comparator on it,
    # and derives its claim namespace through the recorder's pure helpers with
    # the arm spec passed first, exactly as the recorder's measurement loop does.
    zero_records: list[tuple[str, object, str, str]] = []
    for arm in zero.ARMS:
        arm_positions = arm.held_out_positions(zero.DEFAULT_POSITIONS)
        arm_counts = arm.load_counts()
        arm_ailm = zero.correctness_flags(
            arm.comparator_predictions(arm_counts, arm_positions),
            arm_positions,
        )
        arm_offline_symbol = arm.protocol.response_inventory[0]
        arm_probe = [gold == arm_offline_symbol for _prefix, gold in arm_positions]
        arm_total = len(arm_positions)
        arm_values = zero._base_claim_values(
            arm,
            total=arm_total,
            ailm_flags=arm_ailm,
            maximum_prompt_characters=zero._maximum_prompt_characters(arm, arm_positions),
            maximum_prompt_tokens=_OFFLINE_PROMPT_EVAL_TOKENS,
            cache_hits=0,
            cache_misses=len(zero.MODELS) * arm_total * 2,
        )
        for name in sorted(zero.MODELS):
            arm_values.update(
                zero._probe_claim_values(
                    arm,
                    name=name,
                    reference_flags=arm_probe,
                    short_flags=arm_probe,
                    ailm_flags=arm_ailm,
                    unparsable=0,
                    short_unparsable=0,
                    full_first=(arm_total + 1) // 2,
                    short_first=arm_total // 2,
                )
            )
        zero_records.extend(
            (label, value, arm.protocol.run_id, zero.SOURCE) for label, value in arm_values.items()
        )
    zero_records.extend(
        (label, value, zero.LETTER_ARM.protocol.run_id, zero.SOURCE)
        for label, value in {
            "bench_synth_openrouter_model": zero.OPENROUTER_MODEL,
            "bench_synth_openrouter_provider": "offline-fixture-route",
        }.items()
    )
    zero_values = {label: value for label, value, _run_id, _source in zero_records}
    if len(zero_values) != len(zero_records):
        raise AssertionError("the stipulated arms' claim namespaces overlap")
    offline_symbol = zero.SYNTHETIC_ZERO_SHOT_PROTOCOL.response_inventory[0]

    context_positions = context._scored_positions(context.DEFAULT_POSITIONS)
    context_pool = context._example_pool(context_positions)
    context_counts = context.load_synthetic_counts()
    context_ailm = context.correctness_flags(
        context.synthetic_letter_predictions(context_counts, context_positions),
        context_positions,
    )
    context_probe = [gold == offline_symbol for _prefix, gold in context_positions]
    arm_names = tuple(arm for arm, _k, _deranged in context._arm_specs())
    arms = {arm: list(context_probe) for arm in arm_names}
    unparsable = dict.fromkeys(arm_names, 0)
    context_values: dict[str, object] = {
        "bench_ctx_openrouter_model": context.OPENROUTER_MODEL,
        "bench_ctx_openrouter_provider": "offline-fixture-route",
    }
    for name in sorted(context.MODELS):
        context_values.update(
            context._claim_values(
                name,
                arms,
                unparsable,
                context_ailm,
                len(context_positions),
            )
        )

    mapping_treatment, mapping_deranged = context._mapping_control_flags(
        context_positions,
        context_pool,
    )
    mapping_paired = context.paired_discordance(mapping_treatment, mapping_deranged)
    context_values.update(
        {
            "bench_ctx_mapping_control_top1_treatment": (
                sum(mapping_treatment) / len(context_positions)
            ),
            "bench_ctx_mapping_control_top1_deranged": (
                sum(mapping_deranged) / len(context_positions)
            ),
            "bench_ctx_mapping_control_paired_b": mapping_paired["b"],
            "bench_ctx_mapping_control_paired_c": mapping_paired["c"],
            "bench_ctx_mapping_control_paired_p": mapping_paired["p_value"],
            "bench_ctx_mapping_control_family_size": len(context.MAPPING_CONTROL_FAMILY_TEMPLATES),
            "bench_ctx_mapping_control_family_alpha": context.POWER_ALPHA
            / len(context.MAPPING_CONTROL_FAMILY_TEMPLATES),
            "bench_ctx_maximum_prompt_characters": context._maximum_prompt_characters(
                len(context_positions)
            ),
            "bench_ctx_maximum_prompt_tokens": _OFFLINE_PROMPT_EVAL_TOKENS,
            "bench_ctx_scored_cache_hits": 0,
            "bench_ctx_scored_cache_misses": (
                len(context.MODELS) * len(context_positions) * len(arm_names)
            ),
        }
    )
    context_values.update(context._draw_diagnostic_claim_values(context_positions, context_pool))

    values = {**zero_values, **context_values}
    expected = {
        label
        for label in REQUIRED_CLAIMS
        if label.startswith(("bench_synth_", "bench_synthphone_", "bench_ctx_"))
    }
    if set(values) != expected:
        raise AssertionError(
            "offline synthetic v2 fixture does not exactly cover the required namespaces: "
            f"missing={sorted(expected - set(values))}, extra={sorted(set(values) - expected)}"
        )
    records = list(zero_records)
    records.extend(
        (label, value, context.RUN_ID, context.SOURCE) for label, value in context_values.items()
    )
    return tuple(sorted(records))


def _synthetic_v2_claim_entries() -> dict[str, dict[str, object]]:
    """Return fresh mutable entries backed by the cached pure derivation."""

    return {
        label: {"value": value, "run_id": run_id, "source": source}
        for label, value, run_id, source in _synthetic_v2_fixture_records()
    }


def _full_claims() -> dict:
    """A complete, internally coherent fixture for every derived branch.

    Uniform ``1.0`` placeholders made impossible McNemar tables and once forced
    production invariants off. The default remains compact, but every count used
    in a paired interval now satisfies ``b + c <= n`` and the labels driving
    categorical interpretations and structured tables satisfy their real
    relationships.
    """
    # Keep the mature non-synthetic families as a compact fixed fixture, but do
    # not let a stale or absent publication recording block v2 unit tests. The
    # synthetic namespaces above are independently derived from protocol and
    # deterministic offline response flags. Real-ledger readiness remains in
    # ``TestRealLedger`` below.
    claims = {}
    for label, entry in _real_claims().items():
        if label.startswith(("bench_synth_", "bench_synthphone_", "bench_ctx_")):
            continue
        if label.startswith(("bench_synthg2p_comparator_", "connected_gum_")):
            # The aligned-G2P and connected-GUM namespaces are checked for provenance
            # at render time against its recorder's binding, so the fixture
            # keeps the recorded source/run_id verbatim rather than
            # overwriting it with the fixture run stamp.
            claims[label] = dict(entry)
        else:
            claims[label] = {**entry, "run_id": "test_run", "source": "test"}
    claims.update(_synthetic_v2_claim_entries())
    return claims


def test_required_claim_inventory_has_no_duplicate_labels():
    assert len(REQUIRED_CLAIMS) == len(set(REQUIRED_CLAIMS))


# ── _format_value ─────────────────────────────────────────────────────────────


class TestFormatValue:
    """Formatting must not corrupt a measurement on its way into prose."""

    def test_small_int_is_bare(self):
        assert _format_value(39) == "39"

    def test_large_int_gets_thousands_separators(self):
        assert _format_value(42820) == "42,820"

    def test_int_stays_integral(self):
        """A position count must never render as 42820.0."""
        assert "." not in _format_value(42820)

    def test_float_trims_trailing_zeros(self):
        assert _format_value(30.0) == "30"

    def test_float_keeps_precision(self):
        assert _format_value(11.9215) == "11.9215"

    @pytest.mark.parametrize(
        "value",
        (11.9215, 0.6293, 0.000594947, 3.468e-06, 1.23456789012345),
    )
    def test_float_text_round_trips_exactly(self, value):
        assert float(_format_value(value)) == value

    def test_tiny_float_uses_scientific_notation(self):
        """p-values must survive as 3.47e-06, not collapse to 0."""
        rendered = _format_value(3.468e-06)
        assert "e-06" in rendered
        assert float(rendered) == pytest.approx(3.468e-06, rel=1e-2)

    def test_zero_is_zero(self):
        assert _format_value(0) == "0"
        assert _format_value(0.0) == "0"

    def test_bool_is_lowercase_word(self):
        assert _format_value(True) == "true"
        assert _format_value(False) == "false"

    def test_string_passes_through(self):
        assert _format_value("abc") == "abc"

    def test_json_collections_use_deterministic_json_not_python_repr(self):
        assert _format_value(["a", 2]) == '["a",2]'
        assert _format_value({"b": 2, "a": 1}) == '{"a":1,"b":2}'

    def test_null_is_rejected_instead_of_rendering_as_none(self):
        with pytest.raises(ManuscriptVariableError, match="JSON scalars"):
            _format_value(None)

    @_SKIP_NO_LEDGER
    def test_distinct_recorded_floats_do_not_collapse_to_one_display(self):
        values = {
            entry["value"]
            for entry in _real_claims().values()
            if isinstance(entry.get("value"), float)
        }
        rendered = {_format_value(value) for value in values}
        assert len(rendered) == len(values), (
            "distinct recorded floats collide after formatting; manuscript "
            "relationships can become false at display precision"
        )


# ── The failure path — this is the load-bearing behavior ────────────────────


class TestMissingClaimFailsLoudly:
    """A missing claim must raise, never emit a default or an unresolved token."""

    def test_absent_ledger_raises(self, tmp_path):
        (tmp_path / "docs" / "manuscript").mkdir(parents=True)
        with pytest.raises(MissingClaimError, match="not found"):
            generate_manuscript_variables(tmp_path)

    def test_empty_ledger_raises(self, tmp_path):
        root = _write_project(tmp_path, {})
        with pytest.raises(MissingClaimError, match="require"):
            generate_manuscript_variables(root)

    def test_one_missing_claim_raises_and_names_it(self, tmp_path):
        """The error must identify WHICH claim is missing, or it is unactionable."""
        claims = _full_claims()
        del claims["stage1_letter_perplexity"]
        root = _write_project(tmp_path, claims)
        with pytest.raises(MissingClaimError, match="stage1_letter_perplexity"):
            generate_manuscript_variables(root)

    @pytest.mark.parametrize("label", REQUIRED_CLAIMS)
    def test_every_required_claim_is_individually_enforced(self, tmp_path, label):
        """POSITIVE CONTROL on the guard: removing ANY required claim must fail.

        Without this, REQUIRED_CLAIMS could list a label that is never checked and
        the guard would be partially vacuous.
        """
        claims = _full_claims()
        del claims[label]
        root = _write_project(tmp_path, claims)
        with pytest.raises(MissingClaimError):
            generate_manuscript_variables(root)

    def test_complete_ledger_does_not_raise(self, tmp_path):
        """NEGATIVE CONTROL: the guard must not fire on a complete ledger."""
        root = _write_project(tmp_path, _full_claims())
        variables = generate_manuscript_variables(root)
        assert variables  # non-empty

    def test_negative_cluster_variance_ratio_fails_before_manuscript_injection(self, tmp_path):
        """A variance ratio cannot be negative, even when the JSON is well-typed."""
        claims = _full_claims()
        claims["cluster_stage2_g2p_variance_inflation"]["value"] = -1.0
        root = _write_project(tmp_path, claims)
        with pytest.raises(ManuscriptVariableError, match="non-negative real"):
            generate_manuscript_variables(root)


# ── Token construction ────────────────────────────────────────────────────────


class TestTokenConstruction:
    """Tokens are the ledger label uppercased — no second mapping to drift."""

    def test_label_is_uppercased(self, tmp_path):
        root = _write_project(tmp_path, _full_claims())
        variables = generate_manuscript_variables(root)
        assert "STAGE1_LETTER_PERPLEXITY" in variables

    def test_no_lowercase_token_names(self, tmp_path):
        root = _write_project(tmp_path, _full_claims())
        variables = generate_manuscript_variables(root)
        offenders = [k for k in variables if k != k.upper()]
        assert offenders == [], f"token names must be UPPERCASE: {offenders}"

    def test_every_value_is_a_string(self, tmp_path):
        """Substitution is textual; a non-str value would break injection."""
        root = _write_project(tmp_path, _full_claims())
        variables = generate_manuscript_variables(root)
        bad = {k: type(v).__name__ for k, v in variables.items() if not isinstance(v, str)}
        assert bad == {}, f"non-string token values: {bad}"

    def test_provenance_companions_present(self, tmp_path):
        root = _write_project(tmp_path, _full_claims())
        variables = generate_manuscript_variables(root)
        assert variables["STAGE1_LETTER_PERPLEXITY_RUN"] == "test_run"
        assert variables["STAGE1_LETTER_PERPLEXITY_SOURCE"] == "test"

    def test_config_metadata_flows_through(self, tmp_path):
        root = _write_project(
            tmp_path,
            _full_claims(),
            {
                "paper": {"title": "A Title", "version": "9.9.9"},
                "authors": [{"name": "Ada"}, {"name": "Grace"}],
                "keywords": ["one", "two"],
                "rendering": _valid_config()["rendering"],
            },
        )
        variables = generate_manuscript_variables(root)
        assert variables["META_TITLE"] == "A Title"
        assert variables["META_VERSION"] == "9.9.9"
        assert variables["META_AUTHOR_COUNT"] == "2"
        assert "Ada" in variables["META_AUTHORS"]
        assert variables["META_KEYWORDS"] == "one, two"

    def test_stage1_fixture_metadata_flows_from_the_authenticated_protocol(self, tmp_path):
        from ailm.evaluation import STAGE1_EVALUATION

        variables = generate_manuscript_variables(_write_project(tmp_path, _full_claims()))
        assert variables["STAGE1_EVALUATION_PROTOCOL"] == STAGE1_EVALUATION.protocol
        assert variables["STAGE1_EVALUATION_SHA256"] == STAGE1_EVALUATION.sha256
        assert variables["DIGEST_SHORT_LENGTH"] == str(DIGEST_SHORT_LENGTH)
        assert (
            variables["STAGE1_EVALUATION_SHA256_SHORT"]
            == STAGE1_EVALUATION.sha256[:DIGEST_SHORT_LENGTH]
        )
        assert int(variables["STAGE1_EVALUATION_SIZE_BYTES"]) == STAGE1_EVALUATION.size_bytes
        assert int(variables["STAGE1_EVALUATION_REPETITIONS"]) == STAGE1_EVALUATION.repetitions
        assert variables["STAGE1_EVALUATION_SOURCE"] == STAGE1_EVALUATION.source
        assert variables["STAGE1_EVALUATION_LICENSE"] == STAGE1_EVALUATION.license

    def test_learning_rate_token_comes_from_the_learning_api_default(self, tmp_path):
        import inspect

        from ailm.learning import CONJUGATE_LEARNING_RATE, learn_letter_transition

        variables = generate_manuscript_variables(_write_project(tmp_path, _full_claims()))
        assert float(variables["ONLINE_LEARNING_RATE"]) == CONJUGATE_LEARNING_RATE
        assert (
            inspect.signature(learn_letter_transition).parameters["lr"].default
            == CONJUGATE_LEARNING_RATE
        )

    def test_historical_stage2_gate_comes_from_the_gate_registry(self, tmp_path):
        from ailm.gates import HISTORICAL_STAGE2_G2P_THRESHOLD_PCT

        root = _write_project(tmp_path, _full_claims())
        variables = generate_manuscript_variables(root)
        assert variables["HISTORICAL_STAGE2_GATE_PCT"] == _format_value(
            HISTORICAL_STAGE2_G2P_THRESHOLD_PCT
        )
        literals = yaml.safe_load(
            (root / "data" / "manuscript_literals.yaml").read_text(encoding="utf-8")
        )["variables"]
        assert "HISTORICAL_STAGE2_GATE_PCT" not in literals

        # POSITIVE CONTROL: the former duplicate accepted any plausible
        # percentage (including 61) while the figure still used 60. The strict
        # registry now has no override path at all.
        literals["HISTORICAL_STAGE2_GATE_PCT"] = {
            "value": 61,
            "reason": "Planted stale duplicate of the code-owned historical threshold.",
        }
        (root / "data" / "manuscript_literals.yaml").write_text(
            yaml.safe_dump({"schema": "ailm-manuscript-literals-v2", "variables": literals}),
            encoding="utf-8",
        )
        with pytest.raises(InvalidLiteralVariableError, match="typed schema"):
            generate_manuscript_variables(root)

    def test_g2p_protocol_words_are_derived_from_recorded_code_constants(self, tmp_path):
        from ailm.levels.phoneme import G2P_FAMILY_PREFIX_LENGTH, G2P_FAMILY_STRIDE

        variables = generate_manuscript_variables(_write_project(tmp_path, _full_claims()))
        assert variables["STAGE2_G2P_FAMSPLIT_PREFIX_LENGTH"] == str(G2P_FAMILY_PREFIX_LENGTH)
        assert variables["STAGE2_G2P_FAMSPLIT_FAMILY_STRIDE"] == str(G2P_FAMILY_STRIDE)
        assert variables["STAGE2_G2P_FAMSPLIT_FAMILY_STRIDE_ORDINAL"] == "fifth"
        assert variables["STAGE2_G2P_SPLIT_STRIDE_ORDINAL"] == "fifth"

    def test_significance_family_prose_counts_are_derived_not_stored(self, tmp_path):
        from ailm.statistics import RETROSPECTIVE_POSTHOC_MEMBERS

        claims = _full_claims()
        variables = generate_manuscript_variables(_write_project(tmp_path, claims))
        assert variables["STAT_POSTHOC_MEMBER_COUNT"] == str(len(RETROSPECTIVE_POSTHOC_MEMBERS))
        assert variables["STAT_POSTHOC_MEMBER_COUNT_WORD"] == "three"
        assert variables["STAT_STRUCTURALLY_UNREJECTABLE_COUNT"] == "2"
        assert variables["STAT_STRUCTURALLY_UNREJECTABLE_COUNT_WORD"] == "two"
        assert "stat_posthoc_member_count" not in claims
        assert "stat_structurally_unrejectable_count" not in claims

    def test_parameter_table_rows_are_generated_from_the_exact_gnn_registry(self, tmp_path):
        from ailm.parameters import GNN_SPEC_FILES

        variables = generate_manuscript_variables(_write_project(tmp_path, _full_claims()))
        rows = variables["PARAMETER_TABLE_ROWS"].splitlines()
        assert len(rows) == len(GNN_SPEC_FILES)
        for path in GNN_SPEC_FILES.values():
            assert sum(f"`{path}`" in row for row in rows) == 1

    def test_missing_config_fails_loudly(self, tmp_path):
        """Metadata may not degrade to a plausible title/version default."""
        (tmp_path / "data").mkdir(parents=True)
        (tmp_path / "data" / "claims.json").write_text(
            json.dumps({"_schema": "ailm-claims-v2", "claims": _full_claims()}),
            encoding="utf-8",
        )
        with pytest.raises(ManuscriptVariableError, match="configuration not found"):
            generate_manuscript_variables(tmp_path)

    def test_source_date_epoch_makes_the_stamp_reproducible(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
        root = _write_project(tmp_path, _full_claims())
        first = generate_manuscript_variables(root)["META_BUILD_TIMESTAMP"]
        second = generate_manuscript_variables(root)["META_BUILD_TIMESTAMP"]
        assert first == second
        assert first.startswith("2023-11-")

    @pytest.mark.parametrize("bad_epoch", ("not-an-integer", "-1", "1.25"))
    def test_nonempty_invalid_source_date_epoch_fails_closed(
        self, tmp_path, monkeypatch, bad_epoch
    ):
        monkeypatch.setenv("SOURCE_DATE_EPOCH", bad_epoch)
        root = _write_project(tmp_path, _full_claims())
        with pytest.raises(ManuscriptVariableError, match="SOURCE_DATE_EPOCH"):
            generate_manuscript_variables(root)

    def test_derived_quantities_are_computed_not_stored(self, tmp_path):
        """Derived tokens must be arithmetic on the ledger, not separate entries."""
        claims = _full_claims()
        claims["stage1_unigram_baseline"] = {"value": 20.0, "run_id": "r", "source": "s"}
        claims["stage1_letter_perplexity"] = {"value": 12.0, "run_id": "r", "source": "s"}
        root = _write_project(tmp_path, claims)
        variables = generate_manuscript_variables(root)
        assert float(variables["STAGE1_UNIGRAM_MARGIN"]) == pytest.approx(8.0)

    def test_stage5_interval_language_tracks_endpoints_and_rejects_reversal(self, tmp_path):
        """Decision 83's prose verdict is derived, not a hard-coded manuscript claim."""
        claims = _full_claims()
        current = generate_manuscript_variables(_write_project(tmp_path / "current", claims))
        assert current["STAGE5_ORACLE_GAIN_SENTENCE_INTERVAL_STATUS"] == "spans zero"
        assert current["STAGE5_ORACLE_GAIN_POSITION_INTERVAL_STATUS"] == "spans zero"
        assert current["STAGE5_ORACLE_GAIN_RESOLUTION"] == "unresolved on both reported scales"

        positive = _full_claims()
        positive["stage5_oracle_gain_sentence_ci_low"]["value"] = 0.001
        positive["stage5_oracle_gain_sentence_ci_high"]["value"] = 0.01
        positive["stage5_oracle_gain_position_ci_low"]["value"] = 0.0001
        positive["stage5_oracle_gain_position_ci_high"]["value"] = 0.001
        resolved = generate_manuscript_variables(_write_project(tmp_path / "resolved", positive))
        assert resolved["STAGE5_ORACLE_GAIN_SENTENCE_INTERVAL_STATUS"] == ("is entirely positive")
        assert resolved["STAGE5_ORACLE_GAIN_RESOLUTION"] == (
            "resolved on at least one reported scale"
        )

        reversed_claims = _full_claims()
        reversed_claims["stage5_oracle_gain_sentence_ci_low"]["value"] = 0.1
        reversed_claims["stage5_oracle_gain_sentence_ci_high"]["value"] = -0.1
        with pytest.raises(ManuscriptVariableError, match="interval endpoints are reversed"):
            generate_manuscript_variables(_write_project(tmp_path / "reversed", reversed_claims))

    def test_case_path_headroom_tokens_are_exact_ledger_derivations(self, tmp_path):
        claims = _full_claims()
        variables = generate_manuscript_variables(_write_project(tmp_path, claims))
        for prefix in ("deprel", "morphology"):
            baseline = claims[f"stage4_path_{prefix}_baseline"]["value"]
            margin = claims[f"stage4_path_{prefix}_margin_pts"]["value"]
            expected = 100.0 * margin / (100.0 - baseline)
            token = f"STAGE4_PATH_{prefix.upper()}_HEADROOM_CAPTURED_PCT"
            assert float(variables[token]) == pytest.approx(expected, abs=0.005)

    def test_case_path_headroom_derivation_rejects_incoherent_margin(self, tmp_path):
        claims = _full_claims()
        claims["stage4_path_deprel_margin_pts"] = {
            **claims["stage4_path_deprel_margin_pts"],
            "value": claims["stage4_path_deprel_margin_pts"]["value"] + 1.0,
        }
        with pytest.raises(ManuscriptVariableError, match="agreement minus baseline"):
            generate_manuscript_variables(_write_project(tmp_path, claims))

    def test_case_folded_claim_collision_is_rejected(self, tmp_path):
        claims = _full_claims()
        claims["STAGE1_LETTER_PERPLEXITY"] = {
            "value": 99,
            "run_id": "collision",
            "source": "test",
        }
        root = _write_project(tmp_path, claims)
        with pytest.raises(DuplicateManuscriptTokenError, match="more than one producer"):
            generate_manuscript_variables(root)

    def test_claim_cannot_shadow_another_claims_provenance_companion(self, tmp_path):
        claims = _full_claims()
        claims["stage1_letter_perplexity_run"] = {
            "value": 99,
            "run_id": "collision",
            "source": "test",
        }
        root = _write_project(tmp_path, claims)
        with pytest.raises(DuplicateManuscriptTokenError, match="more than one producer"):
            generate_manuscript_variables(root)

    def test_untyped_literal_cannot_shadow_a_recorded_claim(self, tmp_path):
        root = _write_project(tmp_path, _full_claims())
        (root / "data" / "manuscript_literals.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema": "ailm-manuscript-literals-v2",
                    "variables": {
                        "STAGE1_LETTER_PERPLEXITY": {
                            "value": "wrong",
                            "reason": "This deliberately duplicates a recorded claim token for testing.",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(InvalidLiteralVariableError, match="typed schema"):
            generate_manuscript_variables(root)

    @pytest.mark.parametrize(
        ("payload", "message"),
        (
            ({"schema": "wrong", "variables": {}}, "schema"),
            ({"schema": "ailm-manuscript-literals-v2", "variables": {}}, "non-empty"),
            (
                {
                    "schema": "ailm-manuscript-literals-v2",
                    "variables": {"bad-name": {"value": 1, "reason": "long enough reason text"}},
                },
                "invalid literal variable name",
            ),
        ),
    )
    def test_literal_variable_schema_fails_closed(self, tmp_path, payload, message):
        root = _write_project(tmp_path, _full_claims())
        (root / "data" / "manuscript_literals.yaml").write_text(
            yaml.safe_dump(payload), encoding="utf-8"
        )
        with pytest.raises(InvalidLiteralVariableError, match=message):
            generate_manuscript_variables(root)

    @pytest.mark.parametrize(
        ("name", "bad_value", "message"),
        (("DECISION_UNOBSERVED_EMISSION_LEARNING", 999999, "absent decision"),),
    )
    def test_literal_values_obey_their_key_specific_schema(
        self, tmp_path, name, bad_value, message
    ):
        root = _write_project(tmp_path, _full_claims())
        path = root / "data" / "manuscript_literals.yaml"
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        payload["variables"][name]["value"] = bad_value
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with pytest.raises(InvalidLiteralVariableError, match=message):
            generate_manuscript_variables(root)

    @pytest.mark.parametrize("payload", ("- not-a-mapping\n", "plain scalar\n"))
    def test_literal_registry_root_must_be_a_mapping(self, tmp_path, payload):
        root = _write_project(tmp_path, _full_claims())
        (root / "data" / "manuscript_literals.yaml").write_text(payload, encoding="utf-8")
        with pytest.raises(InvalidLiteralVariableError, match="YAML mapping"):
            generate_manuscript_variables(root)

    def test_missing_literal_registry_fails_loudly(self, tmp_path):
        root = _write_project(tmp_path, _full_claims())
        (root / "data" / "manuscript_literals.yaml").unlink()
        with pytest.raises(InvalidLiteralVariableError, match="registry not found"):
            generate_manuscript_variables(root)

    @pytest.mark.parametrize(
        "artifact",
        (
            "brown_counts.json",
            "cmudict_counts.json",
            "brown_word_counts.json",
            "ud_ewt_case_counts.json",
        ),
    )
    def test_each_required_vendor_artifact_is_enforced(self, tmp_path, artifact):
        root = _write_project(tmp_path, _full_claims())
        (root / "vendor" / artifact).unlink()
        with pytest.raises(ManuscriptVariableError, match=artifact):
            generate_manuscript_variables(root)


class TestInputValidation:
    def test_duplicate_json_claim_key_fails_instead_of_last_wins(self, tmp_path):
        root = _write_project(tmp_path, _full_claims())
        (root / "data" / "claims.json").write_text(
            '{"_schema":"ailm-claims-v2","claims":{},"claims":{}}',
            encoding="utf-8",
        )
        with pytest.raises(DuplicateInputKeyError, match="duplicate JSON key 'claims'"):
            generate_manuscript_variables(root)


class TestLiteratureSearchVariables:
    """Priority-search prose is bounded by a structured evidence record."""

    def _payload(self, root):
        path = root / "data" / "literature_search.json"
        return path, json.loads(path.read_text(encoding="utf-8"))

    def test_cutoff_and_priority_policy_are_derived_from_search_record(self, tmp_path):
        root = _write_project(tmp_path, _full_claims())
        variables = _load_literature_variables(root)
        assert variables["LITERATURE_SEARCH_DATE"] == "2026-08-23"
        assert variables["LITERATURE_SEARCH_CUTOFF"] == "August 2026"
        assert variables["LITERATURE_PRIORITY_CLAIM_ALLOWED"] == "false"
        assert variables["LITERATURE_SEARCH_SOURCE_COUNT"] == "3"
        assert variables["LITERATURE_SCREENED_CANDIDATE_COUNT"] == "7"

    def test_incomplete_search_cannot_authorize_a_priority_claim(self, tmp_path):
        """NEGATIVE CONTROL for the manuscript's former 'first' wording."""
        root = _write_project(tmp_path, _full_claims())
        path, payload = self._payload(root)
        payload["claim_policy"]["priority_claim_allowed"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(InvalidLiteratureSearchError, match="cannot allow a priority claim"):
            _load_literature_variables(root)

    def test_candidate_eligibility_is_the_conjunction_of_declared_facets(self, tmp_path):
        root = _write_project(tmp_path, _full_claims())
        path, payload = self._payload(root)
        candidate = payload["screened_candidates"][0]
        candidate["natural_language_corpus_fit"] = True
        candidate["heldout_language_model_metric"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(InvalidLiteratureSearchError, match="eligibility contradicts"):
            _load_literature_variables(root)

        candidate["eligible"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(InvalidLiteratureSearchError, match="eligible counterexample"):
            _load_literature_variables(root)

    def test_cutoff_must_be_derived_from_real_search_date(self, tmp_path):
        root = _write_project(tmp_path, _full_claims())
        path, payload = self._payload(root)
        payload["cutoff_display"] = "September 2026"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(InvalidLiteratureSearchError, match="derived from conducted_on"):
            _load_literature_variables(root)

    def test_missing_and_duplicate_search_records_fail_loudly(self, tmp_path):
        root = _write_project(tmp_path, _full_claims())
        path = root / "data" / "literature_search.json"
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace(
                '"conducted_on": "2026-08-23"',
                '"conducted_on": "2026-01-01", "conducted_on": "2026-08-23"',
                1,
            ),
            encoding="utf-8",
        )
        with pytest.raises(DuplicateInputKeyError, match="duplicate JSON key 'conducted_on'"):
            _load_literature_variables(root)

        path.unlink()
        with pytest.raises(InvalidLiteratureSearchError, match="record not found"):
            _load_literature_variables(root)


class TestCanonicalReferenceVariables:
    """Deposited-source tokens have one mutation-sensitive JSON owner."""

    @pytest.mark.parametrize(
        ("keys", "replacement", "token", "expected"),
        (
            (
                ("sources", "cerebrum", "zenodo_record", "version"),
                "1.5",
                "CEREBRUM_ZENODO_VERSION",
                "1.5",
            ),
            (
                ("sources", "cerebrum", "printed_identity", "version"),
                "1.3",
                "CEREBRUM_PRINTED_VERSION",
                "1.3",
            ),
            (
                ("sources", "cerebrum", "zenodo_record", "publication_date"),
                "2025-04-17",
                "CEREBRUM_ZENODO_DATE",
                "2025-04-17",
            ),
            (
                ("sources", "cerebrum", "printed_identity", "date"),
                "2025-04-13",
                "CEREBRUM_PRINTED_DATE",
                "2025-04-13",
            ),
            (
                ("sources", "cerebrum", "deposited_file", "pdf_pages"),
                147,
                "CEREBRUM_PDF_PAGE_COUNT",
                "147",
            ),
            (
                ("sources", "compositional_case", "deposited_file", "pdf_pages"),
                114,
                "COMPOSITIONAL_CASE_PDF_PAGE_COUNT",
                "114",
            ),
            (
                ("sources", "cerebrum", "pdf_evidence", "novel_generative_case", "locator"),
                "Supplement 2.13",
                "CEREBRUM_GENERATIVE_SUPPLEMENT",
                "2.13",
            ),
            (
                ("sources", "cerebrum", "pdf_evidence", "core_case_formulations_locator"),
                "Supplement 12",
                "CEREBRUM_CASE_SUPPLEMENT",
                "12",
            ),
            (
                ("sources", "cerebrum", "pdf_evidence", "active_inference_figure", "locator"),
                "Figure 14",
                "CEREBRUM_ACTIVE_INFERENCE_FIGURE",
                "14",
            ),
            (
                ("sources", "cerebrum", "pdf_evidence", "pomdp_tuple", "locator"),
                "Supplement 8.2",
                "CEREBRUM_POMDP_SUPPLEMENT",
                "8.2",
            ),
        ),
    )
    def test_each_canonical_field_mutation_changes_its_token(
        self, tmp_path, keys, replacement, token, expected
    ):
        root = _write_project(tmp_path, _full_claims())
        path = root / "data" / "reference_sources.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        target = payload
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = replacement
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert _load_reference_variables(root)[token] == expected

    @pytest.mark.parametrize(
        ("keys", "replacement"),
        (
            (("sources", "cerebrum", "pdf_evidence", "novel_generative_case", "locator"), "2.12"),
            (
                ("sources", "cerebrum", "pdf_evidence", "core_case_formulations_locator"),
                "section eleven",
            ),
            (
                ("sources", "cerebrum", "pdf_evidence", "active_inference_figure", "locator"),
                "Fig. 13",
            ),
            (
                ("sources", "cerebrum", "pdf_evidence", "pomdp_tuple", "locator"),
                "Supplement eight.one",
            ),
        ),
    )
    def test_malformed_reference_locators_fail_closed(self, tmp_path, keys, replacement):
        root = _write_project(tmp_path, _full_claims())
        path = root / "data" / "reference_sources.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        target = payload
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = replacement
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(InvalidReferenceSourceError, match="wrong shape"):
            _load_reference_variables(root)

    def test_duplicate_reference_key_fails_instead_of_last_wins(self, tmp_path):
        root = _write_project(tmp_path, _full_claims())
        path = root / "data" / "reference_sources.json"
        text = path.read_text(encoding="utf-8").replace(
            '"version": "1.4"', '"version": "1.4", "version": "9.9"', 1
        )
        path.write_text(text, encoding="utf-8")
        with pytest.raises(DuplicateInputKeyError, match="duplicate JSON key 'version'"):
            _load_reference_variables(root)

    def test_missing_reference_snapshot_fails_loudly(self, tmp_path):
        root = _write_project(tmp_path, _full_claims())
        (root / "data" / "reference_sources.json").unlink()
        with pytest.raises(InvalidReferenceSourceError, match="snapshot not found"):
            _load_reference_variables(root)

    @pytest.mark.parametrize("bad_page_count", (0, -1, True, "113"))
    def test_reference_page_count_must_be_a_positive_integer(self, tmp_path, bad_page_count):
        root = _write_project(tmp_path, _full_claims())
        path = root / "data" / "reference_sources.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["sources"]["compositional_case"]["deposited_file"]["pdf_pages"] = bad_page_count
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(InvalidReferenceSourceError, match="positive page count"):
            _load_reference_variables(root)


class TestCodeOwnedManuscriptConfiguration:
    def test_g2p_bucket_boundaries_and_range_words_come_from_the_registry(self):
        assert _g2p_bucket_variables() == {
            "BENCH_G2P_BUCKET_SHORT_LOW": "0",
            "BENCH_G2P_BUCKET_SHORT_HIGH": "5",
            "BENCH_G2P_BUCKET_SHORT_RANGE": "five phones or fewer",
            "BENCH_G2P_BUCKET_MID_LOW": "6",
            "BENCH_G2P_BUCKET_MID_HIGH": "9",
            "BENCH_G2P_BUCKET_MID_RANGE": "six to nine phones",
            "BENCH_G2P_BUCKET_LONG_LOW": "10",
            "BENCH_G2P_BUCKET_LONG_HIGH": "1000000",
            "BENCH_G2P_BUCKET_LONG_RANGE": "ten phones or more",
        }

    @pytest.mark.parametrize(
        "buckets",
        (
            (("short", 1, 5), ("mid", 6, 9), ("long", 10, 100)),
            (("short", 0, 5), ("mid", 7, 9), ("long", 10, 100)),
            (("short", 0, 5), ("middle", 6, 9), ("long", 10, 100)),
            (("short", 0, True), ("mid", 2, 9), ("long", 10, 100)),
        ),
    )
    def test_invalid_g2p_bucket_registry_fails_closed(self, buckets):
        with pytest.raises(ManuscriptVariableError, match="G2P_BUCKETS|G2P bucket|first"):
            _g2p_bucket_variables(buckets)


class TestInputValidationContinued:
    def test_duplicate_config_key_fails_instead_of_last_wins(self, tmp_path):
        root = _write_project(tmp_path, _full_claims())
        (root / "docs" / "manuscript" / "config.yaml").write_text(
            "paper: {title: First, version: '1'}\n"
            "paper: {title: Second, version: '2'}\n"
            "authors: [{name: Author}]\nkeywords: [testing]\n",
            encoding="utf-8",
        )
        with pytest.raises(DuplicateInputKeyError, match="duplicate YAML key 'paper'"):
            generate_manuscript_variables(root)

    def test_duplicate_literal_key_fails_instead_of_last_wins(self, tmp_path):
        root = _write_project(tmp_path, _full_claims())
        (root / "data" / "manuscript_literals.yaml").write_text(
            "schema: ailm-manuscript-literals-v2\nvariables:\n"
            "  DUPLICATE: {value: first, reason: 'A sufficiently long first reason.'}\n"
            "  DUPLICATE: {value: second, reason: 'A sufficiently long second reason.'}\n",
            encoding="utf-8",
        )
        with pytest.raises(DuplicateInputKeyError, match="duplicate YAML key 'DUPLICATE'"):
            generate_manuscript_variables(root)

    def test_unsafe_yaml_constructor_is_rejected_without_side_effect(self, tmp_path):
        root = _write_project(tmp_path, _full_claims())
        sentinel = tmp_path / "unsafe-yaml-constructor-ran"
        (root / "docs" / "manuscript" / "config.yaml").write_text(
            "paper: !!python/object/apply:builtins.open\n"
            f"  - {json.dumps(str(sentinel))}\n"
            "  - w\n"
            "authors: [{name: Author}]\nkeywords: [testing]\n",
            encoding="utf-8",
        )

        with pytest.raises(ManuscriptVariableError, match="invalid YAML"):
            generate_manuscript_variables(root)

        assert not sentinel.exists()

    @pytest.mark.parametrize("missing_field", ("value", "run_id", "source"))
    def test_claim_entry_missing_required_field_fails(self, tmp_path, missing_field):
        claims = _full_claims()
        del claims["stage1_letter_perplexity"][missing_field]
        root = _write_project(tmp_path, claims)
        with pytest.raises(MissingClaimError, match="Invalid claim entry"):
            generate_manuscript_variables(root)

    @pytest.mark.parametrize(
        "config",
        (
            {},
            {"paper": {"title": "", "version": "1"}, "authors": [{"name": "A"}], "keywords": ["k"]},
            {"paper": {"title": "T", "version": ""}, "authors": [{"name": "A"}], "keywords": ["k"]},
            {"paper": {"title": "T", "version": "1"}, "authors": [], "keywords": ["k"]},
            {
                "paper": {"title": "T", "version": "1"},
                "authors": [{"name": 123}],
                "keywords": ["k"],
            },
            {
                "paper": {"title": "T", "version": "1"},
                "authors": [{"name": True}],
                "keywords": ["k"],
            },
            {"paper": {"title": "T", "version": "1"}, "authors": [{"name": "A"}], "keywords": []},
        ),
    )
    def test_incomplete_metadata_fails(self, tmp_path, config):
        root = _write_project(tmp_path, _full_claims(), config)
        with pytest.raises(ManuscriptVariableError):
            generate_manuscript_variables(root)

    @pytest.mark.parametrize("bad_value", ("0.5", True, None, float("inf")))
    def test_numeric_derivation_rejects_non_numeric_or_non_finite_values(self, tmp_path, bad_value):
        claims = _full_claims()
        claims["generation_tv_improvement"]["value"] = bad_value
        root = _write_project(tmp_path, claims)
        with pytest.raises(ManuscriptVariableError, match="numeric|finite|JSON scalars"):
            generate_manuscript_variables(root)

    @pytest.mark.parametrize(
        ("label", "bad_value", "message"),
        (
            ("stage1_letter_nats", "not-a-number", "numeric"),
            ("stage4_gold_agreement", "not-a-percentage", "numeric"),
            ("bench_letter_top1_gemma3", True, "numeric"),
            ("stage1_brown_bigram_tokens", -999, "non-negative whole"),
            ("stage4_gold_mcnemar_p", 2, "unit interval"),
        ),
    )
    def test_every_exposed_claim_obeys_a_type_and_domain_contract(
        self, tmp_path, label, bad_value, message
    ):
        claims = _full_claims()
        claims[label]["value"] = bad_value
        root = _write_project(tmp_path, claims)
        with pytest.raises(ManuscriptVariableError, match=message):
            generate_manuscript_variables(root)

    @pytest.mark.parametrize(
        ("label", "value"),
        (
            ("stage3_scale_gain_rise_pct", 250.0),
            ("stage3_oov_gap_change_under_matching_pct", 250.0),
            ("stage5_privileged_label_gain_pct", 100.0),
            ("stage5_privileged_label_gain_pct", -1000.0),
        ),
    )
    def test_relative_percentage_changes_are_not_capped_like_shares(self, label, value):
        _validate_required_claim_values({label: {"value": value}})

    @pytest.mark.parametrize(
        ("label", "value", "message"),
        (
            ("stage3_scale_gain_rise_pct", -0.01, "non-negative relative percentage"),
            (
                "stage3_oov_gap_change_under_matching_pct",
                -0.01,
                "non-negative relative percentage",
            ),
            ("stage5_privileged_label_gain_pct", 100.01, "signed relative percentage"),
            ("stage3_coverage_pct", 100.01, "must lie in \\[0, 100\\]"),
            ("stage3_scale_gain_rise_pct", float("inf"), "finite"),
        ),
    )
    def test_relative_percentage_domains_fail_closed(self, label, value, message):
        with pytest.raises(ManuscriptVariableError, match=message):
            _validate_required_claim_values({label: {"value": value}})

    def test_stage3_scale_rise_claim_must_match_the_recorded_endpoints(self, tmp_path):
        claims = _full_claims()
        claims["stage3_scale_gain_rise_pct"]["value"] += 0.01
        root = _write_project(tmp_path, claims)
        with pytest.raises(
            ManuscriptVariableError, match="stage3_scale_gain_rise_pct.*inconsistent"
        ):
            generate_manuscript_variables(root)

    @pytest.mark.parametrize(
        "label",
        (
            "generation_tv_improvement",
            "stage3_oov_gap_change_under_matching_pct",
            "stage5_privileged_label_gain_pct",
        ),
    )
    def test_relative_summary_must_match_its_recorded_operands(self, tmp_path, label):
        claims = _full_claims()
        claims[label]["value"] += 0.01
        root = _write_project(tmp_path, claims)
        with pytest.raises(ManuscriptVariableError, match=rf"{label}.*inconsistent"):
            generate_manuscript_variables(root)

    @pytest.mark.parametrize(
        ("label", "bad_value", "domain"),
        (
            ("stage4_gold_agreement", 100.01, "percentage"),
            ("bench_calib_argmax_agreement_gemma3", 1.01, "unit interval"),
            ("bench_determinism_agreement_qwen25", -0.01, "unit interval"),
            ("stage5_auc", 1.01, "unit interval"),
            ("ewt_boundary_stage5_before_auc", -0.01, "unit interval"),
            ("stage2_g2p_split_initial_tvd", 1.01, "unit interval"),
            ("generation_baseline_tv", -0.01, "unit interval"),
            ("bench_floor_identity_ece", 1.01, "unit interval"),
            ("stat_reference_share", 1.01, "unit interval"),
            ("bench_power_reference_share", -0.01, "unit interval"),
            ("bench_logprob_drift_probability", 1.01, "unit interval"),
            ("bench_floor_located_over_cap_fraction", 1.01, "unit interval"),
            ("bench_g2p_pos_long_qwen25", 1.01, "unit interval"),
            ("bench_letter_delta_gemma3", -1.01, "signed unit interval"),
            ("bench_calib_measurable_gemma3", 2, "binary indicator"),
            ("stage1_brown_bigram_tokens", 1.5, "non-negative whole count"),
            ("stage1_brown_bigram_tokens", -1, "non-negative whole count"),
            ("stage4_n_cases", 1.5, "non-negative whole count"),
            ("bench_calib_top_logprobs", 1.5, "non-negative whole count"),
            ("clause_word_sweep_alpha_0_1_improved", -1, "non-negative whole count"),
        ),
    )
    def test_semantic_domains_cover_model_qualified_and_unit_ambiguous_labels(
        self, label, bad_value, domain
    ):
        with pytest.raises(ManuscriptVariableError, match=domain):
            _validate_required_claim_values({label: {"value": bad_value}})

    def test_every_numeric_exposed_claim_is_classified_once(self):
        numeric = {
            label
            for label, entry in _real_claims().items()
            if isinstance(entry["value"], (int, float)) and not isinstance(entry["value"], bool)
        }
        classified = {label: _numeric_claim_domain(label) for label in numeric}
        assert set(classified) == numeric
        assert all(isinstance(domain, NumericClaimDomain) for domain in classified.values())
        assert classified["stage5_privileged_label_gain_pct"] is (
            NumericClaimDomain.SIGNED_RELATIVE_PERCENTAGE
        )
        assert classified["stage3_oov_gap_change_under_matching_pct"] is (
            NumericClaimDomain.NONNEGATIVE_RELATIVE_PERCENTAGE
        )
        assert classified["bench_calib_argmax_agreement_gemma3"] is (
            NumericClaimDomain.UNIT_INTERVAL
        )
        assert classified["bench_calib_measurable_gemma3"] is (NumericClaimDomain.BINARY_INDICATOR)
        assert classified["bench_letter_delta_gemma3"] is (NumericClaimDomain.SIGNED_UNIT_INTERVAL)

    def test_generated_numeric_table_rejects_markup_injection(self, tmp_path):
        claims = _full_claims()
        claims["bench_ctx_top1_k4_gemma3"]["value"] = "0.2 | injected"
        root = _write_project(tmp_path, claims)
        with pytest.raises(ManuscriptVariableError, match="numeric"):
            generate_manuscript_variables(root)

    @pytest.mark.parametrize("grid", ("0,16,4", "0,4,4", "4,16", "0,-1,4"))
    def test_context_grid_must_equal_the_frozen_protocol(self, tmp_path, grid):
        claims = _full_claims()
        claims["bench_ctx_k_grid"]["value"] = grid
        root = _write_project(tmp_path, claims)
        with pytest.raises(ManuscriptVariableError, match="bench_ctx_k_grid"):
            generate_manuscript_variables(root)

    def test_context_protocol_refuses_an_unregistered_grid_rung(self, tmp_path):
        claims = _full_claims()
        claims["bench_ctx_k_grid"]["value"] = "0,4,16,48,128,256"
        root = _write_project(tmp_path, claims)
        with pytest.raises(ManuscriptVariableError, match="bench_ctx_k_grid"):
            generate_manuscript_variables(root)

    def test_context_tables_are_compact_generated_views_of_exact_claims(self, tmp_path):
        claims = _full_claims()
        variables = generate_manuscript_variables(_write_project(tmp_path, claims))
        tables = variables["BENCH_CTX_TABLES"]

        for probe in SERVED_PROBES:
            assert f"**{probe.key} run**" in tables
            for k in SYNTHETIC_INCONTEXT_PROTOCOL.k_grid:
                expected = float(claims[f"bench_ctx_top1_k{k}_{probe.key}"]["value"])
                assert f"| {k} |" in tables
                assert f"{100 * expected:.2f}%" in tables
        assert variables["BENCH_CTX_UNIFORM_BASELINE_DISPLAY"] == (
            f"{100 * float(claims['bench_ctx_uniform_baseline']['value']):.2f}%"
        )
        assert variables["BENCH_CTX_TOP1_AILM_DISPLAY"] == (
            f"{100 * float(claims['bench_ctx_top1_ailm']['value']):.2f}%"
        )
        headline = variables["BENCH_CTX_HEADLINE_RESULT"]
        for probe in SERVED_PROBES:
            ratio = float(claims[f"bench_ctx_gap_closed_ratio_{probe.key}"]["value"])
            assert f"{probe.key} run" in headline
            assert f"{100 * ratio:.2f}%" in headline

        calibration = variables["BENCH_CALIB_TABLE_ROWS"]
        # Decision 93: the calibration table reports exactly the gate-admitted
        # systems. Gemma 3 4B and SmolLM are withheld, so their rows are gone
        # even though their figures stay in the ledger.
        assert calibration == (
            "| AILM | 30.50% | 26.31% | 6.53% | 0.850 | 2.450 |\n"
            "| Qwen 2.5 3B | 15.50% | 88.93% | 73.43% | 1.532 | 8.329 |"
        )
        assert "| Gemma 3 4B |" not in calibration
        assert "| SmolLM 1.7B Base v0.2 |" not in calibration
        assert "0.2631106997578302" not in calibration

    def test_context_contract_refuses_a_drifted_example_unit(self, tmp_path):
        claims = _full_claims()
        claims["bench_ctx_example_window"]["value"] = (
            int(claims["bench_ctx_example_window"]["value"]) + 1
        )
        root = _write_project(tmp_path, claims)
        with pytest.raises(ManuscriptVariableError, match="bench_ctx_example_window"):
            generate_manuscript_variables(root)

    def test_context_headline_must_match_the_frozen_protocol(self, tmp_path):
        claims = _full_claims()
        claims["bench_ctx_headline_k"]["value"] = SYNTHETIC_INCONTEXT_PROTOCOL.k_grid[-2]
        root = _write_project(tmp_path, claims)
        with pytest.raises(ManuscriptVariableError, match="bench_ctx_headline_k"):
            generate_manuscript_variables(root)

    @pytest.mark.parametrize(
        ("label", "bad_value", "message"),
        (
            ("bench_ctx_family_size", 999, "family_size"),
            ("bench_ctx_family_alpha", 0.123, "family_alpha"),
            ("bench_ctx_k4_vs_zero_paired_p_gemma3", 2, "unit interval"),
            ("bench_ctx_k4_vs_zero_paired_b_gemma3", 999999, "paired population"),
            ("bench_ctx_k128_vs_ailm_delta_gemma3", 999, "signed unit interval"),
            ("bench_ctx_k128_vs_ailm_delta_low_gemma3", 10, "signed unit interval"),
            ("bench_ctx_k128_vs_ailm_delta_high_gemma3", -10, "signed unit interval"),
            ("bench_ctx_gap_closed_ratio_gemma3", 999, "inconsistent"),
        ),
    )
    def test_context_family_is_recomputed_from_counts_and_protocol(
        self, tmp_path, label, bad_value, message
    ):
        claims = _full_claims()
        claims[label]["value"] = bad_value
        root = _write_project(tmp_path, claims)
        with pytest.raises(ManuscriptVariableError, match=message):
            generate_manuscript_variables(root)

    def test_context_hydration_rejects_an_impossible_paired_table(self, tmp_path):
        """Equal marginals cannot hide discordances larger than their successes."""

        claims = _full_claims()
        probe = SERVED_PROBES[0].key
        rung = next(k for k in SYNTHETIC_INCONTEXT_PROTOCOL.k_grid if k)
        total = int(claims["bench_ctx_positions"]["value"])
        successes = round(float(claims[f"bench_ctx_top1_k{rung}_{probe}"]["value"]) * total)
        for cell in ("b", "c"):
            claims[f"bench_ctx_k{rung}_vs_zero_paired_{cell}_{probe}"]["value"] = successes + 1
        claims[f"bench_ctx_k{rung}_vs_zero_paired_p_{probe}"]["value"] = 1.0

        with pytest.raises(ManuscriptVariableError, match="negative concordant"):
            generate_manuscript_variables(_write_project(tmp_path, claims))

    def test_context_hydration_rejects_successes_plus_unparsable_overflow(self, tmp_path):
        """An invalid reply cannot also occupy a recorded success cell."""

        claims = _full_claims()
        probe = SERVED_PROBES[0].key
        rung = next(k for k in SYNTHETIC_INCONTEXT_PROTOCOL.k_grid if k)
        total = int(claims["bench_ctx_positions"]["value"])
        successes = round(float(claims[f"bench_ctx_top1_k{rung}_{probe}"]["value"]) * total)
        claims[f"bench_ctx_unparsable_k{rung}_{probe}"]["value"] = total - successes + 1

        with pytest.raises(ManuscriptVariableError, match="successes plus unparsable"):
            generate_manuscript_variables(_write_project(tmp_path, claims))

    def test_context_hydration_rejects_seeded_draw_diagnostic_drift(self, tmp_path):
        """Conditioning diagnostics must rederive from the frozen draw protocol."""

        claims = _full_claims()
        rung = next(k for k in SYNTHETIC_INCONTEXT_PROTOCOL.k_grid if k)
        label = f"bench_ctx_draw_retries_k{rung}"
        claims[label]["value"] = int(claims[label]["value"]) + 1

        with pytest.raises(ManuscriptVariableError, match="does not reproduce"):
            generate_manuscript_variables(_write_project(tmp_path, claims))

    def test_benchmark_family_must_divide_exactly_across_probes(self, tmp_path):
        claims = _full_claims()
        claims["bench_family_size"]["value"] = 11
        root = _write_project(tmp_path, claims)
        with pytest.raises(ManuscriptVariableError, match="divide exactly"):
            generate_manuscript_variables(root)

    def test_lexical_best_alpha_must_be_a_recorded_sweep_coordinate(self, tmp_path):
        claims = _full_claims()
        claims["lexical_best_alpha"]["value"] = 0.35
        root = _write_project(tmp_path, claims)
        with pytest.raises(ManuscriptVariableError, match="not a recorded sweep coordinate"):
            generate_manuscript_variables(root)

    def test_lexical_sweep_requires_every_canonical_rung(self, tmp_path):
        claims = _full_claims()
        del claims["lexical_sweep_alpha_0_2_accuracy"]
        root = _write_project(tmp_path, claims)
        with pytest.raises(MissingClaimError, match="lexical_sweep_alpha_0_2_accuracy"):
            generate_manuscript_variables(root)

    def test_lexical_sweep_refuses_an_undeclared_extra_rung(self, tmp_path):
        claims = _full_claims()
        for suffix, value in (("accuracy", 1.0), ("delta", 0.0), ("scrambled", 1.0)):
            claims[f"lexical_sweep_alpha_0_4_{suffix}"] = {
                "value": value,
                "run_id": "r",
                "source": "s",
            }
        root = _write_project(tmp_path, claims)
        with pytest.raises(ManuscriptVariableError, match="canonical protocol"):
            generate_manuscript_variables(root)

    def test_lexical_high_alpha_must_be_the_final_sweep_coordinate(self, tmp_path):
        claims = _full_claims()
        claims["lexical_high_alpha"]["value"] = 0.8
        root = _write_project(tmp_path, claims)
        with pytest.raises(ManuscriptVariableError, match="maximum recorded sweep coordinate"):
            generate_manuscript_variables(root)

    @pytest.mark.parametrize(
        ("label", "bad_value", "message"),
        (
            ("lexical_best_alpha", 0.2, "unique accuracy argmax"),
            ("lexical_sweep_alpha_0_3_delta", 999, "inconsistent"),
            ("lexical_sweep_alpha_0_3_mcnemar_p", 2, "unit interval"),
            ("lexical_sweep_alpha_0_3_mcnemar_b", 999999, "impossible"),
            ("lexical_best_mcnemar_b", 1599, "inconsistent"),
            ("lexical_optimum_is_interior", 0, "disagrees"),
            ("lexical_high_alpha_accuracy", 999, "percentage"),
            ("lexical_high_alpha_margin_pts", 999, "inconsistent"),
        ),
    )
    def test_lexical_summary_is_recomputed_from_the_complete_sweep(
        self, tmp_path, label, bad_value, message
    ):
        claims = _full_claims()
        claims[label]["value"] = bad_value
        root = _write_project(tmp_path, claims)
        with pytest.raises(ManuscriptVariableError, match=message):
            generate_manuscript_variables(root)

    @pytest.mark.parametrize("label", ("params_levels", "params_gnn_specs"))
    def test_parameter_claim_count_must_match_primary_gnn_registry(self, tmp_path, label):
        claims = _full_claims()
        claims[label]["value"] = 999
        root = _write_project(tmp_path, claims)
        with pytest.raises(ManuscriptVariableError, match="GNN_SPEC_FILES"):
            generate_manuscript_variables(root)

    @pytest.mark.parametrize("extra_key", ("BAD", "L6_extra"))
    def test_primary_gnn_registry_cannot_outgrow_its_generated_table(self, tmp_path, extra_key):
        from ailm.parameters import GNN_SPEC_FILES

        root = _write_project(tmp_path, _full_claims())
        GNN_SPEC_FILES[extra_key] = "gnn/models/extra.gnn.md"
        try:
            with pytest.raises(ManuscriptVariableError, match="parameter-table inventory"):
                generate_manuscript_variables(root)
        finally:
            del GNN_SPEC_FILES[extra_key]


# ── Against the real ledger ──────────────────────────────────────────────────


class TestRealLedger:
    """The shipped ledger must actually satisfy the manuscript's requirements."""

    @_SKIP_NO_LEDGER
    def test_real_project_generates_variables(self):
        variables = generate_manuscript_variables(PROJECT_ROOT)
        assert len(variables) > 50

    @_SKIP_NO_LEDGER
    def test_stage3_scale_rise_token_has_exactly_one_ledger_owned_value(self):
        variables = generate_manuscript_variables(PROJECT_ROOT)
        recorded = _real_claims()["stage3_scale_gain_rise_pct"]["value"]
        assert variables["STAGE3_SCALE_GAIN_RISE_PCT"] == _format_value(recorded)

    @_SKIP_NO_LEDGER
    def test_subsample_stride_tokens_share_the_recorders_configuration(self):
        variables = generate_manuscript_variables(PROJECT_ROOT)
        assert variables["SUBSAMPLE_MATCHED_STRIDE"] == str(SUBSAMPLE_MATCHED_STRIDE)
        assert variables["SUBSAMPLE_LARGER_STRIDE"] == str(SUBSAMPLE_LARGER_STRIDE)
        assert SUBSAMPLE_MATCHED_ARM == f"stride{SUBSAMPLE_MATCHED_STRIDE}"
        assert SUBSAMPLE_LARGER_ARM == f"stride{SUBSAMPLE_LARGER_STRIDE}"
        assert SUBSAMPLE_STRIDE_BY_ARM == {
            SUBSAMPLE_MATCHED_ARM: SUBSAMPLE_MATCHED_STRIDE,
            SUBSAMPLE_LARGER_ARM: SUBSAMPLE_LARGER_STRIDE,
        }

    @_SKIP_NO_LEDGER
    def test_real_ledger_satisfies_every_required_claim(self):
        """If this fails the manuscript cannot render — that is the point."""
        generate_manuscript_variables(PROJECT_ROOT)  # raises if any is missing

    @_SKIP_NO_LEDGER
    def test_vendor_checksums_are_present_and_well_formed(self):
        variables = generate_manuscript_variables(PROJECT_ROOT)
        artifacts = (
            ("brown_counts.json", "META_BROWN_COUNTS"),
            ("cmudict_counts.json", "META_CMUDICT_COUNTS"),
            ("brown_word_counts.json", "META_BROWN_WORD_COUNTS"),
            ("ud_ewt_case_counts.json", "META_UD_CASE_COUNTS"),
        )
        for artifact, prefix in artifacts:
            key = f"{prefix}_SHA256"
            assert len(variables[key]) == 64
            assert variables[f"{prefix}_SHA256_SHORT"] == variables[key][:DIGEST_SHORT_LENGTH]
            expected_kb = str(round((PROJECT_ROOT / "vendor" / artifact).stat().st_size / 1000))
            assert variables[f"{prefix}_KB"] == expected_kb

    def test_vendor_kb_is_metric_not_a_mislabeled_kib(self, tmp_path):
        vendor = tmp_path / "vendor"
        vendor.mkdir()
        for name in (
            "brown_counts.json",
            "cmudict_counts.json",
            "brown_word_counts.json",
            "ud_ewt_case_counts.json",
        ):
            (vendor / name).write_bytes(b"x" * 1500)
        variables = _vendor_metadata(tmp_path)
        assert variables["META_BROWN_COUNTS_KB"] == "2"
        assert variables["META_BROWN_COUNTS_KB"] != str(round(1500 / 1024))

    @_SKIP_NO_LEDGER
    def test_generation_is_deterministic(self, monkeypatch):
        monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
        assert generate_manuscript_variables(PROJECT_ROOT) == generate_manuscript_variables(
            PROJECT_ROOT
        )

    @_SKIP_NO_LEDGER
    def test_no_token_value_is_empty(self):
        """An empty token renders as nothing and silently drops a claim from prose."""
        variables = generate_manuscript_variables(PROJECT_ROOT)
        empty = [
            k for k, v in variables.items() if not v.strip() and not k.endswith(("_RUN", "_SOURCE"))
        ]
        assert empty == [], f"empty token values: {empty}"


# ── Derived detectability for the arms the recorder left without power ───────


class TestDerivedPairedPower:
    """Power derived at render time must be the recorder's number, not a lookalike.

    The alphabetic-only letter, prompt-format and context-window arms recorded
    their discordant counts but no power claim, so the supplement could state
    each contrast without stating whether the pair count could have detected it.
    Power is a pure function of ``b`` and ``c``, so it is derived here rather
    than remeasured.

    The defect that makes this worth guarding: a derived quantity that merely
    *looks* like the recorded one. If the derivation used a different alpha, a
    different reference share, or the observed share instead of the reference
    share, every derived figure would still be a plausible probability, the
    render would still succeed, and the manuscript would report detectability
    computed one way beside detectability computed another. The equivalence test
    below rules that out by running the derivation on the arms that *do* have a
    recorded power and demanding it reproduce them.
    """

    @_SKIP_NO_LEDGER
    @pytest.mark.parametrize("arm", _RECORDED_POWER_ARMS)
    @pytest.mark.parametrize("model", _BENCH_MODELS)
    def test_derivation_reproduces_recorded_power_of_the_main_arms(self, arm, model):
        """EQUIVALENCE PROOF: derived path == recorder path on the recorded arms.

        These arm-probe pairs are the only place the two paths can be compared,
        because they are the only ones where the recorder wrote a power claim.
        Agreement here is what licenses trusting the derived figures for the three
        arms where no recorded value exists to check against. The count is
        parametrised over the registry rather than written out, so retiring a probe
        narrows the comparison instead of silently skipping a pair.

        A failure here does not mean the tolerance is too tight. It means the
        recorder computes power from something other than b and c, and deriving
        power from b and c alone is the wrong derivation.
        """
        claims = _real_claims()
        b = claims[f"{arm}_paired_b_{model}"]["value"]
        c = claims[f"{arm}_paired_c_{model}"]["value"]
        recorded = claims[f"{arm}_power_{model}"]["value"]
        assert paired_power_from_counts(b, c) == pytest.approx(recorded, rel=1e-12, abs=1e-12)

    def test_derived_and_recorded_power_render_to_the_same_text(self, tmp_path):
        """Equivalence again, at the token layer rather than the float layer.

        The prose reads formatted strings, not floats. Two numbers can agree to
        1e-12 and still render differently if the derived path skipped
        ``_format_value``, which would make the supplement's derived figures
        typographically distinguishable from its recorded ones.
        """
        claims = _full_claims()
        variables = generate_manuscript_variables(_write_project(tmp_path, claims))
        recording_machine = claims["env_machine"]["value"]
        for arm in _RECORDED_POWER_ARMS:
            for model in _BENCH_MODELS:
                b = claims[f"{arm}_paired_b_{model}"]["value"]
                c = claims[f"{arm}_paired_c_{model}"]["value"]
                recorded_value = claims[f"{arm}_power_{model}"]["value"]
                derived_value = paired_power_from_counts(b, c)
                derived_text = _format_value(derived_value)
                recorded_text = variables[f"{arm}_power_{model}".upper()]
                # The recorded path must render through the shared formatter.
                assert recorded_text == _format_value(recorded_value)
                if derived_value != recorded_value:
                    # Another CPU architecture can move the last bit of a
                    # closed-form float (decision 131 records the same drift for
                    # analytic protocol fields). The float-layer test above bounds
                    # it, and the shortest round-trip representations of two
                    # different floats cannot agree, so off the recording
                    # architecture this is a cross-architecture difference, not a
                    # formatter defect. On the recording architecture it remains
                    # a failure: the derivation changed.
                    assert derived_value == pytest.approx(recorded_value, rel=1e-12, abs=1e-12)
                    assert platform.machine() != recording_machine, (
                        f"{arm}/{model}: derived {derived_value!r} differs from recorded "
                        f"{recorded_value!r} on the recording architecture"
                    )
                    continue
                assert derived_text == recorded_text, (
                    f"{arm}/{model}: derived {derived_text!r} renders differently from "
                    f"recorded {recorded_text!r}"
                )

    def test_derivation_is_not_constant(self):
        """POSITIVE CONTROL: a derivation that ignores its inputs would pass everything.

        ``paired_power_from_counts`` returning one fixed number would satisfy no
        assertion above except by coincidence, but a stub returning 0.0 for every
        input would sail through a test that only checked the range. Different
        pair counts must give different answers, and more pairs must buy power.
        """
        by_pairs = {n: paired_power_from_counts(n // 2, n - n // 2) for n in (2, 20, 100, 400)}
        assert len(set(by_pairs.values())) == len(by_pairs), (
            f"the derivation does not distinguish these pair counts: {by_pairs}"
        )
        assert by_pairs[400] > by_pairs[100] > by_pairs[20] > by_pairs[2]

    def test_derived_tokens_track_the_ledger_counts(self, tmp_path):
        """POSITIVE CONTROL on the wiring, not just on the function.

        The function can be correct while the token block reads the wrong labels
        or hard-codes a value. Changing only the recorded counts must change only
        the corresponding derived token.
        """
        claims = _full_claims()
        baseline = generate_manuscript_variables(_write_project(tmp_path / "before", claims))

        claims["bench_letter_ctx_paired_b_gemma3"] = {"value": 600, "run_id": "r", "source": "s"}
        claims["bench_letter_ctx_paired_c_gemma3"] = {"value": 400, "run_id": "r", "source": "s"}
        bumped = generate_manuscript_variables(_write_project(tmp_path / "after", claims))

        assert float(bumped["BENCH_LETTER_CTX_POWER_GEMMA3"]) > float(
            baseline["BENCH_LETTER_CTX_POWER_GEMMA3"]
        ), "raising the discordant count did not raise the derived power"
        assert (
            bumped["BENCH_LETTER_CTX_POWER_QWEN25"] == baseline["BENCH_LETTER_CTX_POWER_QWEN25"]
        ), "a different model's derived power moved; the token block reads the wrong labels"

    @_SKIP_NO_LEDGER
    def test_the_derived_arms_really_lack_a_recorded_power_claim(self):
        """The premise of this whole block, asserted rather than assumed.

        If a recorder later starts writing ``bench_letter_ctx_power_*``, the
        derived token becomes a second producer of the same quantity and the two
        can drift. This fails at that moment so the derivation is deleted rather
        than left to shadow the measurement.
        """
        claims = _real_claims()
        checked = 0
        for arm in DERIVED_POWER_ARM_PREFIXES:
            for model in _BENCH_MODELS:
                assert f"{arm}_paired_b_{model}" in claims
                assert f"{arm}_paired_c_{model}" in claims
                assert f"{arm}_power_{model}" not in claims, (
                    f"{arm}_power_{model} is now recorded; delete the derived token "
                    "and let the recorder be the single producer"
                )
                checked += 1
        expected = len(DERIVED_POWER_ARM_PREFIXES) * len(_BENCH_MODELS)
        assert checked == expected, (
            f"scanned {checked} arm/model pairs, expected {expected}; the "
            f"inventory is incomplete and this scan is partial"
        )
        assert expected >= 4, "too few arm/model pairs for this scan to constrain anything"

    def test_every_derived_power_token_is_a_usable_probability(self, tmp_path):
        """A derived token must resolve to a number the supplement can print."""
        variables = generate_manuscript_variables(_write_project(tmp_path, _full_claims()))
        for arm in DERIVED_POWER_ARM_PREFIXES:
            for model in _BENCH_MODELS:
                token = f"{arm}_power_{model}".upper()
                assert token in variables, f"{token} is not emitted"
                value = float(variables[token])
                assert 0.0 < value < 1.0, f"{token} = {value} is not a power"

    @_SKIP_NO_LEDGER
    def test_no_derived_power_token_shadows_a_ledger_label(self):
        """A derived name matching a ledger label would overwrite the measurement.

        ``generate_manuscript_variables`` writes ledger tokens first and derived
        tokens second, so a collision silently replaces a recorded number with a
        recomputed one. ``tests/test_benchmarks.py`` scans the module source for
        this; this checks the same property from the name inventory, which stays
        true even if the assignments are ever restructured.
        """
        labels = {label.lower() for label in _real_claims()}
        assert labels, "no ledger labels loaded; this scan would be vacuous"
        collisions = sorted(
            f"{arm}_power_{model}"
            for arm in DERIVED_POWER_ARM_PREFIXES
            for model in _BENCH_MODELS
            if f"{arm}_power_{model}" in labels
        )
        assert collisions == [], f"derived tokens collide with ledger labels: {collisions}"

    def test_fractional_count_is_rejected(self):
        """A rate read where a count was meant must not be truncated into an answer.

        ``bench_letter_ctx_paired_b_gemma3`` and a rate label differ by a few
        characters. Truncating 0.27 to 0 would return a power for a pair count
        nobody measured, and nothing downstream could tell.
        """
        with pytest.raises(ValueError, match="whole number"):
            paired_power_from_counts(2.5, 10)
        with pytest.raises(ValueError, match="whole number"):
            paired_power_from_counts(10, 0.27)


@pytest.mark.parametrize(
    "label",
    ["bench_synth_warmup_policy", "bench_synthphone_warmup_policy", "bench_ctx_warmup_policy"],
)
def test_hosted_manuscript_refuses_legacy_local_warmup_metadata(tmp_path, label):
    claims = _full_claims()
    claims[label]["value"] = "uncached-synthetic-contract-v1"
    with pytest.raises(ManuscriptVariableError, match=label):
        generate_manuscript_variables(_write_project(tmp_path, claims))


@pytest.mark.parametrize("bad", [True, 0, -1, 10**1000, float("nan"), float("inf"), "9.5", None])
def test_invalid_rendering_typography_is_rejected(tmp_path, bad):
    config = _valid_config()
    config["rendering"]["body_font_pt"] = bad
    _write_project(tmp_path, _real_claims(), config)
    from ailm.manuscript_variables.literature import _load_config

    with pytest.raises(ManuscriptVariableError, match=r"rendering\.body_font_pt\b"):
        _load_config(tmp_path)


def test_reference_links_follow_canonical_physical_pages(tmp_path):
    payload = json.loads((PROJECT_ROOT / "data/reference_sources.json").read_text())
    source = payload["sources"]["cerebrum"]
    source["pdf_evidence"]["pomdp_tuple"]["physical_page"] += 1
    (tmp_path / "data").mkdir()
    (tmp_path / "data/reference_sources.json").write_text(json.dumps(payload))
    values = _load_reference_variables(tmp_path)
    assert values["CEREBRUM_POMDP_SUPPLEMENT_URL"] == (
        source["deposited_file"]["download_url"]
        + "#page="
        + str(source["pdf_evidence"]["pomdp_tuple"]["physical_page"])
    )


@pytest.mark.parametrize("bad", [True, 0, -1, 147, "90", None])
def test_reference_link_rejects_invalid_physical_page(tmp_path, bad):
    payload = json.loads((PROJECT_ROOT / "data/reference_sources.json").read_text())
    payload["sources"]["cerebrum"]["pdf_evidence"]["pomdp_tuple"]["physical_page"] = bad
    (tmp_path / "data").mkdir()
    (tmp_path / "data/reference_sources.json").write_text(json.dumps(payload))
    with pytest.raises(InvalidReferenceSourceError, match="physical page"):
        _load_reference_variables(tmp_path)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("record_id", True),
        ("record_id", 0),
        ("record_id", "15231156"),
        ("filename", "other.pdf"),
        ("filename", "../source.pdf"),
        ("download_url", "https://example.org/source.pdf"),
        (
            "download_url",
            "https://zenodo.org/api/records/15231156/files/CEREBRUM_1-4_April-16-2025.pdf/content#page=90",
        ),
    ],
)
def test_reference_link_rejects_conflicting_source_identity(tmp_path, field, bad):
    payload = json.loads((PROJECT_ROOT / "data/reference_sources.json").read_text())
    source = payload["sources"]["cerebrum"]
    mapping = source["zenodo_record"] if field == "record_id" else source["deposited_file"]
    mapping[field] = bad
    (tmp_path / "data").mkdir()
    (tmp_path / "data/reference_sources.json").write_text(json.dumps(payload))
    with pytest.raises(InvalidReferenceSourceError, match="deposited PDF URL"):
        _load_reference_variables(tmp_path)


def test_rendering_configuration_drives_produced_tokens(tmp_path):
    from ailm.manuscript_variables.literature import _load_rendering_variables

    config = _valid_config()
    config["rendering"]["margin_right_in"] = 0.75
    _write_project(tmp_path, _real_claims(), config)
    variables = _load_rendering_variables(tmp_path)
    assert variables["META_MARGIN_RIGHT_IN"] == "0.75"
    assert variables["META_BODY_FONT_PT"] == str(config["rendering"]["body_font_pt"])


@pytest.mark.parametrize("rendering", [None, {}, {"body_font_pt": 9.5}])
def test_absent_or_partial_rendering_configuration_refuses(tmp_path, rendering):
    from ailm.manuscript_variables.literature import _load_rendering_variables

    config = _valid_config()
    config["rendering"] = rendering
    _write_project(tmp_path, _real_claims(), config)
    with pytest.raises(ManuscriptVariableError, match="rendering"):
        _load_rendering_variables(tmp_path)


def test_unsuitable_rendering_baseline_refuses(tmp_path):
    from ailm.manuscript_variables.literature import _load_rendering_variables

    config = _valid_config()
    config["rendering"]["body_baseline_pt"] = 10.0
    _write_project(tmp_path, _real_claims(), config)
    with pytest.raises(ManuscriptVariableError, match="baseline must suit"):
        _load_rendering_variables(tmp_path)


def test_default_smoothing_tokens_follow_the_actual_level_configuration():
    from ailm.config import DEFAULT_LEVEL_CONFIG

    variables = generate_manuscript_variables(PROJECT_ROOT)
    configuration = DEFAULT_LEVEL_CONFIG
    assert float(variables["MODEL_DEFAULT_SMOOTHING"]) == configuration.smoothing
    assert (
        float(variables["MODEL_DEFAULT_EMISSION_SMOOTHING"])
        == configuration.resolved_emission_smoothing
    )
