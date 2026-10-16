"""Exact deposited-source identity and attribution guards for the case scaffold.

The source PDFs are not vendored, so ``data/reference_sources.json`` is the
always-on snapshot of the exact Zenodo records and the inspected PDF locators.
These tests bind that snapshot to code, manuscript variables, bibliography, and
current public documentation. A network-free checkout can therefore reject the
same Figure-13 and ``[GEN]`` conflations that motivated the snapshot.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ailm.cases import (
    AILM_PYMDP_HEURISTIC_CROSSWALK,
    CEREBRUM_CASE_FORMULATIONS,
    CEREBRUM_FUNCTION_ABBREVIATIONS,
    CEREBRUM_POMDP_SYMBOLS,
    PYMDP_COMPONENTS_WITHOUT_CEREBRUM_TUPLE_ANALOGUE,
    CaseRole,
    CerebrumFunctionRole,
)
from ailm.manuscript_variables import _load_reference_variables

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = PROJECT_ROOT / "data" / "reference_sources.json"
BIBLIOGRAPHY_PATH = PROJECT_ROOT / "docs" / "manuscript" / "references.bib"

_SOURCE_CASE_CONTRACT = {
    "NOM": {
        "kind": "likelihood_weight",
        "pages": (122,),
        "form": "likelihood term weighted by alpha_NOM",
        "source_auxiliary_symbol": None,
        "source_downstream_observation_symbol": None,
        "precision_relation": "alpha_NOM > 1",
    },
    "ACC": {
        "kind": "complexity_weight",
        "pages": (123,),
        "form": "complexity term weighted by beta_ACC",
        "source_auxiliary_symbol": None,
        "source_downstream_observation_symbol": None,
        "precision_relation": "beta_ACC < 1",
    },
    "GEN": {
        "kind": "augmented_latent",
        "pages": (124,),
        "form": "joint latent state (s, r)",
        "source_auxiliary_symbol": "r",
        "source_downstream_observation_symbol": None,
        "precision_relation": "pi_GEN(r) > pi_GEN(s)",
    },
    "DAT": {
        "kind": "augmented_latent",
        "pages": (125,),
        "form": "joint latent state (s, g)",
        "source_auxiliary_symbol": "g",
        "source_downstream_observation_symbol": None,
        "precision_relation": "pi_DAT(g) approximately equals pi_DAT(s)",
    },
    "INS": {
        "kind": "augmented_latent",
        "pages": (126,),
        "form": "joint latent state (s, a)",
        "source_auxiliary_symbol": "a",
        "source_downstream_observation_symbol": None,
        "precision_relation": "pi_INS(a) > pi_INS(s)",
    },
    "LOC": {
        "kind": "augmented_latent",
        "pages": (126, 127),
        "form": "joint latent state (s, e)",
        "source_auxiliary_symbol": "e",
        "source_downstream_observation_symbol": None,
        "precision_relation": "pi_LOC(e) > pi_LOC(s)",
    },
    "ABL": {
        "kind": "augmented_latent",
        "pages": (127, 128),
        "form": "joint latent state (s, o)",
        "source_auxiliary_symbol": "o",
        "source_downstream_observation_symbol": "o_prime",
        "precision_relation": "pi_ABL(o) > pi_ABL(s)",
    },
    "VOC": {
        "kind": "augmented_latent",
        "pages": (128,),
        "form": "joint latent state (s, c)",
        "source_auxiliary_symbol": "c",
        "source_downstream_observation_symbol": None,
        "precision_relation": "pi_VOC(c) > pi_VOC(s)",
    },
}


def _snapshot() -> dict:
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def _bibtex_entry(text: str, key: str) -> str:
    """Extract one brace-delimited BibTeX entry without a BibTeX dependency."""
    marker = f"@techreport{{{key},"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"missing BibTeX entry {key!r}")
    opening = text.index("{", start)
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError(f"unterminated BibTeX entry {key!r}")


def _bibtex_braced_field(entry: str, field: str) -> str:
    """Read a nested-brace field value from an already isolated BibTeX entry."""
    match = re.search(rf"(?m)^\s*{re.escape(field)}\s*=\s*", entry)
    if match is None or match.end() >= len(entry) or entry[match.end()] != "{":
        raise ValueError(f"missing braced BibTeX field {field!r}")
    opening = match.end()
    depth = 0
    for index in range(opening, len(entry)):
        if entry[index] == "{":
            depth += 1
        elif entry[index] == "}":
            depth -= 1
            if depth == 0:
                return entry[opening + 1 : index]
    raise ValueError(f"unterminated BibTeX field {field!r}")


def _bibtex_plain(value: str) -> str:
    """Normalize TeX case-protection braces and layout, not substantive text."""
    return " ".join(value.replace(r"\_", "_").replace("{", "").replace("}", "").split())


def _case_formulation_defects(recorded: object) -> list[str]:
    """Bind exact Supplement-11 transcriptions to every code-side formulation."""

    if not isinstance(recorded, dict) or set(recorded) != set(_SOURCE_CASE_CONTRACT):
        return ["case formulation role inventory"]

    defects: list[str] = []
    for role_name, expected_source in _SOURCE_CASE_CONTRACT.items():
        entry = recorded[role_name]
        if not isinstance(entry, dict):
            defects.append(f"{role_name} formulation object")
            continue
        pages = tuple(entry.get("physical_pages", (entry.get("physical_page"),)))
        actual_source = {
            "kind": expected_source["kind"],
            "pages": pages,
            "form": entry.get("form"),
            "source_auxiliary_symbol": entry.get("source_auxiliary_symbol"),
            "source_downstream_observation_symbol": entry.get(
                "source_downstream_observation_symbol"
            ),
            "precision_relation": entry.get("precision_relation"),
        }
        if actual_source != expected_source:
            defects.append(f"{role_name} exact source formulation")

        role = CaseRole[role_name]
        formulation = CEREBRUM_CASE_FORMULATIONS[role]
        actual_ailm = {
            "kind": formulation.kind.value,
            "auxiliary_symbol": formulation.auxiliary_symbol,
            "precision_relation": formulation.precision_relation,
        }
        expected_ailm = {
            "kind": expected_source["kind"],
            "auxiliary_symbol": entry.get("ailm_auxiliary_symbol"),
            "precision_relation": entry.get("ailm_precision_relation"),
        }
        if actual_ailm != expected_ailm:
            defects.append(f"{role_name} AILM formulation crosswalk")
    return defects


def _bibliography_defects(text: str, payload: dict) -> list[str]:
    """Bind both citations to record identity and disclosed printed identity."""
    defects: list[str] = []
    keys = {"cerebrum": "friedman2024cerebrum", "compositional_case": "friedman2024ccd"}
    for source_key, bib_key in keys.items():
        source = payload["sources"][source_key]
        record = source["zenodo_record"]
        deposited = source["deposited_file"]
        printed = source["printed_identity"]
        try:
            entry = _bibtex_entry(text, bib_key)
            title = _bibtex_plain(_bibtex_braced_field(entry, "title"))
            doi = _bibtex_plain(_bibtex_braced_field(entry, "doi"))
            year = _bibtex_plain(_bibtex_braced_field(entry, "year"))
            note = _bibtex_plain(_bibtex_braced_field(entry, "note"))
        except ValueError as exc:
            defects.append(str(exc))
            continue

        if title != record["title"]:
            defects.append(f"{source_key} record title")
        if doi != record["record_doi"]:
            defects.append(f"{source_key} record DOI")
        if year != record["publication_date"][:4]:
            defects.append(f"{source_key} record year")
        if f"Zenodo record version {record['version']}" not in note:
            defects.append(f"{source_key} record version")
        if record["publication_date"] not in note:
            defects.append(f"{source_key} record date")
        if deposited["filename"] not in note:
            defects.append(f"{source_key} deposited filename")
        if printed["title"] not in note:
            defects.append(f"{source_key} printed title")
        if printed["date"] not in note:
            defects.append(f"{source_key} printed date")
        if printed["version"] is None:
            if "no printed version" not in note:
                defects.append(f"{source_key} absent printed version")
        elif f"Version {printed['version']}" not in note:
            defects.append(f"{source_key} printed version")
    return defects


def _snapshot_defects(payload: dict) -> list[str]:
    """Independent high-signal validation of the source-identity snapshot."""
    defects: list[str] = []
    if payload.get("_schema") != "ailm-reference-sources-v1":
        defects.append("schema")
        return defects
    sources = payload.get("sources")
    if not isinstance(sources, dict) or set(sources) != {"cerebrum", "compositional_case"}:
        defects.append("source inventory")
        return defects

    for key in ("cerebrum", "compositional_case"):
        source = sources[key]
        record = source["zenodo_record"]
        deposited = source["deposited_file"]
        printed = source["printed_identity"]
        if record["record_doi"] == record["concept_doi"]:
            defects.append(f"{key} concept/version DOI collapse")
        if not re.fullmatch(r"[0-9a-f]{64}", deposited["sha256"]):
            defects.append(f"{key} sha256")
        if not re.fullmatch(r"[0-9a-f]{32}", deposited["md5"]):
            defects.append(f"{key} md5")
        if deposited["size_bytes"] <= 0 or deposited["pdf_pages"] <= 0:
            defects.append(f"{key} extent")
        if printed["title"] == record["title"] and printed.get("version") == record["version"]:
            defects.append(f"{key} identity conflict erased")

    evidence = sources["cerebrum"]["pdf_evidence"]
    if evidence.get("core_case_formulations_locator") != "Supplement 11":
        defects.append("Supplement 11 case formulations")
    defects.extend(_case_formulation_defects(evidence.get("core_case_formulations")))
    if evidence["novel_generative_case"].get("locator") != "Supplement 2.12":
        defects.append("Supplement 2.12 generative role")
    figure = evidence["active_inference_figure"]
    if figure != {
        "physical_page": 21,
        "locator": "Figure 13",
        "caption": "Figure 13: Active Inference Integration Framework.",
        "content_kind": "caption_only",
        "provides_symbol_crosswalk": False,
    }:
        defects.append("Figure 13 evidence")
    tuple_evidence = evidence["pomdp_tuple"]
    if tuple_evidence.get("locator") != "Supplement 8.1":
        defects.append("Supplement 8.1 locator")
    if tuple(tuple_evidence["symbols"]) != ("S", "A", "T", "Ω", "O", "F", "π"):
        defects.append("Supplement 8.1 tuple")
    if evidence["case_transition_free_energy"]["kl_scaled"] is not False:
        defects.append("unscaled F_trans")
    if evidence["expected_free_energy_transition"] != {
        "physical_page": 98,
        "locator": "Supplement 8.8.3",
        "form": "G_transition = G_base + gamma * KL(q_target || q_source)",
        "kl_scaled": True,
        "scale_symbol": "γ",
    }:
        defects.append("gamma-scaled EFE transition")
    return defects


def test_reference_source_snapshot_is_complete_and_self_consistent() -> None:
    assert SNAPSHOT_PATH.is_file()
    assert _snapshot_defects(_snapshot()) == []


def test_snapshot_guard_rejects_erasing_the_cerebrum_version_conflict() -> None:
    """POSITIVE CONTROL: collapsing record and title-page versions must fail."""
    planted = json.loads(json.dumps(_snapshot()))
    cerebrum = planted["sources"]["cerebrum"]
    cerebrum["printed_identity"]["version"] = cerebrum["zenodo_record"]["version"]
    cerebrum["printed_identity"]["title"] = cerebrum["zenodo_record"]["title"]
    assert "cerebrum identity conflict erased" in _snapshot_defects(planted)


def test_exact_deposited_file_identities_are_pinned() -> None:
    sources = _snapshot()["sources"]
    assert sources["cerebrum"]["deposited_file"] == {
        "filename": "CEREBRUM_1-4_April-16-2025.pdf",
        "size_bytes": 1327184,
        "md5": "7b44c4f126b9acc2925196a667f68183",
        "sha256": "105121d737fa0b288c26a67d456bfea12c36764364a3b57c9ffe6b6adee01451",
        "pdf_pages": 146,
        "download_url": (
            "https://zenodo.org/api/records/15231156/files/CEREBRUM_1-4_April-16-2025.pdf/content"
        ),
    }
    assert sources["compositional_case"]["deposited_file"] == {
        "filename": "cognitive_case_diagrams_v1_DAF_04-23-2026.pdf",
        "size_bytes": 13601721,
        "md5": "8740f4b058d8ca2202c5fdccc9e4a5ce",
        "sha256": "3767c3c3090994c67bf9bbf5ddf47d38142a3c76ed859d56d36a052c6c9e284f",
        "pdf_pages": 113,
        "download_url": (
            "https://zenodo.org/api/records/19695260/files/"
            "cognitive_case_diagrams_v1_DAF_04-23-2026.pdf/content"
        ),
    }


def test_record_identity_and_printed_identity_are_both_retained() -> None:
    sources = _snapshot()["sources"]
    cerebrum = sources["cerebrum"]
    assert cerebrum["zenodo_record"]["version"] == "1.4"
    assert cerebrum["zenodo_record"]["publication_date"] == "2025-04-16"
    assert cerebrum["printed_identity"]["version"] == "1.2"
    assert cerebrum["printed_identity"]["date"] == "2025-04-12"

    categorical = sources["compositional_case"]
    assert (
        categorical["zenodo_record"]["title"]
        == "Compositional Approaches to Linguistic Case for Cognitive Modeling"
    )
    assert (
        categorical["printed_identity"]["title"]
        == "Cognitive Diagrams: Reviewing Categorical Accounts of Linguistic Case"
    )
    assert categorical["zenodo_record"]["publication_date"] == "2026-04-23"
    assert categorical["printed_identity"]["date"] == "2026-04-22"
    assert categorical["optional_companion_repository"] == {
        "url": "https://github.com/docxology/cognitive_case_diagrams",
        "revision": "ce0811b7da5fb38a365fb2bf441d6a534ecd3e91",
        "manuscript_title": "Cognitive Diagrams: Reviewing Categorical Accounts of Linguistic Case",
        "manuscript_version": "2.3",
        "manuscript_date": "2026-04-22",
    }


def test_code_crosswalk_matches_the_exact_supplement_tuple() -> None:
    tuple_evidence = _snapshot()["sources"]["cerebrum"]["pdf_evidence"]["pomdp_tuple"]
    assert tuple(CEREBRUM_POMDP_SYMBOLS) == tuple(tuple_evidence["symbols"])
    assert dict(CEREBRUM_POMDP_SYMBOLS) == tuple_evidence["meanings"]
    assert {
        symbol: (
            correspondence.source_symbol,
            correspondence.pymdp_component,
            correspondence.relation,
            correspondence.caveat,
        )
        for symbol, correspondence in AILM_PYMDP_HEURISTIC_CROSSWALK.items()
    } == {
        "S": ("S", "hidden-state factor domains", "domain correspondence", "AILM heuristic"),
        "A": (
            "A",
            "action and control-factor domains",
            "domain correspondence",
            "source A is action space; not pymdp likelihood A",
        ),
        "T": ("T", "B", "transition dynamics", "source T corresponds heuristically to pymdp B"),
        "Ω": (
            "Ω",
            "observation-modality domains",
            "domain correspondence",
            "AILM heuristic",
        ),
        "O": ("O", "A", "likelihood mapping", "source O corresponds heuristically to pymdp A"),
        "F": ("F", None, "variational-free-energy objective", "not pymdp G"),
        "π": ("π", None, "precision hyperparameters", "not pymdp E"),
    }
    assert PYMDP_COMPONENTS_WITHOUT_CEREBRUM_TUPLE_ANALOGUE == {"C", "D", "E"}


def test_case_formulation_inventory_matches_exact_pdf_evidence() -> None:
    evidence = _snapshot()["sources"]["cerebrum"]["pdf_evidence"]
    recorded = evidence["core_case_formulations"]
    assert _case_formulation_defects(recorded) == []


def test_case_formulation_guard_rejects_a_reversed_source_relation() -> None:
    """NEGATIVE CONTROL: a plausible-looking inequality reversal must fail."""

    recorded = json.loads(
        json.dumps(_snapshot()["sources"]["cerebrum"]["pdf_evidence"]["core_case_formulations"])
    )
    recorded["GEN"]["precision_relation"] = "pi_GEN(r) < pi_GEN(s)"
    assert "GEN exact source formulation" in _case_formulation_defects(recorded)


def test_case_formulation_guard_separates_source_o_from_ailm_o_origin() -> None:
    """NEGATIVE CONTROL: the project alias must never overwrite the PDF symbol."""

    recorded = json.loads(
        json.dumps(_snapshot()["sources"]["cerebrum"]["pdf_evidence"]["core_case_formulations"])
    )
    recorded["ABL"]["form"] = "joint latent state (s, o_origin)"
    recorded["ABL"]["source_auxiliary_symbol"] = "o_origin"
    recorded["ABL"]["precision_relation"] = "pi_ABL(o_origin) > pi_ABL(s)"
    assert "ABL exact source formulation" in _case_formulation_defects(recorded)


def test_source_gen_collision_is_retained_in_code_and_snapshot() -> None:
    evidence = _snapshot()["sources"]["cerebrum"]["pdf_evidence"]
    assert evidence["novel_generative_case"]["source_abbreviation"] == "GEN"
    gen_roles = {
        role
        for role, abbreviation in CEREBRUM_FUNCTION_ABBREVIATIONS.items()
        if abbreviation == "GEN"
    }
    assert gen_roles == {
        CerebrumFunctionRole.CORE_GENITIVE,
        CerebrumFunctionRole.NOVEL_GENERATIVE,
    }


def test_manuscript_reference_variables_come_directly_from_the_source_snapshot() -> None:
    variables = _load_reference_variables(PROJECT_ROOT)
    sources = _snapshot()["sources"]
    cerebrum = sources["cerebrum"]
    evidence = cerebrum["pdf_evidence"]
    expected = {
        "CEREBRUM_ZENODO_VERSION": cerebrum["zenodo_record"]["version"],
        "CEREBRUM_PRINTED_VERSION": cerebrum["printed_identity"]["version"],
        "CEREBRUM_ZENODO_DATE": cerebrum["zenodo_record"]["publication_date"],
        "CEREBRUM_PRINTED_DATE": cerebrum["printed_identity"]["date"],
        "CEREBRUM_PDF_PAGE_COUNT": str(cerebrum["deposited_file"]["pdf_pages"]),
        "CEREBRUM_PDF_FILENAME": cerebrum["deposited_file"]["filename"],
        "COMPOSITIONAL_CASE_PDF_PAGE_COUNT": str(
            sources["compositional_case"]["deposited_file"]["pdf_pages"]
        ),
        "CEREBRUM_CASE_SUPPLEMENT": evidence["core_case_formulations_locator"].split(" ", 1)[1],
    }
    for name, key in {
        "GENERATIVE_SUPPLEMENT": "novel_generative_case",
        "ACTIVE_INFERENCE_FIGURE": "active_inference_figure",
        "POMDP_SUPPLEMENT": "pomdp_tuple",
        "TRANSFORMATION_SUPPLEMENT": "case_transition_free_energy",
        "EFE_TRANSITION_SUPPLEMENT": "expected_free_energy_transition",
        "CORE_FUNCTION_TABLE": "core_function_table",
    }.items():
        expected[f"CEREBRUM_{name}"] = evidence[key]["locator"].split(" ", 1)[1]
    entries = {
        "GENERATIVE_SUPPLEMENT": evidence["novel_generative_case"],
        "ACTIVE_INFERENCE_FIGURE": evidence["active_inference_figure"],
        "POMDP_SUPPLEMENT": evidence["pomdp_tuple"],
        "TRANSFORMATION_SUPPLEMENT": evidence["case_transition_free_energy"],
        "EFE_TRANSITION_SUPPLEMENT": evidence["expected_free_energy_transition"],
        "CORE_FUNCTION_TABLE": evidence["core_function_table"],
        "CASE_SUPPLEMENT": evidence["core_case_formulations"]["NOM"],
        "GENITIVE_CASE": evidence["core_case_formulations"]["GEN"],
    }
    for name, entry in entries.items():
        pages = entry.get("physical_pages", [entry.get("physical_page")])
        expected[f"CEREBRUM_{name}_URL"] = (
            cerebrum["deposited_file"]["download_url"] + f"#page={pages[0]}"
        )
        expected[f"CEREBRUM_{name}_PAGES"] = ", ".join(map(str, pages))
    for prefix, key in (("CEREBRUM", "cerebrum"), ("COMPOSITIONAL_CASE", "compositional_case")):
        source = sources[key]
        for suffix, parent, field in (
            ("CONCEPT_DOI", "zenodo_record", "concept_doi"),
            ("RECORD_DOI", "zenodo_record", "record_doi"),
            ("RECORD_TITLE", "zenodo_record", "title"),
            ("PRINTED_TITLE", "printed_identity", "title"),
            ("PDF_FILENAME", "deposited_file", "filename"),
            ("PDF_MD5", "deposited_file", "md5"),
            ("PDF_SHA256", "deposited_file", "sha256"),
        ):
            expected[f"{prefix}_{suffix}"] = source[parent][field]
        expected[f"{prefix}_PDF_SIZE_BYTES"] = f"{source['deposited_file']['size_bytes']:,}"
    source = sources["compositional_case"]
    for suffix, parent, field in (
        ("ZENODO_VERSION", "zenodo_record", "version"),
        ("ZENODO_DATE", "zenodo_record", "publication_date"),
        ("PRINTED_DATE", "printed_identity", "date"),
        ("COMPANION_REVISION", "optional_companion_repository", "revision"),
        ("COMPANION_VERSION", "optional_companion_repository", "manuscript_version"),
        ("COMPANION_DATE", "optional_companion_repository", "manuscript_date"),
    ):
        expected[f"COMPOSITIONAL_CASE_{suffix}"] = source[parent][field]
    expected.update(
        {
            "PROTOCOL_P1_FIT_EVALUATION_URL": "ailm-supporting-docs/docs__extension_protocols.html#protocol-p1-fit-evaluation",
            "PROTOCOL_P2_ALIGNED_CORRESPONDENCE_URL": "ailm-supporting-docs/docs__extension_protocols.html#protocol-p2-aligned-correspondence",
            "PROTOCOL_P3_BRIDGE_URL": "ailm-supporting-docs/docs__extension_protocols.html#protocol-p3-bridge",
        }
    )
    assert variables == expected


def test_bibliography_binds_record_and_printed_pdf_identities() -> None:
    assert _bibliography_defects(BIBLIOGRAPHY_PATH.read_text(encoding="utf-8"), _snapshot()) == []


@pytest.mark.parametrize(
    ("old", "new", "expected_defect"),
    (
        (
            "10.5281/zenodo.15231156",
            "10.5281/zenodo.15170907",
            "cerebrum record DOI",
        ),
        (
            "Cognitive Diagrams: Reviewing Categorical\n"
            "                 Accounts of Linguistic Case",
            "Compositional Approaches to Linguistic\n                 Case for Cognitive Modeling",
            "compositional_case printed title",
        ),
    ),
)
def test_bibliography_identity_guard_rejects_planted_mutation(
    old: str, new: str, expected_defect: str
) -> None:
    """POSITIVE CONTROL: a concept DOI or erased printed title must fail."""
    text = BIBLIOGRAPHY_PATH.read_text(encoding="utf-8")
    assert old in text
    planted = text.replace(old, new, 1)
    assert expected_defect in _bibliography_defects(planted, _snapshot())


def test_current_design_docs_preserve_record_versus_printed_identity() -> None:
    sources = _snapshot()["sources"]
    cerebrum = sources["cerebrum"]
    compositional = sources["compositional_case"]

    generation_doc = (PROJECT_ROOT / "docs/design/generation_problem.md").read_text(
        encoding="utf-8"
    )
    generation_plain = " ".join(generation_doc.split())
    assert f"version {cerebrum['zenodo_record']['version']}" in generation_plain
    assert f"version {cerebrum['printed_identity']['version']}" in generation_plain
    assert compositional["zenodo_record"]["title"] in generation_plain
    assert compositional["zenodo_record"]["version"] in generation_plain
    assert compositional["printed_identity"]["title"] in generation_plain
    assert "no version" in generation_plain

    category_doc = (PROJECT_ROOT / "docs/design/no_embeddings.md").read_text(encoding="utf-8")
    category_plain = " ".join(category_doc.split())
    assert compositional["zenodo_record"]["title"] in category_plain
    assert compositional["zenodo_record"]["record_doi"] in category_plain
    assert compositional["printed_identity"]["title"] in category_plain
    assert compositional["printed_identity"]["date"] in category_plain
    assert "no version" in category_plain


_CURRENT_ATTRIBUTION_PATHS = (
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "AGENTS.md",
    PROJECT_ROOT / "src" / "ailm" / "cases.py",
    PROJECT_ROOT / "src" / "ailm" / "AGENTS.md",
    PROJECT_ROOT / "docs" / "architecture.md",
    PROJECT_ROOT / "docs" / "design" / "cerebrum_mapping.md",
    PROJECT_ROOT / "docs" / "design" / "generation_problem.md",
    PROJECT_ROOT / "docs" / "design" / "no_embeddings.md",
    PROJECT_ROOT / "docs" / "reference" / "formalism_map.md",
    PROJECT_ROOT / "docs" / "provenance" / "sources.md",
    PROJECT_ROOT / "docs" / "manuscript" / "02_background.md",
    PROJECT_ROOT / "docs" / "manuscript" / "06_related_work.md",
    PROJECT_ROOT / "docs" / "manuscript" / "references.bib",
)
_RETIRED_ATTRIBUTIONS = (
    re.compile(r"CEREBRUM_PYMDP_MAPPING"),
    re.compile(r"CASE_FREE_ENERGY_PARAMS"),
    re.compile(r"CerebrumFunctionRole\.GENERATOR"),
    re.compile(r"derived from (?:CEREBRUM )?Figure (?:13|\{\{[^}]*FIGURE[^}]*\}\})", re.I),
    re.compile(
        r"Figure 13 (?:provides|was chosen|is the canonical) .*?(?:mapping|crosswalk)", re.I
    ),
    re.compile(r"CEREBRUM(?:→|-to-)pymdp mapping", re.I),
)


def _retired_attributions(text: str) -> list[str]:
    return [pattern.pattern for pattern in _RETIRED_ATTRIBUTIONS if pattern.search(text)]


def test_current_surfaces_contain_no_retired_source_attribution() -> None:
    defects = {
        str(path.relative_to(PROJECT_ROOT)): _retired_attributions(path.read_text(encoding="utf-8"))
        for path in _CURRENT_ATTRIBUTION_PATHS
    }
    assert not {path: matches for path, matches in defects.items() if matches}


@pytest.mark.parametrize(
    "planted",
    (
        "CEREBRUM_PYMDP_MAPPING maps pi to E.",
        "The crosswalk was derived from CEREBRUM Figure 13.",
        "Figure 13 provides the canonical mapping.",
        "CASE_FREE_ENERGY_PARAMS contains all cases.",
    ),
)
def test_retired_attribution_scanner_has_live_positive_controls(planted: str) -> None:
    assert _retired_attributions(planted), f"scanner accepted planted false claim: {planted}"
