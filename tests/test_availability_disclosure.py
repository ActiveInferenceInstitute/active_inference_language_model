from __future__ import annotations

import json
import tomllib
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_REPRO = _ROOT / "docs" / "manuscript" / "07_reproducibility.md"
_CITATION = _ROOT / "CITATION.cff"
_CONFIG = _ROOT / "docs" / "manuscript" / "config.yaml"
_PACKAGE = _ROOT / "pyproject.toml"

SOURCE_REPOSITORY = "github.com/ActiveInferenceInstitute/Active_Inference_Language_Model"
RELEASE_DOI = "10.5281/zenodo.22409419"


def _joined(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def _disclosure_problems(text: str) -> list[str]:
    problems = []
    if SOURCE_REPOSITORY not in text:
        problems.append("the section does not name the release repository")
    lowered = text.lower()
    if SOURCE_REPOSITORY + "/tree/v1.0.0" not in text:
        problems.append("the section omits the versioned source")
    if "https://doi.org/" + RELEASE_DOI not in text:
        problems.append("the section omits the release archive identity")
    if "availability do not establish independent reproduction" not in lowered:
        problems.append("the section conflates availability and reproduction")
    if any(phrase in lowered for phrase in ("reserved draft", "publication there remains planned")):
        problems.append("the section retains provisional release metadata")
    if "has been independently reproduced" in lowered:
        problems.append("the section claims an unrecorded independent reproduction")
    if "no independent reproduction outside the project is recorded" not in lowered:
        problems.append(
            "the section dropped the statement that no independent reproduction is "
            "recorded, so public availability now reads as reproduction"
        )
    return problems


def test_the_repository_claim_carries_a_true_availability_statement() -> None:
    assert _disclosure_problems(_joined(_REPRO)) == []


def test_the_disclosure_the_citation_file_and_the_paper_agree() -> None:
    """The places a reader checks availability must not disagree."""
    citation = yaml.safe_load(_CITATION.read_text(encoding="utf-8"))
    config = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    assert citation["repository-code"] == f"https://{SOURCE_REPOSITORY}", (
        "CITATION.cff must point at the release repository the manuscript names"
    )
    package = tomllib.loads(_PACKAGE.read_text(encoding="utf-8"))
    assert package["project"]["urls"]["Repository"] == citation["repository-code"]
    assert citation.get("doi") == (config.get("publication") or {}).get("doi"), (
        "CITATION.cff and the title page carry different DOIs"
    )
    assert citation["doi"] == config["publication"]["doi"] == RELEASE_DOI
    assert package["project"]["urls"]["DOI"] == "https://doi.org/" + RELEASE_DOI
    assert "doi_status" not in config["publication"]


def test_release_repository_matches_abstract_and_all_publication_metadata() -> None:
    repository = f"https://{SOURCE_REPOSITORY}"
    assert repository in _joined(_ROOT / "docs/manuscript/00_abstract.md")
    citation = yaml.safe_load(_CITATION.read_text(encoding="utf-8"))
    package = tomllib.loads(_PACKAGE.read_text(encoding="utf-8"))
    zenodo = json.loads((_ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    assert zenodo["license"] == "other-nc"
    assert "mixed-license scheme" in zenodo["notes"]
    assert "non-commercial-use restriction" in zenodo["notes"]
    assert citation["license-url"] == repository + "/blob/main/LICENSE"
    assert package["project"]["urls"]["Homepage"] == repository
    assert package["project"]["urls"]["Repository"] == repository
    assert package["project"]["urls"]["Issues"] == repository + "/issues"
    assert repository in {row["identifier"] for row in zenodo["related_identifiers"]}
    assert str(zenodo["version"]) == str(citation["version"]) == str(package["project"]["version"])


def test_the_guard_would_notice_the_stale_disclosures() -> None:
    """NEGATIVE CONTROL: each missing boundary and a false release fail."""
    current = _joined(_REPRO)
    for phrase in (
        SOURCE_REPOSITORY,
        "/tree/v1.0.0",
        RELEASE_DOI,
        "availability do not establish independent reproduction",
        "No independent reproduction outside the project is recorded",
    ):
        assert phrase in current
        assert _disclosure_problems(current.replace(phrase, "omitted")), phrase
    assert _disclosure_problems(current + " Publication there remains planned.")
    assert _disclosure_problems(current + " It has been independently reproduced.")
