"""Fail-closed tests for manuscript variable parsing and on-disk injection.

These tests call the production contract.  They do not read the tracked JSON as
an oracle for its own source and they never skip because generated output is
absent: missing publication surfaces are failures, not permission to examine
nothing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ailm.manuscript_contract import (
    EXCLUDED_MANUSCRIPT_DOCS,
    MANUSCRIPT_TOKEN_RE,
    ManuscriptContractError,
    find_hardcoded_manuscript_variables,
    resolve_manuscript_text,
    validate_manuscript_sources,
    validate_no_hardcoded_manuscript_variables,
    validate_resolved_manuscript_tree,
)
from ailm.manuscript_variables import generate_manuscript_variables
from scripts.z_generate_manuscript_variables import (
    INPUT_EPOCH_PATHSPEC,
    ManuscriptPublicationError,
    _canonical_variable_bytes,
    _ensure_source_date_epoch,
    _load_canonical_variable_map,
    _publish_staged_artifacts,
    _resolve_template_root,
    _stage_resolved_tree,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _audited_input_epoch(project_root: Path = PROJECT_ROOT) -> int:
    """Return the independently versioned source epoch for publication checks.

    The epoch is the committer time of the last commit that touched a manuscript
    input (any tracked path outside ``output/``), so a render-only commit cannot
    move it. The pathspec is spelled out here rather than imported, so the test
    stays an independent check on the generator's own rule.
    """
    completed = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "HEAD", "--", ":/", ":(exclude,top)output"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    raw = completed.stdout.strip()
    assert raw.isdecimal(), f"git did not provide an integer input-commit epoch: {raw!r}"
    return int(raw)


def test_generator_pathspec_names_the_inputs_the_test_checks_independently():
    assert INPUT_EPOCH_PATHSPEC == (":/", ":(exclude,top)output")


def _git(root: Path, *args: str, epoch: int | None = None) -> None:
    environment = os.environ.copy()
    if epoch is not None:
        stamp = datetime.fromtimestamp(epoch, tz=UTC).isoformat()
        environment["GIT_AUTHOR_DATE"] = stamp
        environment["GIT_COMMITTER_DATE"] = stamp
    subprocess.run(
        ["git", *args],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def _commit_all(root: Path, message: str, *, epoch: int) -> None:
    _git(root, "add", "-A")
    _git(
        root,
        "-c",
        "user.name=AILM Token Test",
        "-c",
        "user.email=token-test@example.invalid",
        "commit",
        "-q",
        "-m",
        message,
        epoch=epoch,
    )


def test_epoch_default_helper_has_a_fixed_point_under_render_only_commits(tmp_path):
    """A commit that only refreshes output/ must not move the derived epoch.

    This is the property the HEAD-epoch rule lacked: every commit of regenerated
    output moved HEAD, which changed the stamp, which dirtied output/ again.
    """
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "docs").mkdir()
    (root / "docs" / "section.md").write_text("Built at {{META_BUILD_TIMESTAMP}}.\n")
    _commit_all(root, "inputs", epoch=1_700_000_000)
    (root / "output").mkdir()
    (root / "output" / "rendered.md").write_text("Built at 2023-11-14T22:13:20Z.\n")
    _commit_all(root, "render only", epoch=1_700_000_100)

    environment: dict[str, str] = {}
    assert _ensure_source_date_epoch(environment, project_root=root) == "1700000000"
    assert environment == {"SOURCE_DATE_EPOCH": "1700000000"}

    # POSITIVE CONTROL: touching an input moves the epoch to that commit.
    (root / "docs" / "section.md").write_text("Changed. Built at {{META_BUILD_TIMESTAMP}}.\n")
    _commit_all(root, "input change", epoch=1_700_000_200)
    assert _ensure_source_date_epoch({}, project_root=root) == "1700000200"

    # NEGATIVE CONTROL: a tree with no input commit at all cannot be stamped.
    bare = tmp_path / "render-only"
    bare.mkdir()
    _git(bare, "init", "-q")
    (bare / "output").mkdir()
    (bare / "output" / "rendered.md").write_text("only output\n")
    _commit_all(bare, "render only", epoch=1_700_000_300)
    with pytest.raises(ManuscriptPublicationError, match="no commit reachable"):
        _ensure_source_date_epoch({}, project_root=bare)


@pytest.mark.timeout(1200)
def test_plain_generator_cli_defaults_to_the_audited_input_epoch(tmp_path):
    """The bare CLI, with no ``SOURCE_DATE_EPOCH``, stamps the audited input epoch.

    The CLI runs inside a shared clone of ``HEAD`` under ``tmp_path`` rather than
    against this checkout, because the suite must never rewrite tracked files
    under ``output/``. The clone shares this repository's history, so its audited
    input epoch is this checkout's, and ``output/`` is left out of the clone's
    checkout so the generator's only write lands in a tree nobody tracks.
    """
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", "--quiet", str(PROJECT_ROOT), str(clone)],
        check=True,
        capture_output=True,
        text=True,
        timeout=480,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", "HEAD", "--", ".", ":(exclude)output"],
        cwd=clone,
        check=True,
        capture_output=True,
        text=True,
        timeout=480,
    )
    tracked_map = PROJECT_ROOT / "output" / "data" / "manuscript_variables.json"
    tracked_bytes_before = tracked_map.read_bytes() if tracked_map.is_file() else None

    environment = os.environ.copy()
    environment.pop("SOURCE_DATE_EPOCH", None)
    subprocess.run(
        [sys.executable, "scripts/z_generate_manuscript_variables.py", "--print-only"],
        cwd=clone,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    published = _load_canonical_variable_map(
        clone / "output" / "data" / "manuscript_variables.json"
    )
    clone_epoch = _audited_input_epoch(clone)
    assert clone_epoch == _audited_input_epoch(), "the shared clone must carry this HEAD's epoch"
    expected = datetime.fromtimestamp(clone_epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert published["META_BUILD_TIMESTAMP"] == expected

    tracked_bytes_after = tracked_map.read_bytes() if tracked_map.is_file() else None
    assert tracked_bytes_after == tracked_bytes_before, (
        "the generator wrote this checkout's tracked token map; it must only write the clone"
    )


def test_epoch_default_helper_preserves_an_explicit_value():
    environment = {"SOURCE_DATE_EPOCH": "1700000000"}
    assert _ensure_source_date_epoch(environment) == "1700000000"
    assert environment == {"SOURCE_DATE_EPOCH": "1700000000"}


def _project(tmp_path: Path, text: str = "Loss: {{RESULT}} nats.\n") -> Path:
    manuscript = tmp_path / "docs" / "manuscript"
    manuscript.mkdir(parents=True)
    (manuscript / "00_test.md").write_text(text, encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    (data / "claims.json").write_text(
        json.dumps({"_schema": "ailm-claims-v2", "claims": {}}), encoding="utf-8"
    )
    return tmp_path


def _resolved(root: Path, variables: dict[str, str]) -> Path:
    target = root / "output" / "manuscript"
    target.mkdir(parents=True)
    source = root / "docs" / "manuscript" / "00_test.md"
    (target / source.name).write_text(
        resolve_manuscript_text(source.read_text(encoding="utf-8"), variables),
        encoding="utf-8",
    )
    config = root / "docs" / "manuscript" / "config.yaml"
    if config.is_file():
        (target / config.name).write_bytes(config.read_bytes())
    for bibliography in (root / "docs" / "manuscript").glob("*.bib"):
        (target / bibliography.name).write_bytes(bibliography.read_bytes())
    return target


class TestCanonicalGrammar:
    def test_digit_bearing_and_plain_tokens_match(self):
        assert MANUSCRIPT_TOKEN_RE.findall("{{STAGE1_LETTER_PERPLEXITY}} {{META_TITLE}}") == [
            "STAGE1_LETTER_PERPLEXITY",
            "META_TITLE",
        ]

    @pytest.mark.parametrize(
        "marker",
        (
            "{{lowercase}}",
            "{{1_LEADING_DIGIT}}",
            "{{ SPACED }}",
            "{{HYPHENATED-TOKEN}}",
            "{{UNCLOSED",
            "{{{RESULT}}}",
            "{{RESULT}}}",
            "{{{RESULT}}",
            "{{RESULT}} orphan }}",
            "{{RESULT}} extra {",
            "{{RESULT}} extra }",
        ),
    )
    def test_renderer_blind_spots_are_fatal(self, tmp_path, marker):
        root = _project(tmp_path, f"Measured value: {marker}.\n")
        with pytest.raises(ManuscriptContractError, match="malformed"):
            validate_manuscript_sources(root, {"RESULT": "1"})

    @pytest.mark.timeout(300)
    @pytest.mark.template_context
    def test_grammar_is_isomorphic_to_the_template_renderer(self):
        template_root = _resolve_template_root()
        assert template_root is not None, "template renderer checkout is required"
        sys.path.insert(0, str(template_root))
        from infrastructure.rendering import manuscript_injection

        assert manuscript_injection._TOKEN_RE.pattern == MANUSCRIPT_TOKEN_RE.pattern
        assert manuscript_injection.EXCLUDED_DOC_FILENAMES == EXCLUDED_MANUSCRIPT_DOCS


class TestPublicationBoundary:
    @staticmethod
    def _tree_snapshot(path: Path) -> tuple[tuple[str, bytes | None], ...]:
        return tuple(
            (entry.relative_to(path).as_posix(), None if entry.is_dir() else entry.read_bytes())
            for entry in sorted(path.rglob("*"))
        )

    def test_explicit_invalid_template_root_never_falls_back(self, tmp_path):
        invalid = tmp_path / "not-a-template"
        invalid.mkdir()
        assert _resolve_template_root({"TEMPLATE_REPO_ROOT": str(invalid)}) is None

    def test_explicit_valid_template_root_wins(self, tmp_path):
        template = tmp_path / "template"
        infrastructure = template / "infrastructure"
        infrastructure.mkdir(parents=True)
        (infrastructure / "__init__.py").write_text("", encoding="utf-8")
        assert _resolve_template_root({"TEMPLATE_REPO_ROOT": str(template)}) == template

    @pytest.mark.timeout(800)
    def test_the_template_beside_the_projects_root_is_found_without_the_home_fallback(
        self, tmp_path
    ):
        """The canonical ``<root>/template`` + ``<root>/projects/working/<name>`` layout.

        The walk-up from the script never reaches a sibling of ``projects``, and
        the home-directory fallback only works where that compatibility symlink
        exists. Exercised on a copy of the script under an isolated ``HOME`` with
        ``TEMPLATE_REPO_ROOT`` unset, in a subprocess so the copy's ``sys.path``
        edits stay out of this interpreter.
        """
        root = tmp_path / "root"
        infrastructure = root / "template" / "infrastructure"
        infrastructure.mkdir(parents=True)
        (infrastructure / "__init__.py").write_text("", encoding="utf-8")
        scripts = root / "projects" / "working" / "x" / "scripts"
        scripts.mkdir(parents=True)
        copied = scripts / "z_generate_manuscript_variables.py"
        copied.write_bytes(
            (PROJECT_ROOT / "scripts" / "z_generate_manuscript_variables.py").read_bytes()
        )
        home = tmp_path / "home"
        home.mkdir()
        environment = {
            name: value for name, value in os.environ.items() if name != "TEMPLATE_REPO_ROOT"
        }
        environment["HOME"] = str(home)
        program = (
            "import importlib.util\n"
            f"spec = importlib.util.spec_from_file_location('copied_generator', {str(copied)!r})\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "print(module._resolve_template_root({}))\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", program],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=400,
        )
        assert completed.stdout.strip() == str(root / "template")

        # NEGATIVE CONTROL: without the sibling checkout the same layout resolves nothing.
        (infrastructure / "__init__.py").unlink()
        completed = subprocess.run(
            [sys.executable, "-c", program],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=400,
        )
        assert completed.stdout.strip() == "None"

    @pytest.mark.parametrize(
        ("payload", "message"),
        (
            (b'{"RESULT":"1","RESULT":"2"}\n', "duplicate key"),
            (b'{"RESULT":"1"}\n', "not canonical bytes"),
        ),
    )
    def test_variable_map_loader_rejects_ambiguous_or_noncanonical_json(
        self, tmp_path, payload, message
    ):
        path = tmp_path / "manuscript_variables.json"
        path.write_bytes(payload)
        with pytest.raises(ManuscriptPublicationError, match=message):
            _load_canonical_variable_map(path)

    def test_variable_map_loader_accepts_only_canonical_bytes(self, tmp_path):
        path = tmp_path / "manuscript_variables.json"
        variables = {"SECOND": "2", "FIRST": "1"}
        path.write_bytes(_canonical_variable_bytes(variables))
        assert _load_canonical_variable_map(path) == variables

    @pytest.mark.template_context
    def test_resolved_tree_is_fully_staged_without_touching_live_tree(self, tmp_path):
        root = _project(tmp_path)
        live = root / "output" / "manuscript"
        live.mkdir(parents=True)
        (live / "sentinel.md").write_bytes(b"prior live tree\n")

        template_root = _resolve_template_root()
        assert template_root is not None, "template renderer checkout is required"
        sys.path.insert(0, str(template_root))
        from infrastructure.rendering.manuscript_injection import (
            write_resolved_manuscript_tree,
        )

        staged = _stage_resolved_tree(
            root,
            {"RESULT": "1.25"},
            write_resolved_manuscript_tree,
            validate_resolved_manuscript_tree,
        )

        assert staged.parent == live.parent
        assert staged != live
        assert (staged / "00_test.md").read_bytes() == b"Loss: 1.25 nats.\n"
        assert (live / "sentinel.md").read_bytes() == b"prior live tree\n"

    def test_pair_publication_commits_both_staged_artifacts(self, tmp_path):
        data = tmp_path / "output" / "data"
        data.mkdir(parents=True)
        target_json = data / "manuscript_variables.json"
        target_json.write_bytes(b"old-map\n")
        staged_json = data / ".manuscript_variables.stage"
        staged_json.write_bytes(b"new-map\n")

        target_tree = tmp_path / "output" / "manuscript"
        (target_tree / "nested").mkdir(parents=True)
        (target_tree / "old.md").write_bytes(b"old manuscript\n")
        (target_tree / "nested" / "old.bib").write_bytes(b"old bibliography\n")
        staged_tree = tmp_path / "output" / ".manuscript.stage"
        staged_tree.mkdir()
        (staged_tree / "new.md").write_bytes(b"new manuscript\n")

        def verify_new_pair() -> None:
            assert target_json.read_bytes() == b"new-map\n"
            assert (target_tree / "new.md").read_bytes() == b"new manuscript\n"

        _publish_staged_artifacts(
            staged_json=staged_json,
            staged_tree=staged_tree,
            target_json=target_json,
            target_tree=target_tree,
            precommit_check=verify_new_pair,
        )

        assert target_json.read_bytes() == b"new-map\n"
        assert self._tree_snapshot(target_tree) == (("new.md", b"new manuscript\n"),)
        assert not staged_json.exists()
        assert not staged_tree.exists()
        assert not tuple((tmp_path / "output").glob("*.rollback"))
        assert not tuple(data.glob("*.rollback"))

    def test_late_pair_failure_restores_prior_artifacts_byte_for_byte(self, tmp_path):
        data = tmp_path / "output" / "data"
        data.mkdir(parents=True)
        target_json = data / "manuscript_variables.json"
        old_json = b'{\n  "OLD": "preserve exact whitespace"\n}\n'
        target_json.write_bytes(old_json)
        staged_json = data / ".manuscript_variables.stage"
        staged_json.write_bytes(b'{\n  "NEW": "replacement"\n}\n')

        target_tree = tmp_path / "output" / "manuscript"
        (target_tree / "nested").mkdir(parents=True)
        (target_tree / "00_old.md").write_bytes(b"old manuscript\r\n")
        (target_tree / "nested" / "references.bib").write_bytes(b"@misc{old}\n")
        old_tree = self._tree_snapshot(target_tree)
        staged_tree = tmp_path / "output" / ".manuscript.stage"
        staged_tree.mkdir()
        (staged_tree / "00_new.md").write_bytes(b"new manuscript\n")

        class ForcedLateFailure(RuntimeError):
            pass

        def fail_after_both_replacements() -> None:
            assert target_json.read_bytes() != old_json
            assert (target_tree / "00_new.md").is_file()
            raise ForcedLateFailure("forced failure after both replacements")

        with pytest.raises(ForcedLateFailure, match="after both replacements"):
            _publish_staged_artifacts(
                staged_json=staged_json,
                staged_tree=staged_tree,
                target_json=target_json,
                target_tree=target_tree,
                precommit_check=fail_after_both_replacements,
            )

        assert target_json.read_bytes() == old_json
        assert self._tree_snapshot(target_tree) == old_tree
        assert not tuple((tmp_path / "output").glob("*.rollback"))
        assert not tuple(data.glob("*.rollback"))


class TestSourceContract:
    def test_known_nonempty_variable_succeeds(self, tmp_path):
        root = _project(tmp_path)
        assert validate_manuscript_sources(root, {"RESULT": "1.25"}) == {"RESULT"}

    def test_unknown_uppercase_variable_fails(self, tmp_path):
        root = _project(tmp_path, "{{DOES_NOT_EXIST}}\n")
        with pytest.raises(ManuscriptContractError, match="no producer"):
            validate_manuscript_sources(root, {"RESULT": "1.25"})

    def test_empty_used_variable_fails(self, tmp_path):
        root = _project(tmp_path)
        with pytest.raises(ManuscriptContractError, match="empty"):
            validate_manuscript_sources(root, {"RESULT": "  "})

    def test_invalid_variable_map_key_fails_even_when_unused(self, tmp_path):
        root = _project(tmp_path)
        with pytest.raises(ManuscriptContractError, match="invalid.*names"):
            validate_manuscript_sources(root, {"RESULT": "1", "bad-key": "2"})

    def test_recursive_variable_value_fails(self, tmp_path):
        root = _project(tmp_path)
        with pytest.raises(ManuscriptContractError, match="token braces"):
            validate_manuscript_sources(root, {"RESULT": "{{OTHER}}"})

    def test_zero_numbered_sections_cannot_pass_vacuously(self, tmp_path):
        manuscript = tmp_path / "docs" / "manuscript"
        manuscript.mkdir(parents=True)
        (manuscript / "preamble.md").write_text("{{RESULT}}", encoding="utf-8")
        with pytest.raises(ManuscriptContractError, match="no numbered"):
            validate_manuscript_sources(tmp_path, {"RESULT": "1"})

    def test_guidance_examples_are_outside_the_render_contract(self, tmp_path):
        root = _project(tmp_path)
        for name in EXCLUDED_MANUSCRIPT_DOCS:
            (root / "docs" / "manuscript" / name).write_text(
                "Examples: {{TOKEN}}, {{lowercase}}, {{1_BAD}}.\n", encoding="utf-8"
            )
        assert validate_manuscript_sources(root, {"RESULT": "1"}) == {"RESULT"}


class TestResolvedTreeContract:
    def test_exact_independent_substitution_succeeds(self, tmp_path):
        root = _project(tmp_path)
        target = _resolved(root, {"RESULT": "1.25"})
        validate_resolved_manuscript_tree(root, {"RESULT": "1.25"}, target)

    def test_manual_or_stale_output_fails(self, tmp_path):
        root = _project(tmp_path)
        target = _resolved(root, {"RESULT": "1.25"})
        (target / "00_test.md").write_text("Loss: 9.99 nats.\n", encoding="utf-8")
        with pytest.raises(ManuscriptContractError, match="drifted"):
            validate_resolved_manuscript_tree(root, {"RESULT": "1.25"}, target)


class TestHardcodedVariableContract:
    VARIABLES = {
        "BUILT_LEVEL_COUNT": "6",
        "BUILT_LEVEL_COUNT_WORD": "six",
        "CHANNEL_COUNT": "5",
        "CHANNEL_COUNT_WORD": "five",
    }

    @pytest.mark.parametrize(
        "prose",
        (
            "Perplexity was 11.9215 nats.",
            "Accuracy was 60%.",
            "The paired result had p = 0.5.",
            "The sample had n = 400.",
            "Perplexity was $11.9215$ nats.",
            "The measured result was $p = 0.0123$.",
            "Accuracy was `$60%`.",
            "Gemma 3 4B was evaluated.",
            "Llama 3.2 3B was evaluated.",
            "The served tag was `qwen2.5:3b`.",
            "The served tag was `llama3.2:3b`.",
            "inferactively-pymdp 1.0.3 supplied the agent.",
            "pymdp version 1.0.4 supplied the agent.",
            "CEREBRUM v1.4 supplied the mapping.",
            "CEREBRUM v1.5 supplied the mapping.",
            "The search ran through August 2026.",
            "The search ran through September 2026.",
            "Six levels are built.",
            "Seven levels are built.",
            "Six built levels are reported.",
            "There are 6 built levels.",
            "Five cross-level channels were measured.",
            "Four channels were measured.",
            "There are 6 measured channels.",
            "Two served probes were evaluated.",
            "Six contrast types were declared.",
            "Three narrative phases were measured.",
            "The readout uses default add-1 Laplace throughout.",
            "The readout uses add-0.5 smoothing.",
            "The readout uses add-one Laplace smoothing.",
            "The readout uses add-one\nsmoothing.",
            "The readout uses add-two Laplace smoothing.",
            "The readout uses add-0.5\nsmoothing.",
            "The fixed comparator is the add-one count emission.",
            "Five mis-wirings were built.",
            "Falcon 7B was evaluated.",
            "OLMo 2 7B was evaluated.",
            "Command R 35B was evaluated.",
            "The served tag was `command-r:latest`.",
            "The served tag was `llama3.2:latest`.",
            "Thirteen levels are built.",
            "Six separately scored levels are built.",
            "Six distinct linguistic representations are reported.",
            "Three exact composition seams are reported.",
            "Four bytes per float32 entry are allocated.",
            r"Perplexity was \(11.9215\) nats.",
            r"Accuracy was $60\%$.",
            "The p-value was $0.0123$.",
            "The served tag was `mistral:latest`.",
            "Granite 3.3 8B was evaluated.",
            "The measurement reads one hundred percent.",
            "Support collapses within three to four letters.",
            "The package ships a five percent sample.",
            "The result is sixty years old.",
            "The related system appeared six years earlier.",
            "The path captures roughly a fifth of the available headroom.",
            "Every source is primary and none is later than 1973.",
            "A learning rate other than one breaks the identity.",
            "The table contains eight rows.",
            "The design has four cells and two sweeps.",
            "Three arms share four statistics.",
            "Two controls catch three degeneracies.",
            "The level owns three matrices.",
            "Four comparators and three effects are reported.",
            "Four thresholds define the family.",
            "The figure shows three examples.",
            "Seven roles share abbreviations.",
            "Three mappings were considered.",
            "Five paths were measured.",
            "Two models were served.",
            "Stage 2 keeps two corrupted emissions as controls.",
            "Quantitative statements come in five strengths.",
            "Two claim-strength labels govern the result.",
            "The panel shows five levels at once.",
            "The caption compares stride[::5] and stride[::3].",
            "The three subsample re-analyses are dependent measurements.",
            "One earlier corpus-fitted system would retire the claim.",
            "Two strides of different spacing were compared.",
            "The constant is applied to all four comparisons.",
            "The nearest neighbors are the four works above.",
            "All five failed their audits as declared, in three distinct ways.",
            "The latter three were not executed as a connected cascade.",
            "Of the six, it is the one least represented.",
            "Two of them enter the same receiver.",
            "A second channel helps without a paired test.",
            "The remaining two are less clean.",
            "The largest of the six derived values bounds the other five.",
            # Mutable claim/configuration forms found by the independent audit.
            "The alternative asks for a gain of at least two nats.",
            "The inherited two-nat wording is impossible.",
            "The minimum gain remains three nats.",
            "The available gain is under one nat.",
            "The total gain remains below a single nat.",
            "Held-out words share a four-character prefix.",
            "Held-out words share a four-character morphological prefix.",
            "The split selects every fifth sorted family.",
            "The split selects every fifth alphabetically sorted family.",
            "Words of length four or more are disjoint.",
            "Words with a minimum length of four characters are disjoint.",
            "Short words contain five phones or fewer.",
            "Mid-length words (six to nine) form the middle bucket.",
            "Long words (ten or more) form the final bucket.",
            "The endpoint collapses onto one repeated word.",
            "The degenerate endpoint keeps emitting one identical word indefinitely.",
            "The endpoint repeats a single word.",
            "All three surviving mis-wirings alter the prior.",
            "Three word-level mis-wirings clear the bar.",
            "All three empirically surviving word-level mis-wirings clear the bar.",
            "The update uses `lr = 1`.",
            "The update uses `lr := 1`.",
            "Two of the declared members could not have rejected.",
            "Two declared family members can never reject.",
            "Three of its members were introduced after inspection.",
            "The table prints sha256 (first 12).",
            "The checksum uses the leading twelve characters.",
            "None of the first four conditions holds; the fifth remains unattempted.",
            "Each of the last three checks failed, while the fourth passed.",
            "The decoder produced zero decision disagreements.",
            "The observation carries zero mutual information.",
            "The three strata partition the scored population.",
            "Two guards pin the emission down.",
            "The measured error was 1.2e-06.",
            "The other twelve members of the family remain development-population inference.",
            "Two of the sixteen family members rest on a population fixed in advance.",
            "Five measured separately, the case-word bridge is an exploratory sweep.",
            "Two further case-word seams carry no scored estimand.",
        ),
    )
    def test_each_bypass_form_is_detected(self, tmp_path, prose):
        root = _project(tmp_path, prose + "\n")
        findings = find_hardcoded_manuscript_variables(root, self.VARIABLES)
        assert findings, f"scanner missed planted bypass: {prose}"
        with pytest.raises(ManuscriptContractError, match="hard-coded"):
            validate_no_hardcoded_manuscript_variables(root, self.VARIABLES)

    def test_hardcoded_scan_uses_the_live_variable_map(self, tmp_path):
        root = _project(tmp_path, "The selected coefficient is 0.387.\n")
        findings = find_hardcoded_manuscript_variables(root, {"COEFFICIENT": "0.387"})
        assert any(finding.category == "mapped-numeric-value" for finding in findings)
        assert find_hardcoded_manuscript_variables(root, {"COEFFICIENT": "0.388"}) == ()

    def test_structural_literals_are_not_misclassified(self, tmp_path):
        root = _project(
            tmp_path,
            "Stage 6 follows Level 5 [@source2024].\n"
            "The identity is $p = p_0$ and index is $x_1$.\n"
            "![layout](figure.png){#fig:test width=95%}\n"
            "Supplement 8.1, Figure 13, and Table 1 are source locators.\n"
            r"The recurrence has $x_0 = 1$ and $x_{i+1}=x_i+1$."
            "\n"
            "A conventional HMM implementation supplies the comparison.\n",
        )
        assert find_hardcoded_manuscript_variables(root, self.VARIABLES) == ()

    def test_tokenized_quantity_is_clean(self, tmp_path):
        root = _project(tmp_path, "Perplexity was {{RESULT}} nats.\n")
        assert find_hardcoded_manuscript_variables(root, self.VARIABLES) == ()

    def test_tokenized_subsample_stride_is_clean(self, tmp_path):
        root = _project(tmp_path, "The rule is stride[::{{RESULT}}].\n")
        assert find_hardcoded_manuscript_variables(root, self.VARIABLES) == ()

    def test_tokenized_smoothing_is_clean(self, tmp_path):
        root = _project(
            tmp_path,
            "The default is add-{{SMOOTHING}} Laplace smoothing.\n"
            "The recorded comparator is the add-{{EMISSION_SMOOTHING}} count emission.\n",
        )
        assert find_hardcoded_manuscript_variables(root, self.VARIABLES) == ()

    def test_hardcoded_scan_does_not_rewrite_an_unsupported_ledger(self, tmp_path):
        root = _project(tmp_path, "Perplexity was {{RESULT}} nats.\n")
        ledger = root / "data" / "claims.json"
        unsupported = b'{"_schema":"ailm-claims-v1","claims":{}}\n'
        ledger.write_bytes(unsupported)
        assert find_hardcoded_manuscript_variables(root, self.VARIABLES) == ()
        assert ledger.read_bytes() == unsupported

    def test_unresolved_output_fails(self, tmp_path):
        root = _project(tmp_path)
        target = _resolved(root, {"RESULT": "1.25"})
        (target / "00_test.md").write_text("{{RESULT}}\n", encoding="utf-8")
        with pytest.raises(ManuscriptContractError, match="unresolved"):
            validate_resolved_manuscript_tree(root, {"RESULT": "1.25"}, target)

    @pytest.mark.parametrize("mutation", ("missing", "extra"))
    def test_source_output_inventory_mismatch_fails(self, tmp_path, mutation):
        root = _project(tmp_path)
        target = _resolved(root, {"RESULT": "1.25"})
        if mutation == "missing":
            (target / "00_test.md").unlink()
        else:
            (target / "ghost.md").write_text("stale", encoding="utf-8")
        with pytest.raises(ManuscriptContractError, match="file set"):
            validate_resolved_manuscript_tree(root, {"RESULT": "1.25"}, target)

    @pytest.mark.parametrize("extra_path", ("ghost.txt", "stale/ghost.md"))
    def test_any_extra_output_entry_fails_exact_tree_contract(self, tmp_path, extra_path):
        root = _project(tmp_path)
        target = _resolved(root, {"RESULT": "1.25"})
        extra = target / extra_path
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_text("stale", encoding="utf-8")
        with pytest.raises(ManuscriptContractError, match="file set"):
            validate_resolved_manuscript_tree(root, {"RESULT": "1.25"}, target)

    @pytest.mark.parametrize("mutation", ("stale-config", "extra-bibliography"))
    def test_auxiliary_output_inventory_and_content_are_exact(self, tmp_path, mutation):
        root = _project(tmp_path)
        manuscript = root / "docs" / "manuscript"
        (manuscript / "config.yaml").write_text("paper: current\n", encoding="utf-8")
        (manuscript / "references.bib").write_text("@misc{current}\n", encoding="utf-8")
        target = _resolved(root, {"RESULT": "1.25"})
        if mutation == "stale-config":
            (target / "config.yaml").write_text("paper: stale\n", encoding="utf-8")
        else:
            (target / "ghost.bib").write_text("@misc{ghost}\n", encoding="utf-8")
        with pytest.raises(ManuscriptContractError, match="drifted|file set"):
            validate_resolved_manuscript_tree(root, {"RESULT": "1.25"}, target)


def test_real_authored_manuscript_has_complete_live_variables():
    variables = generate_manuscript_variables(PROJECT_ROOT)
    used = validate_manuscript_sources(PROJECT_ROOT, variables)
    assert len(used) >= 900, "token scan is unexpectedly narrow"


def test_real_authored_manuscript_has_no_hardcoded_variables():
    variables = generate_manuscript_variables(PROJECT_ROOT)
    validate_no_hardcoded_manuscript_variables(PROJECT_ROOT, variables)


@pytest.mark.template_context
def test_real_resolved_tree_is_exactly_current(monkeypatch):
    published_path = PROJECT_ROOT / "output" / "data" / "manuscript_variables.json"
    published = _load_canonical_variable_map(published_path)
    input_epoch = _audited_input_epoch()
    expected_timestamp = datetime.fromtimestamp(input_epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", str(input_epoch))
    fresh = generate_manuscript_variables(PROJECT_ROOT)
    assert fresh["META_BUILD_TIMESTAMP"] == expected_timestamp
    assert published["META_BUILD_TIMESTAMP"] == expected_timestamp, (
        "published timestamp is not the independently audited input-commit epoch"
    )
    assert _canonical_variable_bytes(fresh) == published_path.read_bytes(), (
        "published manuscript variable map is stale or noncanonical against source inputs"
    )
    validate_resolved_manuscript_tree(PROJECT_ROOT, fresh)


def test_resolved_tree_rejects_a_stale_timestamp_from_an_independent_epoch(tmp_path):
    input_epoch = _audited_input_epoch()
    expected = datetime.fromtimestamp(input_epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    stale = datetime.fromtimestamp(input_epoch + 1, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert stale != expected
    root = _project(tmp_path, "Built at {{META_BUILD_TIMESTAMP}}.\n")
    target = _resolved(root, {"META_BUILD_TIMESTAMP": stale})
    with pytest.raises(ManuscriptContractError, match="drifted"):
        validate_resolved_manuscript_tree(root, {"META_BUILD_TIMESTAMP": expected}, target)


def test_no_editorial_scaffolding_in_numbered_manuscript():
    markers = ("INSERT AFTER THIS LINE", "TODO:", "FIXME:", "XXX:")
    offenders: list[str] = []
    for path in sorted((PROJECT_ROOT / "docs" / "manuscript").glob("[0-9]*.md")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(marker in line for marker in markers):
                offenders.append(f"{path.name}:{lineno}: {line.strip()[:80]}")
    assert offenders == [], f"editorial scaffolding in manuscript prose: {offenders}"


@pytest.mark.template_context
def test_staging_resolves_copied_auxiliary_preamble(tmp_path):
    root = _project(tmp_path)
    (root / "docs/manuscript/preamble.md").write_text(
        "```latex\n\\fontsize{ {{BODY}} }{ {{BASELINE}} }\\selectfont\n```\n"
    )
    template_root = _resolve_template_root()
    assert template_root is not None
    sys.path.insert(0, str(template_root))
    from infrastructure.rendering.manuscript_injection import write_resolved_manuscript_tree

    variables = {"RESULT": "1.25", "BODY": "9.5", "BASELINE": "11.4"}
    staged = _stage_resolved_tree(
        root, variables, write_resolved_manuscript_tree, validate_resolved_manuscript_tree
    )
    assert (staged / "preamble.md").read_text() == (
        "```latex\n\\fontsize{ 9.5 }{ 11.4 }\\selectfont\n```\n"
    )


def _article_typography_tree(root):
    import shutil

    from ailm.manuscript_variables.literature import _load_rendering_variables

    _project(root)
    for name in ("preamble.md", "config.yaml"):
        shutil.copy2(PROJECT_ROOT / "docs/manuscript" / name, root / "docs/manuscript" / name)
    variables = {"RESULT": "1.25", **_load_rendering_variables(root)}
    template_root = _resolve_template_root()
    assert template_root is not None
    sys.path.insert(0, str(template_root))
    from infrastructure.rendering.manuscript_injection import write_resolved_manuscript_tree

    staged = _stage_resolved_tree(
        root, variables, write_resolved_manuscript_tree, validate_resolved_manuscript_tree
    )
    target = root / "output/manuscript"
    staged.rename(target)
    return variables, target


@pytest.mark.template_context
def test_article_style_survives_actual_provider_preamble_refresh_and_compiles(tmp_path):
    import re

    variables, target = _article_typography_tree(tmp_path)
    from infrastructure.rendering._manuscript_source import resolve_manuscript_dir
    from infrastructure.rendering._pdf_combined_preamble import inject_latex_preamble

    # Exercise the provider's actual destructive-copy boundary before compiling.
    consumed = resolve_manuscript_dir(tmp_path)
    assert consumed == target
    assert (target / "preamble.md").read_bytes() == (
        tmp_path / "docs/manuscript/preamble.md"
    ).read_bytes()
    validate_resolved_manuscript_tree(tmp_path, variables, consumed)
    assert "{{" not in (target / "preamble.md").read_text()
    tex = r"""\documentclass[10pt,letterpaper]{article}
\usepackage{lmodern}
\title{Style integration}
\author{Test author}
\begin{document}
\maketitle
\section{Style probe}
\makeatletter
\typeout{STYLE-BODY=\f@size}
\typeout{STYLE-BASELINE=\the\baselineskip}
\typeout{STYLE-LEFT=\the\dimexpr1in+\hoffset+\oddsidemargin\relax}
\typeout{STYLE-WIDTH=\the\textwidth}
\makeatother
Readable article body after title and heading resets.
\end{document}
"""
    pdf_dir = tmp_path / "output/pdf"
    pdf_dir.mkdir()
    (pdf_dir / "probe.tex").write_text(inject_latex_preamble(tex, consumed, skip_title_page=True))
    completed = subprocess.run(
        ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "probe.tex"],
        cwd=pdf_dir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (pdf_dir / "probe.pdf").is_file()
    log = (pdf_dir / "probe.log").read_text()

    def value(label):
        match = re.search(rf"{label}=([0-9.]+)", log)
        assert match is not None, label
        return float(match.group(1))

    assert value("STYLE-BODY") == pytest.approx(float(variables["META_BODY_FONT_PT"]))
    assert value("STYLE-BASELINE") == pytest.approx(
        float(variables["META_BODY_BASELINE_PT"]), abs=0.01
    )
    assert value("STYLE-LEFT") == pytest.approx(
        float(variables["META_MARGIN_LEFT_IN"]) * 72.27, abs=0.02
    )
    assert value("STYLE-WIDTH") == pytest.approx(
        (8.5 - float(variables["META_MARGIN_LEFT_IN"]) - float(variables["META_MARGIN_RIGHT_IN"]))
        * 72.27,
        abs=0.02,
    )
    text = subprocess.run(
        ["pdftotext", "probe.pdf", "-"],
        cwd=pdf_dir,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    assert "Readable article body after title and heading resets." in text


@pytest.mark.parametrize("defect", ["missing", "modified", "stale", "changed_config"])
@pytest.mark.template_context
def test_article_style_auxiliary_refuses_missing_or_stale_bytes(tmp_path, defect):
    import yaml

    from ailm.manuscript_variables.literature import _load_rendering_variables

    variables, target = _article_typography_tree(tmp_path)
    auxiliary = target / "article_typography.sty"
    if defect == "missing":
        auxiliary.unlink()
    elif defect == "modified":
        auxiliary.write_bytes(auxiliary.read_bytes() + b"% tampered\n")
    else:
        config = tmp_path / "docs/manuscript/config.yaml"
        payload = yaml.safe_load(config.read_text())
        payload["rendering"]["margin_left_in"] = 0.65
        config.write_text(yaml.safe_dump(payload))
        (target / "config.yaml").write_bytes(config.read_bytes())
        if defect == "stale":
            variables.update(_load_rendering_variables(tmp_path))
    with pytest.raises(ManuscriptContractError, match="article_typography|typography variables"):
        validate_resolved_manuscript_tree(tmp_path, variables, target)


@pytest.mark.template_context
def test_missing_article_style_refuses_actual_latex_compile(tmp_path):
    variables, target = _article_typography_tree(tmp_path)
    from infrastructure.rendering._manuscript_source import resolve_manuscript_dir
    from infrastructure.rendering._pdf_combined_preamble import inject_latex_preamble

    consumed = resolve_manuscript_dir(tmp_path)
    (target / "article_typography.sty").unlink()
    pdf_dir = tmp_path / "output/pdf"
    pdf_dir.mkdir()
    source = "\\documentclass{article}\n\\begin{document}\nBody.\n\\end{document}\n"
    (pdf_dir / "negative.tex").write_text(
        inject_latex_preamble(source, consumed, skip_title_page=True)
    )
    result = subprocess.run(
        ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "negative.tex"],
        cwd=pdf_dir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert "article_typography.sty" in result.stdout
    assert not (pdf_dir / "negative.pdf").exists()


@pytest.mark.parametrize(
    "directive",
    [
        r"\input{article_typography.sty}",
        r"\input{../../other/article_typography.sty}",
        r"\input{../manuscript/article_typography.tex}",
    ],
)
def test_article_style_refuses_alternative_input_paths(tmp_path, directive):
    from ailm.manuscript_contract import article_typography_required

    root = _project(tmp_path)
    (root / "docs/manuscript/preamble.md").write_text(directive)
    with pytest.raises(ManuscriptContractError, match="exact single input directive"):
        article_typography_required(root)


@pytest.mark.template_context
def test_article_style_beamer_branch_preserves_slide_geometry(tmp_path):
    import re

    _, target = _article_typography_tree(tmp_path)
    slides = tmp_path / "output/slides"
    slides.mkdir()
    source = r"""\documentclass{beamer}
\typeout{BEAMER-BEFORE-WIDTH=\the\textwidth}
\input{../manuscript/article_typography.sty}
\typeout{BEAMER-AFTER-WIDTH=\the\textwidth}
\begin{document}
\begin{frame}{Style boundary}
Slide text.
\end{frame}
\end{document}
"""
    (slides / "probe.tex").write_text(source)
    result = subprocess.run(
        ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "probe.tex"],
        cwd=slides,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    before = re.search(r"BEAMER-BEFORE-WIDTH=([0-9.]+)", result.stdout)
    after = re.search(r"BEAMER-AFTER-WIDTH=([0-9.]+)", result.stdout)
    assert before is not None and after is not None
    assert before.group(1) == after.group(1)
    assert (slides / "probe.pdf").is_file()


def _render_combined_with_provider(template_root: Path, consumed: Path, output: Path) -> Path:
    """Run the external renderer in its own runtime, as the template pipeline does.

    Its canonicalization worker uses sys.executable and needs the template's
    pypdf dependency, which is deliberately absent from controlled AILM Python.
    Never resolve the venv executable symlink: that would lose its site context.
    """
    python = template_root / ".venv/bin/python"
    assert python.is_file(), f"provider interpreter is missing: {python}"
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(name, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["SOURCE_DATE_EPOCH"] = "1700000000"
    program = r"""
import sys
from pathlib import Path
from infrastructure.rendering.config import RenderingConfig
from infrastructure.rendering.manuscript_discovery import discover_manuscript_files
from infrastructure.rendering.pdf_renderer import PDFRenderer

consumed, output = map(Path, sys.argv[1:])
discovered = discover_manuscript_files(consumed)
assert [path.name for path in discovered] == ["00_test.md"]
renderer = PDFRenderer(RenderingConfig(
    manuscript_dir=str(consumed), output_dir=str(output),
    pdf_dir=str(output / "pdf"), figures_dir=str(output / "figures"),
))
rendered = renderer.render_combined(discovered, consumed, "scanner_fixture")
assert rendered == output / "pdf/scanner_fixture_combined.pdf"
assert rendered.is_file()
# These values are written by the real canonicalization worker, not XeTeX.
from pypdf import PdfReader
metadata = PdfReader(rendered).metadata
assert metadata["/Creator"] == "XeTeX deterministic output"
assert metadata["/CreationDate"] == "D:20231114221320Z"
"""
    subprocess.run(
        [str(python), "-B", "-c", program, str(consumed), str(output)],
        cwd=template_root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    return output / "pdf/scanner_fixture_combined.pdf"


@pytest.mark.template_context
def test_article_style_is_not_a_standalone_manuscript_source(tmp_path):
    from ailm.manuscript_contract import ARTICLE_TYPOGRAPHY_NAME

    variables, target = _article_typography_tree(tmp_path)
    from infrastructure.rendering._manuscript_source import resolve_manuscript_dir
    from infrastructure.rendering.manuscript_discovery import discover_manuscript_files

    consumed = resolve_manuscript_dir(tmp_path)
    validate_resolved_manuscript_tree(tmp_path, variables, consumed)
    discovered = discover_manuscript_files(consumed)
    assert (target / ARTICLE_TYPOGRAPHY_NAME).is_file()
    assert (target / ARTICLE_TYPOGRAPHY_NAME) not in discovered
    assert [path.name for path in discovered] == ["00_test.md"]
    template_root = _resolve_template_root()
    assert template_root is not None
    rendered = _render_combined_with_provider(template_root, consumed, tmp_path / "output")
    assert rendered.is_file()
    text = subprocess.run(
        ["pdftotext", str(rendered), "-"], capture_output=True, text=True, check=True, timeout=30
    ).stdout
    assert "Loss: 1.25 nats." in text


@pytest.mark.parametrize("defect", ["missing", "wrong_runtime"])
@pytest.mark.template_context
def test_combined_provider_boundary_refuses_invalid_interpreter(tmp_path, defect):
    _, target = _article_typography_tree(tmp_path)
    provider = tmp_path / "provider"
    provider.mkdir()
    if defect == "missing":
        with pytest.raises(AssertionError, match="provider interpreter is missing"):
            _render_combined_with_provider(provider, target, tmp_path / "output")
    else:
        template_root = _resolve_template_root()
        assert template_root is not None
        (provider / "infrastructure").symlink_to(template_root / "infrastructure")
        (provider / ".venv").symlink_to(PROJECT_ROOT / ".venv", target_is_directory=True)
        # The actual AILM runtime compiles, then refuses canonicalization.
        with pytest.raises(subprocess.CalledProcessError) as failure:
            _render_combined_with_provider(provider, target, tmp_path / "output")
        assert "Deterministic PDF canonicalization worker failed" in failure.value.stderr
        assert "No module named 'pypdf'" in failure.value.stderr


@pytest.mark.template_context
def test_standalone_tex_auxiliary_is_detected_and_refused(tmp_path):
    from ailm.manuscript_contract import ARTICLE_TYPOGRAPHY_NAME

    variables, target = _article_typography_tree(tmp_path)
    from infrastructure.core.exceptions import CompilationError
    from infrastructure.rendering.config import RenderingConfig
    from infrastructure.rendering.manuscript_discovery import discover_manuscript_files
    from infrastructure.rendering.pdf_renderer import PDFRenderer

    legacy = target / "article_typography.tex"
    legacy.write_bytes((target / ARTICLE_TYPOGRAPHY_NAME).read_bytes())
    discovered = discover_manuscript_files(target)
    assert legacy in discovered
    with pytest.raises(ManuscriptContractError, match="extra=.*article_typography.tex"):
        validate_resolved_manuscript_tree(tmp_path, variables, target)
    renderer = PDFRenderer(RenderingConfig(pdf_dir=str(tmp_path / "output/pdf")))
    with pytest.raises(CompilationError):
        renderer.render(legacy)


@pytest.mark.template_context
def test_article_style_transition_replaces_legacy_tree_without_standalone_auxiliary(tmp_path):
    import shutil

    from ailm.manuscript_contract import ARTICLE_TYPOGRAPHY_NAME

    variables, target = _article_typography_tree(tmp_path)
    from infrastructure.rendering._manuscript_source import resolve_manuscript_dir
    from infrastructure.rendering.manuscript_discovery import discover_manuscript_files
    from infrastructure.rendering.manuscript_injection import write_resolved_manuscript_tree

    # Start with the exact old namespace that made the provider compile a style file.
    legacy = target / "article_typography.tex"
    (target / ARTICLE_TYPOGRAPHY_NAME).rename(legacy)
    preamble = target / "preamble.md"
    preamble.write_text(
        preamble.read_text().replace("article_typography.sty", "article_typography.tex")
    )
    retained = tmp_path / "retained-failure/article_typography.tex"
    retained.parent.mkdir()
    shutil.copy2(legacy, retained)
    before = retained.read_bytes()
    assert legacy in discover_manuscript_files(target)
    with pytest.raises(ManuscriptContractError, match="file set differs"):
        validate_resolved_manuscript_tree(tmp_path, variables, target)

    # Use the real generator staging and transactional tree replacement, not a cleanup glob.
    staged = _stage_resolved_tree(
        tmp_path, variables, write_resolved_manuscript_tree, validate_resolved_manuscript_tree
    )
    data = tmp_path / "output/data"
    data.mkdir()
    target_json = data / "manuscript_variables.json"
    target_json.write_bytes(_canonical_variable_bytes({"OLD": "retained epoch"}))
    staged_json = data / "candidate.json"
    staged_json.write_bytes(_canonical_variable_bytes(variables))

    def check():
        assert _load_canonical_variable_map(target_json) == variables
        validate_resolved_manuscript_tree(tmp_path, variables, target)

    _publish_staged_artifacts(
        staged_json=staged_json,
        staged_tree=staged,
        target_json=target_json,
        target_tree=target,
        precommit_check=check,
    )
    consumed = resolve_manuscript_dir(tmp_path)
    assert retained.read_bytes() == before
    assert not legacy.exists()
    assert (target / ARTICLE_TYPOGRAPHY_NAME).is_file()
    assert [path.name for path in discover_manuscript_files(consumed)] == ["00_test.md"]
    validate_resolved_manuscript_tree(tmp_path, variables, consumed)


@pytest.mark.parametrize(
    "display",
    [
        "$$x = 1$$",
        "$$x = 1",
        "```{=latex}\n" + r"\begin{equation*}x = 1\end{equation*}" + "\n```",
        "```{=latex}\n$$x = 1$$ {#eq:raw}\n```",
        "$$$$ {#eq:empty}",
        r"\[x = 1\]",
        r"\begin{equation*}x = 1\end{equation*}",
        r"\begin{align}x &= 1\end{align}",
        r"\begin{definition}$$x = 1$$ {#eq:trapped}\end{definition}",
        r"$$x = 1\tag{4}$$ {#eq:manual}",
        r"$$x = 1\nonumber$$ {#eq:suppressed}",
        "$$x = 1$$ {#eq:duplicate}\n\n$$y = 2$$ {#eq:duplicate}",
    ],
)
def test_display_equation_contract_rejects_unportable_numbering(tmp_path, display):
    root = tmp_path / "source"
    manuscript = root / "docs" / "manuscript"
    manuscript.mkdir(parents=True)
    (manuscript / "01_body.md").write_text("{{RESULT}}\n\n" + display)
    with pytest.raises(ManuscriptContractError, match="display equation"):
        validate_manuscript_sources(root, {"RESULT": "1"})


def test_display_equation_contract_accepts_numbered_aligned_math_and_fenced_examples(tmp_path):
    root = tmp_path / "source"
    manuscript = root / "docs" / "manuscript"
    manuscript.mkdir(parents=True)
    (manuscript / "01_body.md").write_text(
        "{{RESULT}}\n\n"
        r"$$\begin{aligned}x &= 1 \\ y &= 2\end{aligned}$$ {#eq:aligned}"
        "\n\n```text\n$$unnumbered example$$\n```\n"
    )
    assert validate_manuscript_sources(root, {"RESULT": "1"}) == {"RESULT"}
