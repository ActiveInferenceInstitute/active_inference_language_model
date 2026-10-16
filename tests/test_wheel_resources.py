"""Installed-wheel smoke tests for committed runtime corpus artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import site
import stat
import subprocess
import sys
import tarfile
import textwrap
import tomllib
import zipfile
from collections.abc import Iterable, Mapping
from email.parser import BytesParser
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from ailm.resources import (
    RAW_CORPUS_CACHE_ENV,
    RUNTIME_DATA_ARTIFACTS,
    RUNTIME_VENDOR_ARTIFACTS,
    _platform_user_cache_directory,
    default_raw_corpus_directory,
    resolve_data_artifact,
    resolve_raw_corpus_path,
    resolve_vendored_artifact,
)
from scripts.verify_release import DISTRIBUTION_MARKER

PROJECT_ROOT = Path(__file__).resolve().parents[1]
#: The declared package version, read from pyproject.toml so the synthetic archive
#: member names below track a release bump instead of pinning a stale version string.
PACKAGE_VERSION = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
    "project"
]["version"]
PACKAGED_ARTIFACTS = RUNTIME_VENDOR_ARTIFACTS
PACKAGED_DATA_ARTIFACTS = RUNTIME_DATA_ARTIFACTS
INSTALLED_CMUDICT = b"cat  K AE1 T\n"
INSTALLED_UD_EWT = b"""\
# sent_id = installed-wheel
# text = Cats sleep .
1\tCats\tcat\tNOUN\tNNS\tNumber=Plur\t2\tnsubj\t_\t_
2\tsleep\tsleep\tVERB\tVBP\tMood=Ind|Tense=Pres\t0\troot\t_\t_
3\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\tSpaceAfter=No

"""

SDIST_ROOT_FILES = frozenset(
    {".gitignore", "LICENSE", "PKG-INFO", "README.md", "THIRD_PARTY_NOTICES.md", "pyproject.toml"}
)
SDIST_SOURCE_SUFFIXES = frozenset({".css", ".html", ".js", ".md", ".py"})
SDIST_SENSITIVE_COMPONENTS = frozenset(
    {".advance", ".claude", ".env", ".git", "credential", "credentials", "secret", "secrets"}
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_safe_relative_archive_path(name: str) -> bool:
    """Whether an archive member is an unambiguous relative POSIX path."""
    if not name or "\x00" in name or "\\" in name:
        return False
    path_text = name[:-1] if name.endswith("/") else name
    if not path_text:
        return False
    parts = path_text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    posix_path = PurePosixPath(path_text)
    windows_path = PureWindowsPath(path_text)
    return (
        not posix_path.is_absolute() and not windows_path.is_absolute() and not windows_path.drive
    )


def _unsafe_wheel_members(members: Iterable[zipfile.ZipInfo]) -> set[str]:
    """Reject unsafe paths and Unix wheel members that are not files/directories."""
    unsafe: set[str] = set()
    for member in members:
        if not _is_safe_relative_archive_path(member.filename):
            unsafe.add(member.filename)
            continue
        if member.create_system != 3:
            # ZIP's portable model has only files and explicit directory names.
            # Special Unix types exist only when the creator supplies Unix mode
            # bits, handled below.
            continue
        unix_mode = (member.external_attr >> 16) & 0xFFFF
        member_type = stat.S_IFMT(unix_mode)
        if member.is_dir():
            if member_type not in {0, stat.S_IFDIR}:
                unsafe.add(member.filename)
        elif member_type not in {0, stat.S_IFREG}:
            unsafe.add(member.filename)
    return unsafe


def _unsafe_sdist_members(members: Iterable[tarfile.TarInfo]) -> set[str]:
    """Reject traversal and every tar entry type except regular file/directory."""
    return {
        member.name
        for member in members
        if not _is_safe_relative_archive_path(member.name)
        or not (member.isfile() or member.isdir())
    }


def _artifact_closure_defects(expected: Mapping[str, str], actual: Mapping[str, str]) -> set[str]:
    """Compare named artifact digests without letting an omission pass vacuously."""
    defects = {f"missing:{name}" for name in expected.keys() - actual.keys()}
    defects.update(f"unexpected:{name}" for name in actual.keys() - expected.keys())
    defects.update(
        f"digest-mismatch:{name}"
        for name in expected.keys() & actual.keys()
        if expected[name] != actual[name]
    )
    return defects


def _unexpected_sdist_members(names: list[str]) -> set[str]:
    """Return members outside the explicit public source-distribution surface."""
    if not names:
        return {"<empty-sdist>"}
    unexpected = {name for name in names if not _is_safe_relative_archive_path(name)}
    safe_names = [name for name in names if name not in unexpected]
    roots = {PurePosixPath(name).parts[0] for name in safe_names}
    if len(roots) != 1:
        unexpected.update(safe_names)
        return unexpected
    for name in safe_names:
        parts = PurePosixPath(name).parts
        relative = parts[1:]
        if not relative:
            continue
        lowered = {part.lower() for part in relative}
        if lowered & SDIST_SENSITIVE_COMPONENTS or any(
            part.startswith(".") for part in relative[1:]
        ):
            unexpected.add(name)
            continue
        if len(relative) == 1 and relative[0] in SDIST_ROOT_FILES:
            continue
        if relative[:2] == ("src", "ailm") and Path(relative[-1]).suffix in SDIST_SOURCE_SUFFIXES:
            continue
        if "/".join(relative) in {f"data/{name}" for name in PACKAGED_DATA_ARTIFACTS}:
            continue
        if relative[:2] == ("gnn", "models") and len(relative) == 3:
            if relative[-1] in {
                path.name for path in (PROJECT_ROOT / "gnn/models").glob("*.gnn.md")
            }:
                continue
        if relative[0] == "vendor" and relative[-1] in {
            *PACKAGED_ARTIFACTS,
            "AGENTS.md",
            "README.md",
            # Decision-79 confirmatory prep vendored the pinned UD EWT splits
            # (tracked upstream data, CC BY-SA 4.0, covered by
            # THIRD_PARTY_NOTICES.md). They ship in the sdist as build inputs
            # but are not force-included into the wheel's ailm/_vendor surface.
            "en_ewt-ud-dev.conllu",
            "en_ewt-ud-test.conllu",
            "en_gum-ud-test.conllu",
        }:
            continue
        unexpected.add(name)
    return unexpected


def test_sdist_member_gate_rejects_planted_private_and_untracked_surfaces() -> None:
    root = f"ailm-{PACKAGE_VERSION}/"
    planted = [
        root + ".claude/settings.local.json",
        root + ".advance/REPORT.md",
        root + ".env",
        root + "package-lock.json",
        root + "src/ailm/.secrets/token.txt",
        root + "src/ailm/credentials.json",
        root + "vendor/cmudict.dict",
        root + "data/claims.json",
        root + "gnn/models/private.json",
    ]
    assert _unexpected_sdist_members(planted) == set(planted)


@pytest.mark.parametrize(
    "name",
    (
        "../escape.py",
        "/absolute.py",
        "ailm/../../escape.py",
        "ailm/./payload.py",
        "ailm//payload.py",
        "C:/payload.py",
        "C:payload.py",
        "ailm\\payload.py",
        "ailm/\x00payload.py",
    ),
)
def test_archive_member_path_gate_rejects_ambiguous_or_escaping_paths(name: str) -> None:
    wheel_member = zipfile.ZipInfo(name)
    wheel_member.create_system = 3
    wheel_member.external_attr = (stat.S_IFREG | 0o644) << 16
    sdist_member = tarfile.TarInfo(name)
    sdist_member.type = tarfile.REGTYPE
    assert _unsafe_wheel_members([wheel_member]) == {wheel_member.filename}
    assert _unsafe_sdist_members([sdist_member]) == {name}


def test_archive_member_type_gates_accept_only_regular_files_and_directories() -> None:
    wheel_file = zipfile.ZipInfo("ailm/module.py")
    wheel_file.create_system = 3
    wheel_file.external_attr = (stat.S_IFREG | 0o644) << 16
    wheel_directory = zipfile.ZipInfo("ailm/")
    wheel_directory.create_system = 3
    wheel_directory.external_attr = (stat.S_IFDIR | 0o755) << 16

    sdist_file = tarfile.TarInfo(f"ailm-{PACKAGE_VERSION}/src/ailm/module.py")
    sdist_file.type = tarfile.REGTYPE
    sdist_directory = tarfile.TarInfo(f"ailm-{PACKAGE_VERSION}/src/ailm")
    sdist_directory.type = tarfile.DIRTYPE

    assert _unsafe_wheel_members([wheel_file, wheel_directory]) == set()
    assert _unsafe_sdist_members([sdist_file, sdist_directory]) == set()


@pytest.mark.parametrize(
    "file_type",
    (stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR, stat.S_IFBLK, stat.S_IFSOCK),
)
def test_wheel_member_type_gate_rejects_planted_unix_special_files(file_type: int) -> None:
    planted = zipfile.ZipInfo("ailm/planted.py")
    planted.create_system = 3
    planted.external_attr = (file_type | 0o777) << 16
    assert _unsafe_wheel_members([planted]) == {planted.filename}


@pytest.mark.parametrize(
    "member_type",
    (
        tarfile.SYMTYPE,
        tarfile.LNKTYPE,
        tarfile.FIFOTYPE,
        tarfile.CHRTYPE,
        tarfile.BLKTYPE,
    ),
)
def test_sdist_member_type_gate_rejects_planted_links_and_special_files(
    member_type: bytes,
) -> None:
    planted = tarfile.TarInfo(f"ailm-{PACKAGE_VERSION}/src/ailm/planted.py")
    planted.type = member_type
    planted.linkname = "../../outside" if member_type in {tarfile.SYMTYPE, tarfile.LNKTYPE} else ""
    assert _unsafe_sdist_members([planted]) == {planted.name}


WHEEL_NAME = f"ailm-{PACKAGE_VERSION}-py3-none-any.whl"


@pytest.mark.parametrize(
    ("actual", "expected_defect"),
    (
        ({}, f"missing:{WHEEL_NAME}"),
        (
            {WHEEL_NAME: "different"},
            f"digest-mismatch:{WHEEL_NAME}",
        ),
    ),
)
def test_artifact_closure_gate_rejects_planted_omission_and_mismatch(
    actual: dict[str, str], expected_defect: str
) -> None:
    expected = {WHEEL_NAME: "canonical"}
    assert expected_defect in _artifact_closure_defects(expected, actual)


def test_source_resource_is_the_canonical_vendor_file() -> None:
    resource = resolve_vendored_artifact("brown_counts.json")
    canonical = PROJECT_ROOT / "vendor" / "brown_counts.json"
    assert resource.read_bytes() == canonical.read_bytes()


def test_runtime_manifest_owns_every_committed_vendor_json() -> None:
    committed = {path.name for path in (PROJECT_ROOT / "vendor").glob("*.json")}
    assert set(PACKAGED_ARTIFACTS) == committed
    assert tuple(sorted(PACKAGED_ARTIFACTS)) == PACKAGED_ARTIFACTS


@pytest.mark.parametrize("filename", ["", ".", "..", "../brown_counts.json", "x/y.json"])
def test_resource_resolver_rejects_non_basename(filename: str) -> None:
    with pytest.raises(ValueError, match="must be a basename"):
        resolve_vendored_artifact(filename)
    with pytest.raises(ValueError, match="must be a basename"):
        resolve_raw_corpus_path(filename)


def test_resource_resolver_rejects_conflicting_overrides(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_vendored_artifact(
            "brown_counts.json",
            path=tmp_path / "counts.json",
            directory=tmp_path,
        )


def test_source_checkout_raw_downloads_default_to_vendor(monkeypatch) -> None:
    monkeypatch.delenv(RAW_CORPUS_CACHE_ENV, raising=False)
    assert default_raw_corpus_directory() == PROJECT_ROOT / "vendor"
    assert resolve_raw_corpus_path("cmudict.dict") == PROJECT_ROOT / "vendor" / "cmudict.dict"


def test_explicit_raw_corpus_cache_overrides_the_source_checkout(
    tmp_path: Path, monkeypatch
) -> None:
    cache = tmp_path / "corpus-cache"
    monkeypatch.setenv(RAW_CORPUS_CACHE_ENV, str(cache))
    assert default_raw_corpus_directory() == cache
    assert resolve_raw_corpus_path("en_ewt-ud-dev.conllu") == (cache / "en_ewt-ud-dev.conllu")


@pytest.mark.parametrize("configured", ["", "relative/cache"])
def test_explicit_raw_corpus_cache_must_be_an_absolute_nonempty_path(
    configured: str, monkeypatch
) -> None:
    monkeypatch.setenv(RAW_CORPUS_CACHE_ENV, configured)
    with pytest.raises(ValueError, match=RAW_CORPUS_CACHE_ENV):
        default_raw_corpus_directory()


def test_platform_user_cache_locations_are_stable_and_user_owned(tmp_path: Path) -> None:
    home = tmp_path / "home"
    assert (
        _platform_user_cache_directory({}, platform="darwin", operating_system="posix", home=home)
        == home / "Library" / "Caches" / "ailm" / "corpora"
    )
    assert (
        _platform_user_cache_directory({}, platform="linux", operating_system="posix", home=home)
        == home / ".cache" / "ailm" / "corpora"
    )

    xdg = tmp_path / "xdg"
    assert (
        _platform_user_cache_directory(
            {"XDG_CACHE_HOME": str(xdg)},
            platform="linux",
            operating_system="posix",
            home=home,
        )
        == xdg / "ailm" / "corpora"
    )
    assert (
        _platform_user_cache_directory(
            {"XDG_CACHE_HOME": "relative/cache"},
            platform="linux",
            operating_system="posix",
            home=home,
        )
        == home / ".cache" / "ailm" / "corpora"
    )

    local_app_data = tmp_path / "local-app-data"
    assert (
        _platform_user_cache_directory(
            {"LOCALAPPDATA": str(local_app_data)},
            platform="win32",
            operating_system="nt",
            home=home,
        )
        == local_app_data / "ailm" / "corpora"
    )
    assert (
        _platform_user_cache_directory({}, platform="win32", operating_system="nt", home=home)
        == home / "AppData" / "Local" / "ailm" / "corpora"
    )


def test_runtime_data_resources_preserve_source_paths_and_refuse_unknown_inputs() -> None:
    for relative in PACKAGED_DATA_ARTIFACTS:
        assert resolve_data_artifact(relative) == PROJECT_ROOT / "data" / relative
    for unregistered in ("../claims.json", "claims.json", "/tmp/data.json"):
        with pytest.raises(ValueError, match="unregistered runtime data artifact"):
            resolve_data_artifact(unregistered)


@pytest.mark.timeout(900)
def test_installed_wheel_embeds_exact_artifacts_and_default_loaders_work(
    tmp_path: Path, local_http_server
) -> None:
    """Build direct and sdist wheels, then smoke the sdist-derived wheel."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.fail("uv is required to verify the wheel built by this uv-managed project")

    dist = tmp_path / "dist"
    repeat_dist = tmp_path / "repeat-dist"
    sdist_wheel_dist = tmp_path / "sdist-wheel-dist"
    build_environment = {**os.environ}
    # Network fallback would let an unreviewed future backend execute inside a
    # green package test. Build isolation is disabled deliberately: hatchling
    # is an exact dev dependency in uv.lock, so the already-verified project
    # environment is the build environment. A cold uv cache proves nothing is
    # downloaded behind that contract.
    build_environment["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
    build_environment["UV_NO_PROGRESS"] = "1"
    # The sdist rebuild runs from a temp path uv cannot associate with the
    # project environment, so the declared build environment is pinned
    # explicitly instead of inherited from the invoking shell's activation.
    build_environment["VIRTUAL_ENV"] = str(PROJECT_ROOT / ".venv")
    # A user-level uv Python pin (``~/.config/uv/.python-version``) otherwise
    # selects its own interpreter for that temp-path rebuild, one without the
    # locked hatchling; the declared environment's interpreter is pinned too.
    build_environment["UV_PYTHON"] = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    source_date_epoch = build_environment.get("SOURCE_DATE_EPOCH")
    if source_date_epoch is None:
        commit_epoch = subprocess.run(
            ["git", "show", "-s", "--format=%ct", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert commit_epoch.returncode == 0, commit_epoch.stderr
        source_date_epoch = commit_epoch.stdout.strip()
    assert source_date_epoch.isdigit() and str(int(source_date_epoch)) == source_date_epoch
    build_environment["SOURCE_DATE_EPOCH"] = source_date_epoch

    def build(destination: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                uv,
                "build",
                "--offline",
                "--no-build-isolation",
                "--sdist",
                "--wheel",
                "--out-dir",
                str(destination),
            ],
            cwd=PROJECT_ROOT,
            env=build_environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )

    built = build(dist)
    assert built.returncode == 0, f"locked offline distribution build failed:\n{built.stderr}"
    repeated = build(repeat_dist)
    assert repeated.returncode == 0, (
        f"repeated locked offline distribution build failed:\n{repeated.stderr}"
    )

    wheels = list(dist.glob("ailm-*.whl"))
    sdists = list(dist.glob("ailm-*.tar.gz"))
    assert len(wheels) == 1, f"expected one AILM wheel, found {wheels}"
    assert len(sdists) == 1, f"expected one AILM sdist, found {sdists}"
    repeated_artifacts = {path.name: _sha256(path) for path in repeat_dist.iterdir()}
    first_artifacts = {path.name: _sha256(path) for path in dist.iterdir()}
    assert repeated_artifacts == first_artifacts, (
        "two locked offline builds from one source epoch produced different distributions"
    )

    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    source_metadata = pyproject["project"]
    assert pyproject["build-system"]["requires"] == ["hatchling==1.32.0"]
    assert "hatchling==1.32.0" in pyproject["dependency-groups"]["dev"]
    assert pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"] == [
        "src/ailm",
        "vendor",
        "gnn/models",
        "data/case_word/ud_ewt_case_word_counts.json",
        "data/widened/ud_ewt_clause_widened_counts.json",
        "README.md",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "pyproject.toml",
    ]
    with zipfile.ZipFile(wheels[0]) as archive:
        unsafe_wheel_members = _unsafe_wheel_members(archive.infolist())
        assert not unsafe_wheel_members, (
            "wheel contains an unsafe path or non-file/directory member: "
            f"{sorted(unsafe_wheel_members)}"
        )
        members = set(archive.namelist())
        metadata_members = [name for name in members if name.endswith(".dist-info/METADATA")]
        assert len(metadata_members) == 1, metadata_members
        metadata_member = metadata_members[0]
        metadata = BytesParser().parsebytes(archive.read(metadata_member))
        assert metadata["Version"] == source_metadata["version"]
        assert (
            metadata["License-Expression"] == source_metadata["license"] == "LicenseRef-AILM-Mixed"
        )
        license_directory = metadata_member.removesuffix("METADATA") + "licenses/"
        assert license_directory + "LICENSE" in members
        assert license_directory + "THIRD_PARTY_NOTICES.md" in members

    with tarfile.open(sdists[0], "r:gz") as archive:
        archive_members = archive.getmembers()
        unsafe_sdist_members = _unsafe_sdist_members(archive_members)
        assert not unsafe_sdist_members, (
            "source distribution contains an unsafe path, link, or special file: "
            f"{sorted(unsafe_sdist_members)}"
        )
        names = [member.name for member in archive_members]
        members = {Path(name).parts[-1] for name in names}
        assert {"pyproject.toml", "LICENSE", "THIRD_PARTY_NOTICES.md"} <= members
        assert not _unexpected_sdist_members(names), (
            "source distribution contains files outside its public allowlist: "
            f"{sorted(_unexpected_sdist_members(names))}"
        )

    rebuilt_from_sdist = subprocess.run(
        [
            uv,
            "build",
            "--offline",
            "--no-build-isolation",
            "--wheel",
            "--out-dir",
            str(sdist_wheel_dist),
            str(sdists[0]),
        ],
        cwd=PROJECT_ROOT,
        env=build_environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert rebuilt_from_sdist.returncode == 0, (
        f"locked offline wheel build from the emitted sdist failed:\n{rebuilt_from_sdist.stderr}"
    )
    sdist_wheels = list(sdist_wheel_dist.glob("ailm-*.whl"))
    assert len(sdist_wheels) == 1, (
        f"expected one AILM wheel rebuilt from the sdist, found {sdist_wheels}"
    )
    direct_wheel_artifacts = {path.name: _sha256(path) for path in wheels}
    sdist_wheel_artifacts = {path.name: _sha256(path) for path in sdist_wheels}
    closure_defects = _artifact_closure_defects(direct_wheel_artifacts, sdist_wheel_artifacts)
    assert not closure_defects, (
        f"the emitted sdist does not close back to the direct wheel: {sorted(closure_defects)}"
    )
    assert sdist_wheels[0].read_bytes() == wheels[0].read_bytes(), (
        "the wheel rebuilt from the emitted sdist is not byte-identical to the direct wheel"
    )
    with zipfile.ZipFile(sdist_wheels[0]) as archive:
        unsafe_sdist_wheel_members = _unsafe_wheel_members(archive.infolist())
        assert not unsafe_sdist_wheel_members, (
            "sdist-derived wheel contains an unsafe path or non-file/directory member: "
            f"{sorted(unsafe_sdist_wheel_members)}"
        )

    environment = tmp_path / "venv"
    created = subprocess.run(
        [sys.executable, "-m", "venv", str(environment)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert created.returncode == 0, created.stderr
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    installed = subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            str(sdist_wheels[0]),
        ],
        cwd=tmp_path,
        env=build_environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert installed.returncode == 0, installed.stderr
    queried_site_packages = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert queried_site_packages.returncode == 0, queried_site_packages.stderr
    installed_site_packages = Path(queried_site_packages.stdout.strip()).resolve()

    expected_digests = {
        name: hashlib.sha256((PROJECT_ROOT / "vendor" / name).read_bytes()).hexdigest()
        for name in PACKAGED_ARTIFACTS
    }
    connected_resources = {
        f"_data/{name}": hashlib.sha256((PROJECT_ROOT / "data" / name).read_bytes()).hexdigest()
        for name in PACKAGED_DATA_ARTIFACTS
    }
    connected_resources.update(
        {
            f"_gnn/models/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (PROJECT_ROOT / "gnn/models").glob("*.gnn.md")
        }
    )
    cmudict_route = "/cmudict.dict"
    ud_route = "/en_ewt-ud-dev.conllu"
    local_http_server.enqueue_bytes(cmudict_route, INSTALLED_CMUDICT)
    local_http_server.enqueue_bytes(ud_route, INSTALLED_UD_EWT, content_type="text/plain")
    installed_corpus_cache = tmp_path / "installed-corpus-cache"
    smoke_program = textwrap.dedent(
        f"""
        import hashlib
        import json
        import os
        import sys
        from importlib.resources import files
        from pathlib import Path

        import ailm
        from ailm.benchmarks.harness import _load_benchmark_artifacts
        from ailm.corpus.cmudict import CANONICAL_ARPABET, load_cmudict
        from ailm.corpus.ud import load_ud_ewt_dev
        from ailm.levels.case import DEFAULT_ARTIFACT as CASE_DEFAULT
        from ailm.levels.case import load_case_artifact
        from ailm.levels.case_czech import DEFAULT_ARTIFACT as CZECH_DEFAULT
        from ailm.levels.case_czech import load_czech_case_artifact
        from ailm.levels.clause import DEFAULT_ARTIFACT as CLAUSE_DEFAULT
        from ailm.levels.clause import load_clause_artifact
        from ailm.levels.letter import load_brown_counts
        from ailm.levels.narrative import DEFAULT_ARTIFACT as NARRATIVE_DEFAULT
        from ailm.levels.narrative import load_narrative_artifact
        from ailm.levels.phoneme import PHONES, load_phoneme_counts
        from ailm.levels.synthetic import load_synthetic_counts
        from ailm.levels.word import DEFAULT_ARTIFACT as WORD_DEFAULT
        from ailm.levels.word import load_word_artifact
        from ailm.resources import default_raw_corpus_directory
        from ailm.levels.connected_builder import load_connected_model
        from ailm.levels.connected import ConnectedRuntime

        expected = {json.dumps(expected_digests, sort_keys=True)}
        project_root = Path({str(PROJECT_ROOT)!r}).resolve()
        environment = Path({str(environment)!r}).resolve()
        module_path = Path(ailm.__file__).resolve()
        if project_root in module_path.parents:
            raise RuntimeError(f"imported source checkout instead of wheel: {{module_path}}")
        if environment not in module_path.parents:
            raise RuntimeError(f"wheel import escaped isolated venv: {{module_path}}")

        configured_cache = os.environ.pop({RAW_CORPUS_CACHE_ENV!r})
        try:
            user_cache = default_raw_corpus_directory()
        finally:
            os.environ[{RAW_CORPUS_CACHE_ENV!r}] = configured_cache
        if os.name == "nt":
            expected_user_cache = Path(os.environ["LOCALAPPDATA"]) / "ailm" / "corpora"
        elif sys.platform == "darwin":
            expected_user_cache = Path.home() / "Library" / "Caches" / "ailm" / "corpora"
        else:
            expected_user_cache = Path(os.environ["XDG_CACHE_HOME"]) / "ailm" / "corpora"
        if user_cache != expected_user_cache:
            raise RuntimeError(
                f"installed default raw-corpus cache is not user-owned: {{user_cache}}"
            )
        expected_corpus_cache = Path({str(installed_corpus_cache)!r})
        if default_raw_corpus_directory() != expected_corpus_cache:
            raise RuntimeError(
                f"installed raw-corpus cache escaped override: {{default_raw_corpus_directory()}}"
            )

        directory = files("ailm").joinpath("_vendor")
        resources = {{item.name: item for item in directory.iterdir() if item.is_file()}}
        if set(resources) != set(expected):
            raise RuntimeError(f"packaged artifact names differ: {{sorted(resources)}}")
        for name, digest in expected.items():
            resource = resources[name]
            actual = hashlib.sha256(resource.read_bytes()).hexdigest()
            if resource.name != name or actual != digest:
                raise RuntimeError(f"resource identity mismatch for {{name}}: {{actual}}")
        payloads = {{
            name: json.loads(resource.read_text(encoding="utf-8"))
            for name, resource in resources.items()
        }}
        if any(not payload for payload in payloads.values()):
            raise RuntimeError("a packaged JSON artifact is empty")
        connected_resources = {json.dumps(connected_resources, sort_keys=True)}
        for relative, digest in connected_resources.items():
            resource = files("ailm").joinpath(*relative.split("/"))
            if hashlib.sha256(resource.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"connected resource identity mismatch: {{relative}}")

        defaults = {{
            "brown_word_counts.json": WORD_DEFAULT,
            "ud_czech_case_counts.json": CZECH_DEFAULT,
            "ud_ewt_case_counts.json": CASE_DEFAULT,
            "ud_ewt_clause_counts.json": CLAUSE_DEFAULT,
            "ud_ewt_narrative_counts.json": NARRATIVE_DEFAULT,
        }}
        for name, resource in defaults.items():
            if resource.name != name or not resource.is_file():
                raise RuntimeError(f"default artifact is not packaged: {{name}} -> {{resource}}")

        brown = load_brown_counts()
        phoneme = load_phoneme_counts()
        word = load_word_artifact()
        full_word = load_word_artifact(resources["brown_word_counts_full.json"])
        case = load_case_artifact()
        clause = load_clause_artifact()
        czech = load_czech_case_artifact()
        narrative = load_narrative_artifact()
        synthetic = load_synthetic_counts()
        benchmark_brown, benchmark_phoneme = _load_benchmark_artifacts()
        cmudict_digest = {hashlib.sha256(INSTALLED_CMUDICT).hexdigest()!r}
        raw_cmudict = load_cmudict(
            url={str(local_http_server.base_url + cmudict_route)!r},
            expected_sha256=cmudict_digest,
            expected_size_bytes={len(INSTALLED_CMUDICT)!r},
        )
        raw_ud = load_ud_ewt_dev(
            url={str(local_http_server.base_url + ud_route)!r},
            source_revision={"e" * 40!r},
            expected_sha256={hashlib.sha256(INSTALLED_UD_EWT).hexdigest()!r},
            expected_size_bytes={len(INSTALLED_UD_EWT)!r},
        )
        if load_cmudict(
            download=False,
            expected_sha256=cmudict_digest,
            expected_size_bytes={len(INSTALLED_CMUDICT)!r},
        ) != raw_cmudict:
            raise RuntimeError("installed CMUdict cache replay changed the parsed dictionary")
        if len(
            load_ud_ewt_dev(
                download=False,
                url={str(local_http_server.base_url + ud_route)!r},
                source_revision={"e" * 40!r},
                expected_sha256={hashlib.sha256(INSTALLED_UD_EWT).hexdigest()!r},
                expected_size_bytes={len(INSTALLED_UD_EWT)!r},
            )
        ) != len(raw_ud):
            raise RuntimeError("installed UD EWT cache replay changed the parsed corpus")
        loaded = (
            brown,
            word.vocab,
            full_word.vocab,
            case.train,
            clause.train,
            czech.train,
            narrative.train,
            synthetic,
            benchmark_brown,
            benchmark_phoneme,
            raw_cmudict,
            raw_ud,
        )
        if any(not value for value in loaded):
            raise RuntimeError("a packaged loader returned an empty artifact")
        if tuple(phoneme["phones"]) != PHONES or set(PHONES) != CANONICAL_ARPABET:
            raise RuntimeError("packaged CMUdict counts disagree with the canonical inventory")
        runtime = ConnectedRuntime(load_connected_model())
        result = runtime.infer_text("The cat. A dog?")
        if not result.events or len(result.tokens) != 4:
            raise RuntimeError("installed connected inference returned an incomplete trace")
        generated = runtime.generate_next_word(result)
        if not generated.policies or not generated.word:
            raise RuntimeError("installed connected generation returned no policy trace")
        print("CONNECTED_WHEEL_RUNTIME_OK")
        print(module_path)
        print(",".join(sorted(resources)))
        """
    )
    smoke_environment = {
        **os.environ,
        RAW_CORPUS_CACHE_ENV: str(installed_corpus_cache),
        "HOME": str(tmp_path / "home"),
        "LOCALAPPDATA": str(tmp_path / "local-app-data"),
        "PYTHONNOUSERSITE": "1",
        # Reuse only the already-pinned dependency installation. The AILM wheel
        # itself is installed into and imported from the disposable venv above.
        "PYTHONPATH": os.pathsep.join([str(installed_site_packages), site.getsitepackages()[0]]),
        "XDG_CACHE_HOME": str(tmp_path / "xdg-cache"),
    }
    smoked = subprocess.run(
        [str(python), "-c", smoke_program],
        cwd=tmp_path,
        env=smoke_environment,
        capture_output=True,
        text=True,
        check=False,
        # The smoke imports jax and pymdp from a cold disposable venv; strictly
        # below this test's 180 s mark so that mark, not the default, binds.
        timeout=150,
    )
    assert smoked.returncode == 0, smoked.stderr
    assert str(environment) in smoked.stdout
    assert ",".join(PACKAGED_ARTIFACTS) in smoked.stdout
    assert "CONNECTED_WHEEL_RUNTIME_OK" in smoked.stdout
    negative_resources = (
        (
            "_gnn/models/L1_letter.gnn.md",
            "from ailm.gnn import load_spec; load_spec('L1_letter')",
        ),
        (
            "_data/case_word/ud_ewt_case_word_counts.json",
            "from ailm.corpus.case_word import load_case_word_artifact; load_case_word_artifact()",
        ),
        (
            "_data/widened/ud_ewt_clause_widened_counts.json",
            "from ailm.levels.widened import load_widened_clause_artifact; load_widened_clause_artifact()",
        ),
    )
    for relative, program in negative_resources:
        resource = installed_site_packages / "ailm" / relative
        held = resource.with_name(resource.name + ".held")
        resource.rename(held)
        try:
            missing = subprocess.run(
                [str(python), "-c", program],
                cwd=tmp_path,
                env=smoke_environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        finally:
            held.rename(resource)
        assert missing.returncode != 0, f"missing installed resource silently accepted: {relative}"
        assert "FileNotFoundError" in missing.stderr, missing.stderr
    assert (installed_corpus_cache / "cmudict.dict").read_bytes() == INSTALLED_CMUDICT
    assert (installed_corpus_cache / "en_ewt-ud-dev.conllu").read_bytes() == INSTALLED_UD_EWT
    legacy_vendor = installed_site_packages.parent / "vendor"
    assert not (legacy_vendor / "cmudict.dict").exists()
    assert not (legacy_vendor / "en_ewt-ud-dev.conllu").exists()
    for artifact in (*wheels, *sdists):
        print(
            DISTRIBUTION_MARKER
            + json.dumps(
                {
                    "filename": artifact.name,
                    "sha256": _sha256(artifact),
                    "bytes": artifact.stat().st_size,
                    "source_date_epoch": source_date_epoch,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    assert len(local_http_server.requests_for(cmudict_route)) == 1
    assert len(local_http_server.requests_for(ud_route)) == 1
