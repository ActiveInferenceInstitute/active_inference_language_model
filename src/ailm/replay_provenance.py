"""Source-, configuration-, model-, and call-bound network claim receipts.

Hermetic claim recorders are replayed end to end by the publication suite.  A
recorder that calls a local model server cannot be replayed that way on a clean
checkout, but excluding it must not turn its ledger entries into unauthenticated
numbers.  This module gives those recorders the complementary contract:

* every ledger entry owned by the recorder is included, with its value, run id,
  and source;
* every Python implementation file, locked dependency declaration, recorder,
  and vendored JSON input is hashed;
* the canonical run configuration is hashed and checked against current code;
* every model call is named by an identity-bound current-schema cache key; and
* the receipt is deterministic and self-consistency checked.

Trust boundary: receipt and cache digests are unkeyed SHA-256 integrity checks.
They detect stale, partial, or accidentally modified state while the local
workspace and its writers are trusted. A writer able to replace a receipt or
cache record can recompute those digests, so this contract does not authenticate
against a hostile local writer. Live release checks separately resolve model
identities and re-query a stratified sample.

The call store accepts its current schema only. Unsupported rows invalidate the
store before ``provenance_binding`` can project any keys, so a receipt cannot be
constructed from or validated against a partially filtered cache.

The receipt does not claim that a disk record proves a server still answers the
same way.  The full release profile separately re-measures a stratified sample.
Together, the checks answer two distinct questions: *which exact inputs produced
the recorded claims?* and *does the live model still reproduce cached answers?*
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ailm.call_cache import CACHE_SCHEMA, CallCache

__all__ = [
    "BINDING_SCHEMA",
    "CONFIGURATION_SCHEMA",
    "NETWORK_CALL_ENTRY_POINTS",
    "RUN_SCOPE_PATTERN",
    "NetworkClaimBindingError",
    "binding_path",
    "declared_openrouter_provider",
    "declared_run_scope",
    "discover_network_recorders",
    "generate_run_scope",
    "maximum_prompt_eval_tokens",
    "network_recorder_configuration",
    "scoped_cache_protocol",
    "source_manifest",
    "using_publication_ledger",
    "validate_all_network_claim_bindings",
    "validate_network_claim_binding",
    "validate_network_measurement_inputs",
    "validate_run_scope",
    "write_network_claim_binding",
]

BINDING_SCHEMA = "ailm-network-claim-binding-v2"
CONFIGURATION_SCHEMA = "ailm-network-recorder-configuration-v1"
BINDING_DIRECTORY = Path("cache/network_claim_bindings")

# One definition of "contacts the model server".  Discovery is intentionally
# AST-based: a comment or docstring mentioning a client function cannot buy a
# recorder an exclusion from deterministic replay.
NETWORK_CALL_ENTRY_POINTS = frozenset(
    {
        "ollama_available",
        "ask_openrouter",
        "ask_ollama",
        "ask_ollama_distribution",
        "ask_ollama_answer_candidates",
        "benchmark_model",
        "post_generate_uncached",
        "repeat_agreement",
        "warm_ollama_model",
    }
)

_HEX_SHA256 = frozenset("0123456789abcdef")

#: Shape of a publication run scope: a UTC instant to microsecond resolution.
#:
#: Some measurements are deliberately live on every canonical run — the external
#: benchmark's reliability trials, and every scored call of the stipulated
#: synthetic probes, both of which would be meaningless if a memo could answer
#: them. Their replies are still stored, so two runs of the same recorder ask the
#: server the same questions twice. Under matched server state the answers agree;
#: after a model reload or an intervening workload they do not, and the store
#: refuses to overwrite an existing key with a different reply. That refusal is
#: correct, and the way to stop reaching it is not to relax it but to stop two
#: runs sharing a key: each run scopes its own cache namespaces by the scope
#: below, so a re-run appends its own records beside the previous run's.
#:
#: Microsecond resolution because two runs started in the same second would
#: otherwise be one scope, which is the collision this exists to prevent.
RUN_SCOPE_PATTERN = re.compile(r"\A[0-9]{8}T[0-9]{12}Z\Z")


class NetworkClaimBindingError(ValueError):
    """A network recorder receipt is absent, stale, or internally incoherent."""


def generate_run_scope() -> str:
    """Fix one recorder run's scope, once, before its first server call.

    Called by a recorder at start and passed down explicitly. Deliberately not
    called from inside a measurement function: a clock read there would make a
    call's stored cache identity depend on when the function happened to execute
    rather than on which run the record belongs to.
    """

    scope = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return validate_run_scope(scope)


def scoped_cache_protocol(protocol: str, run_scope: str) -> str:
    """Name one run's cache namespace under a protocol identity.

    The protocol identity keeps naming the measurement — it is what the ledger
    records and what a reader cites. The scope names the run that produced a
    particular set of records under it, and only the cache key carries it.
    """

    if not isinstance(protocol, str) or not protocol.strip() or protocol != protocol.strip():
        raise NetworkClaimBindingError("cache protocol must be a non-empty unpadded string")
    return f"{protocol}/{validate_run_scope(run_scope)}"


def declared_run_scope(project_root: Path, recorder: str | Path) -> str | None:
    """Return the run scope a recorder's receipt declares, or ``None``.

    A run-scoped recorder writes different cache namespaces on every run, so its
    publication configuration cannot be known from code alone: the run is the
    receipt's to state. This reads that one field so a caller can hand it back to
    the recorder and get the configuration that scope implies. The scope is not
    taken on trust — the recorder refuses one it could not have generated, and
    the receipt's cited records must carry exactly the namespaces it names.

    A receipt that is absent, unreadable, or shaped wrong yields ``None`` rather
    than an error. This is a lookup, not the validation; the validation runs
    against the same file immediately after and reports in its own words.
    """

    try:
        payload = json.loads(
            binding_path(project_root, recorder).read_text(encoding="utf-8"),
        )
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict):
        return None
    parameters = configuration.get("parameters")
    if not isinstance(parameters, dict):
        return None
    scope = parameters.get("run_scope")
    return scope if isinstance(scope, str) else None


def declared_openrouter_provider(project_root: Path, recorder: str | Path) -> dict[str, Any] | None:
    """Read a run's explicit provider declaration without consulting budget environment.

    Historical receipts omit the field. An explicit malformed field is refused;
    the recorder validates its schema and exact cached payloads independently.
    """
    path = binding_path(project_root, recorder)
    if not path.exists():
        return None
    payload = _read_json_object(path)
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict):
        raise NetworkClaimBindingError("network receipt configuration is invalid")
    parameters = configuration.get("parameters")
    if not isinstance(parameters, dict):
        raise NetworkClaimBindingError("network receipt configuration parameters are invalid")
    if "openrouter_provider" not in parameters:
        return None
    provider = parameters["openrouter_provider"]
    if not isinstance(provider, dict):
        raise NetworkClaimBindingError("network receipt provider declaration is invalid")
    return provider


def validate_run_scope(scope: object) -> str:
    """Return ``scope`` if a recorder could have generated it, else refuse.

    A receipt states its own run scope, because the run is the one part of a
    publication configuration that code cannot know in advance. This is what
    keeps that field from being a free text field: only a generated shape is
    publishable, so a receipt cannot name a namespace by hand.
    """

    if not isinstance(scope, str) or RUN_SCOPE_PATTERN.fullmatch(scope) is None:
        raise NetworkClaimBindingError(
            f"run scope must be a recorder-generated UTC run id, got {scope!r}"
        )
    return scope


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise NetworkClaimBindingError(
            f"network claim provenance must be finite canonical JSON: {error}"
        ) from error


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and not (set(value) - _HEX_SHA256)


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NetworkClaimBindingError(f"duplicate JSON key in network receipt: {key!r}")
        result[key] = value
    return result


def _read_regular_bytes(path: Path) -> bytes:
    """Read a real regular file without following a leaf symbolic link."""

    try:
        metadata = path.lstat()
    except OSError as error:
        raise NetworkClaimBindingError(f"cannot inspect network receipt {path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise NetworkClaimBindingError(f"network receipt is not a real regular file: {path}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise NetworkClaimBindingError(f"cannot open network receipt {path}: {error}") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise NetworkClaimBindingError(f"opened network receipt is not regular: {path}")
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise NetworkClaimBindingError(f"network receipt changed while opening: {path}")
        content = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            content.extend(chunk)
    finally:
        os.close(descriptor)
    return bytes(content)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            _read_regular_bytes(path).decode("utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                NetworkClaimBindingError(f"non-finite JSON constant {value!r} in {path}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NetworkClaimBindingError(
            f"invalid network receipt JSON at {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise NetworkClaimBindingError(f"network receipt root must be an object: {path}")
    return payload


def _relative_recorder(project_root: Path, recorder: str | Path) -> str:
    root = project_root.resolve()
    raw = Path(recorder)
    candidate = raw if raw.is_absolute() else root / raw
    try:
        candidate_metadata = candidate.lstat()
        if not stat.S_ISREG(candidate_metadata.st_mode):
            raise NetworkClaimBindingError(
                f"network recorder is not a real regular file: {candidate}"
            )
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
    except NetworkClaimBindingError:
        raise
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        raise NetworkClaimBindingError(
            f"network recorder must be an existing file beneath {root}: {recorder}"
        ) from error
    if not relative.startswith("scripts/record_") or not relative.endswith("_claims.py"):
        raise NetworkClaimBindingError(
            f"network recorder path does not name a claim recorder: {relative!r}"
        )
    return relative


def binding_path(project_root: Path, recorder: str | Path) -> Path:
    """Return the ignored receipt path assigned to one network recorder."""

    relative = _relative_recorder(project_root, recorder)
    return project_root.resolve() / BINDING_DIRECTORY / f"{Path(relative).stem}.json"


def _imported_server_names(path: Path) -> set[str]:
    """Resolve supported imports/aliases that expose a server entry point.

    Direct imports are classified even when unused, matching the fail-closed
    recorder boundary. Module imports require a reference to a registered entry
    point so pure helpers imported from :mod:`ailm.benchmarks` stay hermetic.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise NetworkClaimBindingError(f"cannot parse recorder {path}: {error}") from error

    imported: set[str] = set()
    module_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "ailm":
                for alias in node.names:
                    if alias.name == "benchmarks":
                        module_aliases.add(alias.asname or alias.name)
                continue
            if module == "ailm.benchmarks" or module.startswith("ailm.benchmarks."):
                for alias in node.names:
                    if alias.name == "*":
                        imported.update(NETWORK_CALL_ENTRY_POINTS)
                    elif alias.name in NETWORK_CALL_ENTRY_POINTS:
                        imported.add(alias.name)
                    elif alias.name == "ollama":
                        module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ailm.benchmarks" or alias.name.startswith("ailm.benchmarks."):
                    module_aliases.add(alias.asname or alias.name.split(".", 1)[0])

    def root_name(node: ast.expr) -> str | None:
        current = node
        while isinstance(current, ast.Attribute):
            current = current.value
        return current.id if isinstance(current, ast.Name) else None

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in NETWORK_CALL_ENTRY_POINTS
            and root_name(node) in module_aliases
        ):
            imported.add(node.attr)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in module_aliases
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in NETWORK_CALL_ENTRY_POINTS
        ):
            imported.add(str(node.args[1].value))
    return imported


def discover_network_recorders(project_root: Path) -> tuple[str, ...]:
    """Discover every claims recorder that imports a server entry point."""

    scripts = project_root.resolve() / "scripts"
    return tuple(
        path.relative_to(project_root.resolve()).as_posix()
        for path in sorted(scripts.glob("record_*_claims.py"))
        if _imported_server_names(path)
    )


def network_recorder_configuration(
    *,
    endpoint: str,
    call_protocol: str,
    model_tags: list[str] | tuple[str, ...] | set[str] | frozenset[str],
    parameters: Mapping[str, object],
) -> dict[str, object]:
    """Build and validate the uniform configuration recorded by every client."""

    if not isinstance(endpoint, str) or not endpoint.strip():
        raise NetworkClaimBindingError("network recorder endpoint must be non-empty")
    if not isinstance(call_protocol, str) or not call_protocol.strip():
        raise NetworkClaimBindingError("network recorder call protocol must be non-empty")
    tags = sorted(model_tags)
    if not tags or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
        raise NetworkClaimBindingError("network recorder model tags must be non-empty strings")
    if len(tags) != len(set(tags)):
        raise NetworkClaimBindingError("network recorder model tags must be unique")
    if not isinstance(parameters, Mapping):
        raise NetworkClaimBindingError("network recorder parameters must be a JSON object")
    configuration: dict[str, object] = {
        "schema": CONFIGURATION_SCHEMA,
        "endpoint": endpoint.rstrip("/"),
        "call_protocol": call_protocol,
        "model_tags": tags,
        "parameters": dict(parameters),
    }
    # Round-trip to reject tuples, non-string keys, NaN, and custom objects
    # instead of letting their repr become part of a scientific identity.
    normalized = json.loads(_canonical_json(configuration))
    if not isinstance(normalized, dict):  # pragma: no cover - constructed as an object
        raise NetworkClaimBindingError("normalized network configuration is not an object")
    return normalized


def maximum_prompt_eval_tokens(
    store: CallCache,
    keys: set[str] | frozenset[str] | list[str],
) -> int:
    """Return the largest server-reported prompt token count for exact calls.

    Ollama's ``prompt_eval_count`` is the executed tokenizer-level quantity.
    Publication measurements refuse missing, boolean, non-integral, or
    non-positive counts instead of treating a character-length proxy as proof
    that a prompt fit the configured context window.
    """

    selected = set(keys)
    if not selected:
        raise NetworkClaimBindingError("prompt-token validation requires call keys")
    replies = {key: reply for key, _payload, reply in store.entries() if key in selected}
    missing = sorted(selected - set(replies))
    if missing:
        raise NetworkClaimBindingError(
            f"prompt-token validation is missing cache keys: {missing[:3]}"
        )
    counts: list[int] = []
    for key in sorted(selected):
        count = replies[key].get("prompt_eval_count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise NetworkClaimBindingError(
                f"cached Ollama reply {key} has invalid prompt_eval_count {count!r}"
            )
        counts.append(count)
    return max(counts)


def using_publication_ledger(project_root: Path) -> bool:
    """Whether a recorder is writing the tracked ledger rather than an override."""

    override = os.environ.get("AILM_CLAIMS_PATH", "").strip()
    if not override:
        return True
    path = Path(override)
    if not path.is_absolute():
        raise NetworkClaimBindingError(
            "AILM_CLAIMS_PATH must be absolute before network provenance is decided"
        )
    try:
        return path.resolve() == (project_root.resolve() / "data" / "claims.json").resolve()
    except (OSError, RuntimeError) as error:  # pragma: no cover - platform resolution failure
        raise NetworkClaimBindingError("cannot resolve AILM_CLAIMS_PATH") from error


def _validate_configuration(configuration: object) -> dict[str, object]:
    if not isinstance(configuration, dict):
        raise NetworkClaimBindingError("network recorder configuration must be an object")
    if set(configuration) != {
        "schema",
        "endpoint",
        "call_protocol",
        "model_tags",
        "parameters",
    }:
        raise NetworkClaimBindingError("network recorder configuration has unexpected fields")
    if configuration.get("schema") != CONFIGURATION_SCHEMA:
        raise NetworkClaimBindingError("unsupported network recorder configuration schema")
    tags = configuration.get("model_tags")
    parameters = configuration.get("parameters")
    if not isinstance(tags, list) or not isinstance(parameters, dict):
        raise NetworkClaimBindingError("network recorder tags/parameters have invalid types")
    return network_recorder_configuration(
        endpoint=configuration.get("endpoint"),  # type: ignore[arg-type]
        call_protocol=configuration.get("call_protocol"),  # type: ignore[arg-type]
        model_tags=tags,
        parameters=parameters,
    )


def _local_script_import_closure(project_root: Path, recorder_path: Path) -> set[Path]:
    """Return explicit local ``scripts/`` imports reachable from a recorder.

    Network recorders run with both ``src`` and ``scripts`` on ``sys.path``.
    Hashing all of ``src/ailm`` therefore does not bind a shared script helper
    such as ``_artifact_io.py``. Follow ordinary static imports recursively so
    an executed local helper cannot change without invalidating the receipt.
    """

    scripts_root = project_root / "scripts"
    pending = [recorder_path]
    visited: set[Path] = set()
    helpers: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        try:
            metadata = path.lstat()
            path.resolve(strict=True).relative_to(project_root)
            if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
                raise NetworkClaimBindingError(
                    f"manifest implementation is not a real regular file: {path}"
                )
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except NetworkClaimBindingError:
            raise
        except (OSError, RuntimeError, SyntaxError, UnicodeError, ValueError) as error:
            raise NetworkClaimBindingError(
                f"cannot parse manifest implementation {path}: {error}"
            ) from error
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module)
        for module in modules:
            relative_module = module.removeprefix("scripts.")
            relative_path = Path(*relative_module.split("."))
            candidates = (
                scripts_root / relative_path.with_suffix(".py"),
                scripts_root / relative_path / "__init__.py",
            )
            for candidate in candidates:
                if candidate.is_file() or candidate.is_symlink():
                    if candidate not in helpers:
                        helpers.add(candidate)
                        pending.append(candidate)
    return helpers


def _manifest_paths(project_root: Path, recorder: str) -> tuple[Path, ...]:
    root = project_root.resolve()
    recorder_path = root / recorder
    paths = {
        *(root / "src" / "ailm").rglob("*.py"),
        recorder_path,
        *_local_script_import_closure(root, recorder_path),
        root / "pyproject.toml",
        root / "uv.lock",
        *(root / "vendor").glob("*.json"),
    }
    missing = sorted(path for path in paths if not path.exists())
    if missing:
        raise NetworkClaimBindingError(
            "network source/input manifest is incomplete: "
            + ", ".join(path.relative_to(root).as_posix() for path in missing)
        )
    return tuple(sorted(paths))


def source_manifest(project_root: Path, recorder: str | Path) -> dict[str, object]:
    """Hash the complete explicit implementation and immutable recorder inputs."""

    root = project_root.resolve()
    relative = _relative_recorder(root, recorder)
    files: list[dict[str, object]] = []
    for path in _manifest_paths(root, relative):
        try:
            metadata = path.lstat()
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError) as error:
            raise NetworkClaimBindingError(f"cannot authenticate manifest input {path}") from error
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise NetworkClaimBindingError(f"manifest input is not a real regular file: {path}")
        content = path.read_bytes()
        files.append(
            {
                "path": resolved.relative_to(root).as_posix(),
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return {"files": files, "manifest_sha256": _digest(files)}


def _claim_projection(project_root: Path, recorder: str) -> dict[str, object]:
    ledger_path = project_root.resolve() / "data" / "claims.json"
    payload = _read_json_object(ledger_path)
    if payload.get("_schema") != "ailm-claims-v2" or not isinstance(payload.get("claims"), dict):
        raise NetworkClaimBindingError("network claim binding requires an ailm-claims-v2 ledger")
    claims: dict[str, object] = {}
    for label, entry in sorted(payload["claims"].items()):
        if not isinstance(entry, dict):
            raise NetworkClaimBindingError(f"ledger claim {label!r} is not an object")
        if entry.get("source") != recorder:
            continue
        if set(entry) - {"value", "run_id", "source", "status"}:
            raise NetworkClaimBindingError(f"ledger claim {label!r} has unsupported fields")
        if not isinstance(label, str) or not isinstance(entry.get("run_id"), str):
            raise NetworkClaimBindingError(f"ledger claim {label!r} has invalid provenance")
        claims[label] = {
            "value": entry.get("value"),
            "run_id": entry["run_id"],
            "source": entry["source"],
        }
    if not claims:
        raise NetworkClaimBindingError(f"network recorder {recorder} owns no ledger claims")
    labels = sorted(claims)
    return {
        "count": len(labels),
        "labels": labels,
        "labels_sha256": _digest(labels),
        "entries": claims,
        "entries_sha256": _digest(claims),
    }


def _cache_projection(
    store: CallCache,
    keys: set[str] | frozenset[str],
    configuration: Mapping[str, object],
) -> dict[str, object]:
    if not keys:
        raise NetworkClaimBindingError(
            "network claim binding requires at least one validated current-schema call"
        )
    try:
        binding = store.provenance_binding(keys)
    except KeyError as error:
        raise NetworkClaimBindingError(str(error)) from error
    expected_models = configuration["model_tags"]
    if not isinstance(
        expected_models, list
    ):  # pragma: no cover - validated configurations always use a list
        raise NetworkClaimBindingError("configured model tags are not a list")
    if binding["schema"] != CACHE_SCHEMA:  # pragma: no cover - CallCache owns this constant
        raise NetworkClaimBindingError("network claims can only bind current call-cache records")
    if binding["models"] != expected_models:
        raise NetworkClaimBindingError(
            f"cache models {binding['models']} do not match configured models {expected_models}"
        )
    if binding["endpoints"] != [configuration["endpoint"]]:
        raise NetworkClaimBindingError("cache endpoint does not match recorder configuration")
    parameters = configuration.get("parameters")
    if not isinstance(parameters, Mapping):  # validated configuration invariant
        raise NetworkClaimBindingError("network configuration parameters are not an object")
    has_provider = "openrouter_provider" in parameters
    declared_provider = parameters.get("openrouter_provider")
    if has_provider and not isinstance(declared_provider, dict):
        raise NetworkClaimBindingError("configured OpenRouter provider must be an object")
    for key, payload, _reply in store.entries():
        if key in keys and (
            ("provider" in payload) != has_provider or payload.get("provider") != declared_provider
        ):
            raise NetworkClaimBindingError(
                "cached provider constraints do not match the declared recorder configuration"
            )
    declared_protocols = parameters.get("cache_protocols")
    if declared_protocols is None:
        expected_protocols = [configuration["call_protocol"]]
    else:
        if (
            not isinstance(declared_protocols, list)
            or not declared_protocols
            or any(
                not isinstance(protocol, str) or not protocol.strip()
                for protocol in declared_protocols
            )
            or len(declared_protocols) != len(set(declared_protocols))
        ):
            raise NetworkClaimBindingError(
                "configured cache_protocols must be unique non-empty strings"
            )
        expected_protocols = sorted(declared_protocols)
        if configuration["call_protocol"] not in expected_protocols:
            raise NetworkClaimBindingError(
                "configured cache_protocols must include the primary call_protocol"
            )
    if binding["protocols"] != expected_protocols:
        raise NetworkClaimBindingError("cache protocol does not match recorder configuration")
    identities = binding["identities"]
    model_identities = binding["model_identities"]
    if (
        not isinstance(identities, list)
        or len(identities) != len(expected_models)
        or any(
            not isinstance(identity, str)
            or not identity.startswith("sha256:")
            or not _is_sha256(identity.removeprefix("sha256:"))
            for identity in identities
        )
    ):
        raise NetworkClaimBindingError(
            "cache binding must name one immutable SHA-256 identity per configured model"
        )
    if (
        not isinstance(model_identities, list)
        or len(model_identities) != len(expected_models)
        or [row.get("model") for row in model_identities if isinstance(row, dict)]
        != expected_models
        or any(
            not isinstance(row, dict)
            or set(row) != {"model", "identity"}
            or row["identity"] not in identities
            for row in model_identities
        )
    ):
        raise NetworkClaimBindingError(
            "cache binding must pair each configured model with exactly one identity"
        )
    return {**binding, "record_count": len(keys)}


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise NetworkClaimBindingError(
            f"network binding parent must be a real directory: {path.parent}"
        )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_network_claim_binding(
    *,
    project_root: Path,
    recorder: str | Path,
    expected_source_manifest: Mapping[str, object],
    configuration: Mapping[str, object],
    store: CallCache,
    cache_keys: set[str] | frozenset[str],
    measurement: Mapping[str, object] | None = None,
) -> Path:
    """Write one deterministic receipt after a network recorder commits claims."""

    root = project_root.resolve()
    if not using_publication_ledger(root):
        raise NetworkClaimBindingError(
            "an isolated AILM_CLAIMS_PATH run cannot write a publication claim binding"
        )
    relative = _relative_recorder(root, recorder)
    normalized_configuration, normalized_measurement, cache_projection = (
        validate_network_measurement_inputs(
            configuration=configuration,
            store=store,
            cache_keys=cache_keys,
            measurement=measurement,
        )
    )
    current_source = source_manifest(root, relative)
    if current_source != expected_source_manifest:
        raise NetworkClaimBindingError(
            "source or vendored input changed after measurement began; "
            "network claims cannot be bound to a different source snapshot"
        )
    body: dict[str, object] = {
        "schema": BINDING_SCHEMA,
        "recorder": relative,
        "source": current_source,
        "configuration": normalized_configuration,
        "configuration_sha256": _digest(normalized_configuration),
        "claims": _claim_projection(root, relative),
        "cache": cache_projection,
        "measurement": normalized_measurement,
    }
    receipt = {**body, "receipt_sha256": _digest(body)}
    path = binding_path(root, relative)
    _atomic_write(path, _canonical_json(receipt) + "\n")
    return path


def validate_network_measurement_inputs(
    *,
    configuration: Mapping[str, object],
    store: CallCache,
    cache_keys: set[str] | frozenset[str],
    measurement: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object] | None, dict[str, object]]:
    """Validate every receipt input that does not depend on committed claims.

    Network recorders call this before replacing the ledger.  That makes a bad
    configuration, malformed measurement object, missing call key, wrong model
    identity, endpoint drift, or cache-protocol mismatch a pre-publication
    refusal rather than a failure discovered only after claims were committed.
    The writer repeats the same validation at the final trust boundary.
    """

    normalized_configuration = _validate_configuration(dict(configuration))
    normalized_measurement: dict[str, object] | None = None
    if measurement is not None:
        if not isinstance(measurement, Mapping):
            raise NetworkClaimBindingError("network measurement evidence must be an object")
        decoded = json.loads(_canonical_json(dict(measurement)))
        if not isinstance(decoded, dict):  # pragma: no cover - constructed as an object
            raise NetworkClaimBindingError("normalized network measurement is not an object")
        normalized_measurement = decoded
    cache_projection = _cache_projection(store, cache_keys, normalized_configuration)
    return normalized_configuration, normalized_measurement, cache_projection


def _validate_receipt_shape(payload: dict[str, Any]) -> None:
    expected = {
        "schema",
        "recorder",
        "source",
        "configuration",
        "configuration_sha256",
        "claims",
        "cache",
        "measurement",
        "receipt_sha256",
    }
    if set(payload) != expected:
        raise NetworkClaimBindingError("network claim receipt has unexpected fields")
    if payload.get("schema") != BINDING_SCHEMA:
        raise NetworkClaimBindingError("unsupported network claim receipt schema")
    if not _is_sha256(payload.get("configuration_sha256")) or not _is_sha256(
        payload.get("receipt_sha256")
    ):
        raise NetworkClaimBindingError("network claim receipt contains an invalid digest")
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload["receipt_sha256"] != _digest(body):
        raise NetworkClaimBindingError("network claim receipt digest mismatch")


def validate_network_claim_binding(
    *,
    project_root: Path,
    recorder: str | Path,
    expected_configuration: Mapping[str, object],
    cache_path: Path | None = None,
) -> dict[str, Any]:
    """Validate one receipt against current source, claims, config, and cache."""

    root = project_root.resolve()
    relative = _relative_recorder(root, recorder)
    path = binding_path(root, relative)
    payload = _read_json_object(path)
    _validate_receipt_shape(payload)
    if payload["recorder"] != relative:
        raise NetworkClaimBindingError("network claim receipt names the wrong recorder")

    configuration = _validate_configuration(payload["configuration"])
    expected = _validate_configuration(dict(expected_configuration))
    if payload["configuration_sha256"] != _digest(configuration):
        raise NetworkClaimBindingError("network claim configuration digest mismatch")
    if configuration != expected:
        raise NetworkClaimBindingError(
            "network claim receipt configuration is stale or not the publication configuration"
        )

    current_source = source_manifest(root, relative)
    if payload["source"] != current_source:
        raise NetworkClaimBindingError(
            "network claim receipt source/input manifest is stale or from another checkout"
        )
    current_claims = _claim_projection(root, relative)
    if payload["claims"] != current_claims:
        raise NetworkClaimBindingError(
            "network claim receipt does not bind the recorder's current complete ledger namespace"
        )

    store = CallCache(cache_path or (root / "cache" / "bench_calls.jsonl"))
    cached = payload["cache"]
    if not isinstance(cached, dict) or not isinstance(cached.get("keys"), list):
        raise NetworkClaimBindingError("network claim receipt has an invalid cache binding")
    keys = cached["keys"]
    if any(not isinstance(key, str) or not _is_sha256(key) for key in keys):
        raise NetworkClaimBindingError("network claim receipt contains an invalid call key")
    current_cache = _cache_projection(store, frozenset(keys), configuration)
    if cached != current_cache:
        raise NetworkClaimBindingError(
            "network claim receipt cache records are stale, missing, or from another store"
        )
    return payload


def validate_all_network_claim_bindings(
    *,
    project_root: Path,
    expected_configurations: Mapping[str, Mapping[str, object]],
    cache_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate complete, default-deny coverage of every network recorder."""

    root = project_root.resolve()
    discovered = discover_network_recorders(root)
    if not discovered:
        raise NetworkClaimBindingError("no network recorders were discovered")
    if set(expected_configurations) != set(discovered):
        missing = sorted(set(discovered) - set(expected_configurations))
        extra = sorted(set(expected_configurations) - set(discovered))
        raise NetworkClaimBindingError(
            f"network recorder configuration coverage mismatch; missing={missing}, extra={extra}"
        )
    directory = root / BINDING_DIRECTORY
    existing = {path.name for path in directory.glob("*.json")} if directory.is_dir() else set()
    expected_names = {binding_path(root, recorder).name for recorder in discovered}
    if existing != expected_names:
        raise NetworkClaimBindingError(
            "network claim receipt set is incomplete or contains stale extras; "
            f"missing={sorted(expected_names - existing)}, extra={sorted(existing - expected_names)}"
        )
    return {
        recorder: validate_network_claim_binding(
            project_root=root,
            recorder=recorder,
            expected_configuration=expected_configurations[recorder],
            cache_path=cache_path,
        )
        for recorder in discovered
    }
