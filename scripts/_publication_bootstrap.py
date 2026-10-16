"""Stdlib-first bounded publication bootstrap helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pwd
import runpy
import socket
import sys
from dataclasses import asdict
from pathlib import Path
from types import CodeType, FrameType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ailm._publication_types import JSONMap, JSONValue, T
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).absolute().parents[1]


def write_report(path: Path, value: T) -> None:
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    with path.open("xb") as stream:
        stream.write(encoded)


def audited_startup(root: Path) -> tuple[Path, JSONMap]:
    """Use existing release startup audit without importing current project code."""
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise ValueError("publication bootstrap requires -I -S -B")
    specification = importlib.util.spec_from_file_location(
        "_publication_startup_audit", root / "scripts/verify_release.py"
    )
    if specification is None or specification.loader is None:
        raise ValueError("cannot load trusted startup audit")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    error = module._project_python_error(root)
    if error:
        raise ValueError(error)
    state = module._venv_startup_state(root)
    if state["integrity_ok"] is not True:
        raise ValueError(f"unsafe dependency startup: {state}")
    site = module._venv_site_packages(root)
    return site, state


def install_policy(mode: str, endpoint: str, source: Path, trusted: Path) -> set[str]:
    port = urlsplit(endpoint).port
    executed: set[str] = set()
    denied_process = {
        "subprocess.Popen",
        "os.system",
        "os.posix_spawn",
        "os.fork",
        "os.forkpty",
        "os.exec",
    }
    dns = {
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
        "socket.getnameinfo",
    }

    def hook(event: str, args: tuple[CodeType | socket.socket | JSONValue, ...]) -> None:
        if event in denied_process:
            raise PermissionError(f"publication policy denies {event}")
        if event == "exec":
            filename = getattr(args[0], "co_filename", "")
            if filename and not filename.startswith("<"):
                path = Path(filename).absolute()
                for folder in ("src", "scripts"):
                    current = trusted / folder
                    if current in path.parents and source != trusted:
                        raise PermissionError(
                            f"current project execution in original process: {path}"
                        )
                if source in path.parents:
                    executed.add(str(path))
        if event in {"socket.connect", "socket.sendto"}:
            address = args[1]
            if (
                mode != "live"
                or not isinstance(address, tuple)
                or address[:2] != ("127.0.0.1", port)
            ):
                raise PermissionError(f"publication policy denies {event}: {address}")
        if event in dns:
            host = args[0] if args else None
            if mode != "live" or host != "127.0.0.1":
                raise PermissionError(f"publication policy denies {event}")
            if event == "socket.getaddrinfo" and args[1] != port:
                raise PermissionError("publication policy denies different resolution port")

    sys.addaudithook(hook)
    # A live reference to the set permits reporting execs, including transient
    # recorder modules that the original loader never inserts into sys.modules.
    return executed


def configure_imports(source: Path, trusted: Path, site: Path) -> None:
    sys.prefix = str(trusted / ".venv")
    sys.exec_prefix = sys.prefix
    sys.path[:] = [
        str(source / "src"),
        str(source / "scripts"),
        str(site),
        *[
            p
            for p in sys.path
            if p not in {str(trusted), str(trusted / "src"), str(trusted / "scripts"), str(site)}
        ],
    ]
    script_namespace = sys.modules.get("scripts")
    if source != trusted and script_namespace is not None:
        script_namespace.__path__ = [str(source / "scripts")]


def origins(source: Path, executed: set[str]) -> dict[str, str]:
    result = {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name == "ailm" or name.startswith("ailm.") or name.startswith("_ailm_replay_config_"):
            if not filename:
                raise ValueError(f"project module has no origin: {name}")
            path = Path(filename).absolute()
            if source not in path.parents:
                raise ValueError(f"project module outside snapshot: {name}: {path}")
            result[name] = str(path)
    for index, filename in enumerate(sorted(executed)):
        result[f"executed:{index}"] = filename
    if not result:
        raise ValueError("empty original module origins")
    return result


def internal_original(arguments: list[str]) -> int:
    source, site_arg, mode, report_arg, endpoint = arguments
    root = Path(source).absolute()
    site, startup = audited_startup(PROJECT_ROOT)
    if site != Path(site_arg).absolute() or mode not in {"offline", "live", "strict-current"}:
        raise ValueError("invalid original launch roots/mode")
    policy = "offline" if mode == "strict-current" else mode
    executed = install_policy(policy, endpoint, root, PROJECT_ROOT)
    configure_imports(root, PROJECT_ROOT, site)
    target = (
        root
        / "scripts"
        / (
            "record_external_benchmark_claims.py"
            if mode == "live"
            else "verify_network_claim_bindings.py"
        )
    )
    sys.argv = [str(target), *(["--verify-cache", "25"] if mode == "live" else [])]
    exit_code = 0
    serving: JSONMap = {}

    def capture_live_return(frame: FrameType, event: str, result: T) -> None:
        if (
            event == "return"
            and frame.f_code.co_filename == str(target)
            and frame.f_code.co_name == "_verify_cache"
        ):
            local = frame.f_locals
            serving.update(
                json.loads(
                    json.dumps(
                        {
                            "model_identities": dict(local.get("identities", {})),
                            "strata": sorted(local.get("verified_strata", set())),
                            "checked": local.get("checked", 0),
                            "mismatches": len(local.get("mismatches", [])),
                            "unreachable": local.get("unreachable", 0),
                            "return_code": result,
                        }
                    )
                )
            )

    if mode == "live":
        sys.setprofile(capture_live_return)
    try:
        runpy.run_path(str(target), run_name="__main__")
    except SystemExit as error:
        if error.code is None:
            exit_code = 0
        elif isinstance(error.code, int):
            exit_code = error.code
        else:
            print(error.code, file=sys.stderr, flush=True)
            exit_code = 1
    finally:
        if mode == "live":
            sys.setprofile(None)
        report = {
            "policy": policy,
            "endpoint": endpoint,
            "source": str(root),
            "verifier_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "interpreter": str(Path(sys.executable).absolute()),
            "flags": {
                "isolated": sys.flags.isolated,
                "no_site": sys.flags.no_site,
                "dont_write_bytecode": sys.flags.dont_write_bytecode,
            },
            "sys_path": list(sys.path),
            "dependency_startup": startup,
            "module_origins": origins(root, executed),
            "exit_code": exit_code,
            "actual_argv": list(sys.argv),
            "serving": serving if mode == "live" else None,
        }
        write_report(Path(report_arg), report)
    return exit_code


def sanitized_environment(endpoint: str, work: Path) -> dict[str, str]:
    home = pwd.getpwuid(os.getuid()).pw_dir
    return {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
        "HOME": home,
        "TMPDIR": str(work),
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(work / "matplotlib"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TZ": "UTC",
        "PYTHONHASHSEED": "0",
        "AILM_OLLAMA_ENDPOINT": endpoint,
    }


def dependency_state(root: Path) -> JSONMap:
    from ailm.publication_derivative import canonical_bytes, digest, file_identity

    site, startup = audited_startup(root)
    inventory = {}
    for directory, directories, files in os.walk(site, followlinks=False):
        for name in directories:
            if (Path(directory) / name).is_symlink():
                raise ValueError("symlink dependency directory")
        for name in files:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("symlink dependency file")
            inventory[path.relative_to(site).as_posix()] = asdict(file_identity(path))
    binary = Path(sys.executable).resolve(strict=True)
    return {
        "integrity_ok": True,
        "startup": startup,
        "site_packages": str(site),
        "inventory_sha256": digest(canonical_bytes(inventory)),
        "file_count": len(inventory),
        "bytecode_policy": "all-regular-executable-caches-included-v1",
        "interpreter": str(Path(sys.executable).absolute()),
        "interpreter_binary": asdict(file_identity(binary)),
        "venv_configuration": asdict(file_identity(root / ".venv/pyvenv.cfg")),
    }


def internal_dependencies(arguments: list[str]) -> int:
    """Emit an actual current dependency identity in an audited isolated process."""
    if arguments:
        raise ValueError("dependency capture takes no arguments")
    site, _ = audited_startup(PROJECT_ROOT)
    configure_imports(PROJECT_ROOT, PROJECT_ROOT, site)
    print(json.dumps(dependency_state(PROJECT_ROOT), sort_keys=True, allow_nan=False))
    return 0
