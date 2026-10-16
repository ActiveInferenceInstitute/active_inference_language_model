"""Stdlib-first bounded publication controls helpers."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

from ._publication_bootstrap import (
    PROJECT_ROOT,
    audited_startup,
    configure_imports,
    install_policy,
    write_report,
)


def internal_transport(arguments: list[str]) -> int:
    operation, report_arg, endpoint = arguments
    # Receivers are created before installing the audit policy. Every actual
    # client operation occurs after it; the positive control omits policy.
    tcp = socket.socket()
    tcp.bind(("127.0.0.1", 0))
    tcp.listen()
    tcp.settimeout(0.15)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    udp.settimeout(0.15)
    if operation != "baseline":
        install_policy("offline", endpoint, PROJECT_ROOT, PROJECT_ROOT)
    denied = False
    client = socket.socket()
    try:
        if operation in {"baseline", "connect"}:
            client.connect(tcp.getsockname())
        elif operation == "connect_ex":
            client.connect_ex(tcp.getsockname())
        elif operation == "dns":
            socket.getaddrinfo("localhost", tcp.getsockname()[1])
        elif operation == "udp":
            client.close()
            client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            client.sendto(b"controlled-payload", udp.getsockname())
        elif operation == "spawn":
            subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    "import socket; socket.create_connection(("
                    + repr("127.0.0.1")
                    + ","
                    + str(tcp.getsockname()[1])
                    + "))",
                ],
                check=True,
                close_fds=True,
            )
        else:
            raise ValueError("unknown transport control")
    except PermissionError:
        denied = True
    reached = False
    try:
        connection, _ = tcp.accept()
        connection.close()
        reached = True
    except TimeoutError:
        pass  # Deliberate bounded absence check on the controlled receiver.
    try:
        udp.recvfrom(1024)
        reached = True
    except TimeoutError:
        pass  # Deliberate bounded absence check on the controlled receiver.
    finally:
        client.close()
        tcp.close()
        udp.close()
    valid = reached and not denied if operation == "baseline" else denied and not reached
    write_report(
        Path(report_arg),
        {"operation": operation, "denied": denied, "reached": reached, "valid": valid},
    )
    print(f"transport {operation}: denied={denied} reached={reached} valid={valid}")
    return 0 if valid else 1


def internal_predicate(arguments: list[str]) -> int:
    source, kind, contract_arg, parent, report_arg = arguments
    site, _ = audited_startup(PROJECT_ROOT)
    configure_imports(PROJECT_ROOT, PROJECT_ROOT, site)
    from ailm.publication_derivative import (
        capture_derivative_state,
        load_candidate_contract,
        load_contract,
    )

    try:
        root = Path(source)
        contract = (
            load_candidate_contract(Path(contract_arg), root, Path(parent))
            if kind == "candidate"
            else load_contract(Path(contract_arg), root)
        )
        state = capture_derivative_state(contract, root)
        report = {"valid": True, "state_fingerprint": state.fingerprint}
        status = 0
    except (OSError, ValueError) as error:
        report = {"valid": False, "error": str(error)}
        status = 1
    write_report(Path(report_arg), report)
    print(json.dumps(report, sort_keys=True))
    return status
