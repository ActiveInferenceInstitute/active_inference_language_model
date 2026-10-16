#!/usr/bin/env python3
"""Execute reviewed two-source publication evidence or a nonaccepting candidate."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[1]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    choice = result.add_mutually_exclusive_group(required=True)
    choice.add_argument(
        "--contract", type=Path, help="external exact-byte independently reviewed contract"
    )
    choice.add_argument(
        "--candidate",
        type=Path,
        help="declaration JSON; development evidence NEVER accepts a release",
    )
    result.add_argument(
        "--snapshot-parent",
        type=Path,
        help="existing destination parent, required only with --candidate",
    )
    result.add_argument("--mode", required=True, choices=("offline", "live"))
    result.add_argument(
        "--output", required=True, type=Path, help="fresh nonexisting attestation file"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.candidate is not None and args.snapshot_parent is None:
        parser().error("--candidate requires --snapshot-parent")
    if args.contract is not None and args.snapshot_parent is not None:
        parser().error(
            "reviewed contract controls snapshot destination; no --snapshot-parent override"
        )
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts._publication_execution import audited_startup, configure_imports, run_derivative

    site, _ = audited_startup(PROJECT_ROOT)
    configure_imports(PROJECT_ROOT, PROJECT_ROOT, site)
    from ailm.publication_derivative import (
        capture_derivative_state,
        load_candidate_contract,
        load_contract,
        validate_attestation,
    )

    try:
        contract_path = (args.candidate or args.contract).absolute()
        contract = (
            load_candidate_contract(contract_path, PROJECT_ROOT, args.snapshot_parent.absolute())
            if args.candidate
            else load_contract(contract_path, PROJECT_ROOT)
        )
        run_derivative(
            contract,
            PROJECT_ROOT,
            args.mode,
            args.output.absolute(),
            contract_path,
            candidate=args.candidate is not None,
        )
        state = capture_derivative_state(contract, PROJECT_ROOT)
        result = validate_attestation(args.output.absolute(), contract, state, args.mode)
        if args.candidate:
            expected = ("candidate development evidence cannot satisfy release acceptance",)
            if result.errors != expected or result.accepted:
                raise ValueError(f"candidate proof invalid: {result.errors}")
            print(
                f"candidate {args.mode} evidence emitted; accepted=false: {args.output.absolute()}"
            )
        else:
            if not result.accepted:
                raise ValueError(f"reviewed proof refused: {result.errors}")
            print(f"publication derivative {args.mode} proof verified: {result.attestation_sha256}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"publication derivative verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        # Re-exec before project or dependency imports; never implicit site startup
        # in the trusted execution process. Standard invocation remains ergonomic.
        os.execv(
            str(PROJECT_ROOT / ".venv/bin/python"),
            [
                str(PROJECT_ROOT / ".venv/bin/python"),
                "-I",
                "-S",
                "-B",
                str(Path(__file__).absolute()),
                *sys.argv[1:],
            ],
        )
    raise SystemExit(main())
