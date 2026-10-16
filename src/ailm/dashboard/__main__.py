"""Entry point for ``python -m ailm.dashboard [port] [host]``.

The package entry point owns configuration validation and bind precedence.
The server can serve static assets and configuration before the numerical
engine is imported; a computation endpoint loads that engine when needed.

## Where the port and host come from

Three sources, most specific first: a command-line argument, then
``AILM_DASHBOARD_PORT`` / ``AILM_DASHBOARD_HOST``, then the shipped default in
``ailm.dashboard.config``.

This module holds no defaults of its own. It used to — ``DEFAULT_PORT = 8080``
and ``DEFAULT_HOST = "127.0.0.1"``, passed to the server unconditionally — which
meant the documented environment overrides were read, validated, and then
discarded before the socket was opened. Nothing failed; the server came up on
8080 and the operator believed they had moved it. A default written twice is a
default that will disagree with itself, and this is the copy the config module
exists to not have.

The server accepts only explicit loopback bind addresses. A per-process token is
issued to the locally served page as a strict, HTTP-only cookie; every API route
requires that token and rejects cross-origin browser requests.
"""

from __future__ import annotations

import sys

from ailm.dashboard.config import DashboardConfig, load_config, require_loopback_host
from ailm.dashboard.server import run_dashboard


def resolve_bind(args: list[str], config: DashboardConfig) -> tuple[str, int]:
    """Resolve CLI arguments over one already-validated configuration.

    This pure boundary is also useful to embedders that need to inspect a bind
    before opening a socket.  It deliberately receives the configuration that
    will be handed to the server, so environment loading still has one owner.
    """
    port = config.port
    if args:
        try:
            port = int(args[0])
        except ValueError as error:
            raise ValueError(f"port must be an integer, got {args[0]!r}") from error
        if not 1 <= port <= 65535:
            raise ValueError(f"port {port} outside 1-65535")

    host = args[1] if len(args) > 1 else config.host
    require_loopback_host(host)
    return host, port


def main(argv: list[str] | None = None) -> int:
    """Parse ``[port] [host]`` over the environment and start the server.

    Args:
        argv: Arguments after the module name. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit status: 0 on a clean shutdown, 2 on a bad argument or an
        unusable configuration.
    """
    args = sys.argv[1:] if argv is None else argv

    try:
        config = load_config()
    except ValueError as error:
        # A refused override is reported, never defaulted around: a typo in a
        # limit is exactly the case where a quiet fallback hides the mistake.
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        host, port = resolve_bind(args, config)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    run_dashboard(host=host, port=port, config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
