"""Entry point for ``python -m ailm``.

Runs the offline newcomer demo (see :mod:`ailm.demo`): the committed
gate-verdict table, then a live Stage 2B G2P decode of one word. Both steps
are read-only against committed, vendored data — no network access, no
re-measurement, no ledger writes. The same demo is also reachable as the
``ailm-demo`` console script (``[project.scripts]`` in ``pyproject.toml``),
which calls the identical :func:`ailm.demo.main`.
"""

from __future__ import annotations

from ailm.demo import main

if __name__ == "__main__":
    raise SystemExit(main())
