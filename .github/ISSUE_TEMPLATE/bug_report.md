---
name: Bug report
about: Report something that is broken — a failing gate, an incorrect result, a crash
title: "[BUG] "
labels: bug
assignees: ''
---

## Describe the bug

A clear, concise description of what is broken.

## What kind of bug is this?

- [ ] A crash or exception
- [ ] A test or gate that fails unexpectedly
- [ ] A result that looks wrong (a number, a figure, a claim)
- [ ] A documentation error (a broken link, a stale command, an outdated path)
- [ ] Something else

## To reproduce

Steps to reproduce the behavior, including the exact command run:

```bash
.venv/bin/python -m pytest tests/test_....py -q
```

## Expected behavior

What you expected to happen instead.

## Environment

- OS:
- Python version (`python --version`):
- Installed with `uv sync`? (yes/no):
- Commit or tag (`git rev-parse HEAD`):

## Additional context

Add any other context, logs, or screenshots. If this concerns a specific
recorded claim or gate, please name the `claim:<label>` or stage involved —
see [`docs/method/style_guide.md`](../../docs/method/style_guide.md) and
[`AGENTS.md`](../../AGENTS.md).

**Security-relevant issue?** Please do not file it here — see
[`SECURITY.md`](../../SECURITY.md) instead.
