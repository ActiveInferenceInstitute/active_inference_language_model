"""The symbol registry means what its caption says.

``[@tbl:notation]`` promises that **every registered symbol has exactly one
meaning throughout the paper**, and singles out the case it was built to
prevent: the predictive prior is $\\rho_t$ so that $\\pi$ can mean policy and
nothing else. That promise was false when it was written — one proposition set
the predictive prior to $\\pi_{t+1}$, the exact clash the caption names — and a
promise a reader can rely on is worth a test rather than a proofread.

The rule is deliberately narrow. It binds only the symbols the table registers,
because that is the only claim the caption makes; $\\alpha$ is reused on purpose
and the prose says so, which is why it is not registered and not scanned here.

The scan covers the manuscript sources and ``src/ailm/figures/*.py``. The figure
modules type their mathtext directly, so a symbol drawn on a panel never passes
through the manuscript and would otherwise be the one place the rule cannot see —
which is where the clash survived once already.

Guards on the guard: the registry must parse to the symbols the rule reasons
about, the file list must be non-empty (an empty list passes any rule), and a
planted misuse must be caught (the rule can fail).
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANUSCRIPT = PROJECT_ROOT / "docs" / "manuscript"
FIGURES = PROJECT_ROOT / "src" / "ailm" / "figures"

#: Files excluded from the scan: documentation about the manuscript, not of it.
#: ``SYNTAX.md`` describes the registry rather than using it, so its mention of
#: $\pi$ is commentary and would be a false positive.
_EXCLUDED = {"SYNTAX.md", "AGENTS.md", "README.md", "preamble.md"}

#: A registry row's first cell: ``| $\pi$ | policy (sequence of actions) | ... |``.
#: The pattern takes whatever sits between the first pair of dollar signs rather
#: than a backslash-led name, because two of the six rows are neither: $q_t$ has
#: no backslash and the matrix row packs four symbols into one cell.
_ROW_RE = re.compile(r"^\|\s*\$([^$|]+)\$\s*\|", re.MULTILINE)

#: Uses of $\pi$ that are the *policy*, and therefore correct. Policy $\pi$
#: appears bare, applied to a distribution, or conditioned on — never with a
#: time subscript, which is what makes the misuse mechanically separable: a
#: predictive prior is indexed by the step it predicts, a policy is not.
_PI_MISUSE_RE = re.compile(r"\\pi_\{?[t0-9]")


def _registered_symbols() -> list[str]:
    """The LaTeX symbols listed in the notation table's first column.

    A cell holding several comma-separated symbols contributes each of them, so
    the returned list is the registry's symbols rather than its rows.
    """
    background = (MANUSCRIPT / "02_background.md").read_text(encoding="utf-8")
    table = background.split("{#tbl:notation}")[0]
    symbols: list[str] = []
    for cell in _ROW_RE.findall(table):
        symbols.extend(part.strip() for part in cell.split(",") if part.strip())
    return symbols


def _scanned_sources() -> list[Path]:
    """Manuscript prose and the figure modules that type their own mathtext."""
    prose = [p for p in MANUSCRIPT.glob("*.md") if p.name not in _EXCLUDED]
    drawn = [p for p in FIGURES.glob("*.py") if not p.name.startswith("__")]
    return sorted(prose) + sorted(drawn)


def test_the_registry_is_found_and_non_empty():
    """POSITIVE CONTROL: the rule below reasons about symbols this must parse."""
    symbols = _registered_symbols()
    assert r"\pi" in symbols and r"\rho_t" in symbols, (
        f"the notation table parsed to {symbols}; the rule that $\\pi$ is never "
        f"the predictive prior only restates the registry if both are in it"
    )


def test_the_scan_reads_the_manuscript_and_the_figures():
    """POSITIVE CONTROL: an empty file list passes any rule."""
    sources = _scanned_sources()
    prose = [p for p in sources if p.suffix == ".md"]
    drawn = [p for p in sources if p.suffix == ".py"]
    assert len(prose) >= 12, f"only {len(prose)} manuscript files scanned"
    assert len(drawn) >= 5, f"only {len(drawn)} figure modules scanned"


def test_pi_is_never_the_predictive_prior():
    """The one clash [@tbl:notation] names, enforced everywhere a symbol is set.

    ``fig:inference_cycle`` drew the predictive prior as $\\pi_t$ in eight
    mathtext strings while the prose beside it used $\\rho_{t+1}$, because the
    scan stopped at ``docs/manuscript/``. It does not any more.
    """
    registered = _registered_symbols()
    assert r"\pi" in registered and r"\rho_t" in registered
    offenders = []
    for path in _scanned_sources():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _PI_MISUSE_RE.search(line):
                offenders.append(f"{path.name}:{number}")
    assert offenders == [], (
        f"$\\pi$ carries a time subscript at {offenders}, which reads as the "
        f"one-step predictive prior. [@tbl:notation] reserves $\\pi$ for the "
        f"policy and defines the predictive prior as $\\rho_t$ — use that."
    )


def test_a_planted_misuse_is_caught():
    """POSITIVE CONTROL: the proxy can itself fail."""
    assert _PI_MISUSE_RE.search(r"predictive prior is \(\pi_{t+1} = M q_t\)")
    assert not _PI_MISUSE_RE.search(r"minimizes expected free energy $G(\pi)$")
    assert not _PI_MISUSE_RE.search(r"$p(s_t \mid s_{t-1}, \pi)$")
