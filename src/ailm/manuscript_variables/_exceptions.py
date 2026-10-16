"""Exception types for manuscript-variable generation."""

from __future__ import annotations

from ailm.reference_presentation import (
    DuplicateInputKeyError as DuplicateInputKeyError,
)
from ailm.reference_presentation import (
    InvalidReferenceSourceError as InvalidReferenceSourceError,
)
from ailm.value_formatting import ManuscriptVariableError


class MissingClaimError(ManuscriptVariableError):
    """Raised when the manuscript needs a claim the ledger does not have.

    Fail loudly rather than emitting an unresolved ``{{TOKEN}}`` or, worse, a
    plausible-looking default. An unresolved token in a rendered PDF is a
    fabricated result.
    """


class DuplicateManuscriptTokenError(ManuscriptVariableError):
    """Raised when two independent producers attempt to own one token name."""


class InvalidLiteralVariableError(ManuscriptVariableError):
    """Raised when ``data/manuscript_literals.yaml`` is absent or malformed."""


class InvalidLiteratureSearchError(ManuscriptVariableError):
    """Raised when the bounded literature-search record cannot support prose."""
