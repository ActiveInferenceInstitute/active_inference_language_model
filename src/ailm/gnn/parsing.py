"""GNN source text to a :class:`~ailm.gnn.types.GnnSpec`.

Reading only: nothing here inspects a live agent. Conformance is a separate
concern and lives in :mod:`ailm.gnn.conformance`, so a spec can be parsed and
inspected without pymdp, numpy shapes, or a built model anywhere in reach."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ailm.gnn.types import (
    _LEARNED_FORM,
    _MODEL_PARAM_RE,
    _ONTOLOGY_RE,
    _PARAM_FORM_RE,
    _REQUIRED_SECTIONS,
    _SIZED_FORMS,
    _VARIABLE_RE,
    GNN_SECTIONS,
    GnnConnection,
    GnnSpec,
    GnnSpecError,
)

# ── PARSING ───────────────────────────────────────────────────────────────────


def _fail(origin: str, lineno: int, line: str, problem: str) -> GnnSpecError:
    """Build a :class:`GnnSpecError` that names the offending line."""
    return GnnSpecError(f"{origin}:{lineno}: {problem}\n    offending line: {line!r}")


def _split_sections(text: str, origin: str) -> dict[str, tuple[int, str]]:
    """Split GNN Markdown into ``section -> (header_lineno, body)``.

    Content before the first ``## `` header is a title/preamble and is ignored.

    Raises:
        GnnSpecError: If the same section appears twice, a section name is not
            in :data:`GNN_SECTIONS`, or no section header exists at all.
    """
    sections: dict[str, tuple[int, str]] = {}
    current: str | None = None
    current_lineno = 0
    buffer: list[str] = []

    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.startswith("## "):
            if current is not None:
                sections[current] = (current_lineno, "\n".join(buffer).strip())
            name = line[3:].strip()
            if name not in GNN_SECTIONS:
                raise _fail(
                    origin,
                    lineno,
                    line,
                    f"unknown GNN section {name!r}. Known sections: "
                    f"{', '.join(GNN_SECTIONS)}. A misspelled section name would "
                    f"silently drop the whole block, so it is rejected here.",
                )
            if name in sections:
                raise _fail(origin, lineno, line, f"duplicate GNN section {name!r}")
            current, current_lineno, buffer = name, lineno, []
        elif current is not None:
            buffer.append(line)

    if current is not None:
        sections[current] = (current_lineno, "\n".join(buffer).strip())

    if not sections:
        raise GnnSpecError(f"{origin}: no '## ' section headers found — this is not a GNN spec.")
    return sections


def _content_lines(body: str, header_lineno: int) -> list[tuple[int, str]]:
    """Yield ``(lineno, line)`` for non-blank, non-comment lines of a section."""
    out: list[tuple[int, str]] = []
    for offset, raw in enumerate(body.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append((header_lineno + offset, line))
    return out


def _strip_comment(line: str) -> str:
    """Drop a trailing ``#`` comment from a declaration line."""
    return line.split("#", 1)[0].strip()


def _parse_state_space(
    body: str, header_lineno: int, origin: str
) -> dict[str, tuple[tuple[int, ...], str]]:
    """Parse ``StateSpaceBlock`` into ``name -> (dims, dtype)``.

    Raises:
        GnnSpecError: On a malformed ``name[dims,type=dtype]`` line, a
            non-positive-integer dimension, or a duplicate variable name.
    """
    variables: dict[str, tuple[tuple[int, ...], str]] = {}
    for lineno, raw in _content_lines(body, header_lineno):
        line = _strip_comment(raw)
        if not line:
            continue
        match = _VARIABLE_RE.match(line)
        if match is None:
            raise _fail(
                origin,
                lineno,
                raw,
                "malformed variable declaration; expected "
                "'name[dim,...,type=dtype]' with a closing bracket",
            )
        name = match.group("name")
        if name in variables:
            raise _fail(
                origin,
                lineno,
                raw,
                f"duplicate variable {name!r} in StateSpaceBlock — the second "
                f"declaration would silently win",
            )
        dims: list[int] = []
        dtype = "float"
        seen_type = False
        for token in match.group("body").split(","):
            token = token.strip()
            if not token:
                raise _fail(origin, lineno, raw, "empty dimension token in the bracket")
            if token.startswith("type="):
                dtype = token[len("type=") :].strip()
                if not dtype:
                    raise _fail(origin, lineno, raw, "empty 'type=' specification")
                seen_type = True
                continue
            if seen_type:
                raise _fail(origin, lineno, raw, "dimension token after the 'type=' token")
            if not token.isdigit() or int(token) < 1:
                raise _fail(
                    origin,
                    lineno,
                    raw,
                    f"dimension {token!r} is not a positive integer — symbolic "
                    f"dimensions are not accepted, because a checker cannot bind "
                    f"them to a real matrix",
                )
            dims.append(int(token))
        if not dims:
            raise _fail(origin, lineno, raw, "variable declares no dimensions")
        variables[name] = (tuple(dims), dtype)

    if not variables:
        raise GnnSpecError(f"{origin}: StateSpaceBlock declares no variables — nothing to check.")
    return variables


def _parse_variable_group(group: str) -> tuple[str, ...]:
    """Parse ``X`` or ``(X,Y,Z)`` into a tuple of names."""
    group = group.strip()
    if group.startswith("(") and group.endswith(")"):
        return tuple(part.strip() for part in group[1:-1].split(",") if part.strip())
    return (group,) if group else ()


def _parse_connections(
    body: str, header_lineno: int, origin: str, declared: Sequence[str]
) -> list[GnnConnection]:
    """Parse the ``Connections`` block.

    Raises:
        GnnSpecError: If a line carries no ``>``/``-`` operator, an endpoint is
            empty, or an endpoint is not a declared variable.
    """
    known = set(declared)
    connections: list[GnnConnection] = []
    for lineno, raw in _content_lines(body, header_lineno):
        line = _strip_comment(raw)
        if not line:
            continue
        if ">" in line:
            operator, directed = ">", True
        elif "-" in line:
            operator, directed = "-", False
        else:
            raise _fail(
                origin,
                lineno,
                raw,
                "no connection operator found; use 'X>Y' (directed) or 'X-Y'",
            )
        head, _, tail = line.partition(operator)
        source = _parse_variable_group(head)
        target = _parse_variable_group(tail)
        if not source or not target:
            raise _fail(origin, lineno, raw, "connection has an empty endpoint")
        for name in (*source, *target):
            if name not in known:
                raise _fail(
                    origin,
                    lineno,
                    raw,
                    f"connection references undeclared variable {name!r}; "
                    f"declare it in StateSpaceBlock first",
                )
        connections.append(GnnConnection(source=source, target=target, directed=directed, raw=line))
    return connections


def _parse_initial_parameterization(
    body: str, header_lineno: int, origin: str, declared: Sequence[str]
) -> dict[str, tuple[str, str]]:
    """Parse ``InitialParameterization`` into ``name -> (form, args)``.

    Only structural forms are accepted. An inlined matrix literal is rejected on
    purpose: the learned entries are a derived artifact reproduced by a
    generator, and a copy of them inside the spec is a second source of truth
    that will drift from the first.

    Raises:
        GnnSpecError: On a non-structural line, an unknown form, a sized form
            whose argument is not a positive integer, or a form naming an
            undeclared variable.
    """
    known = set(declared)
    forms: dict[str, tuple[str, str]] = {}
    for lineno, raw in _content_lines(body, header_lineno):
        match = _PARAM_FORM_RE.match(_strip_comment(raw))
        if match is None:
            raise _fail(
                origin,
                lineno,
                raw,
                "InitialParameterization accepts only structural forms — "
                f"{'/'.join(sorted(_SIZED_FORMS))}(n) or {_LEARNED_FORM}(k=v, ...). "
                "Inlining a matrix literal is rejected: learned entries are a "
                "derived artifact, so name the generator and the vendored "
                "artifact instead.",
            )
        name = match.group("name")
        form = match.group("form")
        args = match.group("args").strip()
        if name not in known:
            raise _fail(
                origin,
                lineno,
                raw,
                f"parameterization names undeclared variable {name!r}",
            )
        if name in forms:
            raise _fail(origin, lineno, raw, f"duplicate parameterization for {name!r}")
        if form in _SIZED_FORMS:
            if not args.isdigit() or int(args) < 1:
                raise _fail(
                    origin,
                    lineno,
                    raw,
                    f"{form} takes a single positive integer size, got {args!r}",
                )
        elif form != _LEARNED_FORM:
            raise _fail(
                origin,
                lineno,
                raw,
                f"unknown structural form {form!r}; expected one of "
                f"{', '.join(sorted(_SIZED_FORMS | {_LEARNED_FORM}))}",
            )
        elif not args:
            raise _fail(
                origin,
                lineno,
                raw,
                f"{_LEARNED_FORM}(...) must name its provenance, e.g. "
                f"{_LEARNED_FORM}(generator=scripts/generate_counts.py, "
                f"artifact=vendor/brown_counts.json)",
            )
        forms[name] = (form, args)
    return forms


def parse_learned_arguments(args: str) -> dict[str, str]:
    """Split a ``Learned(...)`` argument list into ``key -> value``.

    The arguments carry real declarations — ``generator``, ``artifact``,
    ``normalization``, ``estimator``, ``key`` — that were stored as one opaque
    string and read by nothing. Exposed rather than kept private because the
    provenance keys are resolved against the repository by the test suite, and
    two parsers for one syntax is one parser too many.

    Args:
        args: The text between the parentheses of a ``Learned`` form.

    Returns:
        Every ``key=value`` entry. A positional entry carrying no ``=`` is
        dropped: it names nothing and there is no key under which to report it.
    """
    entries: dict[str, str] = {}
    for entry in args.split(","):
        key, separator, value = entry.partition("=")
        if not separator:
            continue
        entries[key.strip()] = value.strip()
    return entries


def _parse_ontology(
    body: str, header_lineno: int, origin: str, declared: Sequence[str]
) -> dict[str, str]:
    """Parse ``ActInfOntologyAnnotation`` into ``name -> term``.

    Raises:
        GnnSpecError: On a malformed line, a duplicate annotation, an annotation
            naming an undeclared variable, or a declared variable left
            un-annotated. The last case matters most: an unannotated slot is a
            slot whose semantic role nothing pins, which is the entire reason
            AILM carries these specs.
    """
    known = set(declared)
    ontology: dict[str, str] = {}
    for lineno, raw in _content_lines(body, header_lineno):
        match = _ONTOLOGY_RE.match(_strip_comment(raw))
        if match is None:
            raise _fail(
                origin,
                lineno,
                raw,
                "malformed ontology annotation; expected 'Variable=OntologyTerm'",
            )
        name, term = match.group("name"), match.group("term")
        if name not in known:
            raise _fail(
                origin,
                lineno,
                raw,
                f"ontology annotates undeclared variable {name!r}; declare it in "
                f"StateSpaceBlock or delete the annotation",
            )
        if name in ontology:
            raise _fail(origin, lineno, raw, f"duplicate ontology annotation for {name!r}")
        ontology[name] = term

    unannotated = [name for name in declared if name not in ontology]
    if unannotated:
        raise GnnSpecError(
            f"{origin}: declared but un-annotated variable(s) "
            f"{unannotated} — every declared variable needs an entry in "
            f"ActInfOntologyAnnotation, otherwise nothing pins its role."
        )
    return ontology


def _coerce_scalar(text: str) -> Any:
    """Coerce a ``ModelParameters`` value to int or float where possible."""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _parse_model_parameters(body: str, header_lineno: int, origin: str) -> dict[str, Any]:
    """Parse ``ModelParameters`` into ``key -> value``.

    Raises:
        GnnSpecError: On a line that is neither ``key: value`` nor ``key = value``,
            or on a duplicate key.
    """
    parameters: dict[str, Any] = {}
    for lineno, raw in _content_lines(body, header_lineno):
        match = _MODEL_PARAM_RE.match(_strip_comment(raw))
        if match is None:
            raise _fail(
                origin,
                lineno,
                raw,
                "malformed ModelParameters entry; expected 'key: value'",
            )
        key = match.group("key")
        if key in parameters:
            raise _fail(origin, lineno, raw, f"duplicate ModelParameters key {key!r}")
        parameters[key] = _coerce_scalar(match.group("value").strip())
    return parameters


def parse_gnn_string(content: str, origin: str = "<string>") -> GnnSpec:
    """Parse GNN Markdown text into a :class:`GnnSpec`.

    Args:
        content: The full spec text.
        origin: Label used in error messages (a path, usually).

    Returns:
        A fully-populated :class:`GnnSpec`.

    Raises:
        GnnSpecError: On any malformed input. Never returns a partial spec.
    """
    sections = _split_sections(content, origin)

    missing = [name for name in _REQUIRED_SECTIONS if name not in sections]
    if missing:
        raise GnnSpecError(
            f"{origin}: missing required section(s) {missing}. "
            f"Without ActInfOntologyAnnotation in particular there is nothing to "
            f"pin: the whole point of the spec is naming each matrix's role."
        )

    raw_sections = {name: body for name, (_, body) in sections.items()}

    model_lineno, model_body = sections["ModelName"]
    model_name = model_body.strip()
    if not model_name:
        raise GnnSpecError(f"{origin}:{model_lineno}: ModelName section is empty")

    variables = _parse_state_space(*_pair(sections["StateSpaceBlock"]), origin)
    declared = list(variables)

    ontology = _parse_ontology(*_pair(sections["ActInfOntologyAnnotation"]), origin, declared)

    connections: list[GnnConnection] = []
    if "Connections" in sections:
        connections = _parse_connections(*_pair(sections["Connections"]), origin, declared)

    parameterization: dict[str, tuple[str, str]] = {}
    if "InitialParameterization" in sections:
        parameterization = _parse_initial_parameterization(
            *_pair(sections["InitialParameterization"]), origin, declared
        )

    parameters: dict[str, Any] = {}
    if "ModelParameters" in sections:
        parameters = _parse_model_parameters(*_pair(sections["ModelParameters"]), origin)

    return GnnSpec(
        model_name=model_name,
        variables=variables,
        connections=connections,
        ontology=ontology,
        parameters=parameters,
        parameterization=parameterization,
        sections=raw_sections,
        origin=origin,
    )


def _pair(section: tuple[int, str]) -> tuple[str, int]:
    """Reorder a ``(lineno, body)`` section tuple to ``(body, lineno)``."""
    lineno, body = section
    return body, lineno


def parse_gnn_spec(path: str | Path) -> GnnSpec:
    """Parse a GNN spec file.

    Args:
        path: Path to a ``.md`` GNN spec.

    Returns:
        A fully-populated :class:`GnnSpec`.

    Raises:
        GnnSpecError: On malformed content, or if the file cannot be read.
    """
    file_path = Path(path)
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GnnSpecError(f"cannot read GNN spec {file_path}: {exc}") from exc
    return parse_gnn_string(content, origin=str(file_path))
