"""JSON/YAML loading helpers and the duplicate-checked variable map."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import (
    Any,
)

import yaml

from ailm.reference_presentation import _unique_json_object as _unique_json_object
from ailm.value_formatting import ManuscriptVariableError

from ._exceptions import (
    DuplicateInputKeyError,
    DuplicateManuscriptTokenError,
)

DIGEST_SHORT_LENGTH = 12


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader whose mappings reject duplicate keys."""


def _construct_unique_yaml_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise DuplicateInputKeyError(f"duplicate YAML key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_yaml_mapping,
)


def _load_unique_yaml(path: Path) -> Any:
    """Parse a YAML input without silent duplicate-key overwrite."""
    loader = _UniqueKeyLoader(path.read_text(encoding="utf-8"))
    try:
        # Drive the SafeLoader subclass directly.  Besides preserving the
        # duplicate-key constructor above, this makes the safety boundary
        # explicit to static analyzers that conservatively flag every use of
        # the generic ``yaml.load`` convenience function.
        return loader.get_single_data()
    except yaml.YAMLError as exc:
        raise ManuscriptVariableError(f"invalid YAML in {path}: {exc}") from exc
    finally:
        loader.dispose()


class _StrictVariableMap(dict[str, str]):
    """A token map that refuses silent overwrites.

    A plain ``dict`` made the order of assignments semantically significant:
    a derived token could overwrite a recorded claim, or a claim named
    ``foo_run`` could overwrite the provenance companion generated for ``foo``.
    Both outputs look valid.  Refusing the second write makes producer identity
    a runtime invariant rather than an AST convention maintained by one test.
    """

    def __setitem__(self, key: str, value: str) -> None:
        if key in self:
            raise DuplicateManuscriptTokenError(
                f"manuscript token {key!r} has more than one producer; "
                "rename the derived/configuration token or remove the duplicate claim"
            )
        super().__setitem__(key, value)

    def add_many(self, produced: Mapping[str, str], *, producer: str) -> None:
        """Merge one named producer only when every token is still unowned."""
        duplicates = sorted(self.keys() & produced.keys())
        if duplicates:
            raise DuplicateManuscriptTokenError(
                f"manuscript token(s) {duplicates!r} have more than one producer; "
                f"{producer} attempted to shadow an existing token"
            )
        # ``dict.update`` can bypass a subclass's ``__setitem__``. The complete
        # collision check above is therefore the enforcement point, not an
        # assumption about CPython dispatch.
        super().update(produced)
