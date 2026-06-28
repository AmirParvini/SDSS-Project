"""Serialisation of decoded solutions."""

from __future__ import annotations

import json
from typing import List

import numpy as np


def _to_native(value):
    """Make numpy scalars/arrays JSON-serialisable."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)} is not JSON serialisable")


class JsonSolutionWriter:
    """Render decoded solutions as JSON (string or file)."""

    def __init__(self, indent: int = 4, ensure_ascii: bool = False) -> None:
        self._indent = indent
        self._ensure_ascii = ensure_ascii

    def to_string(self, solutions: List[dict]) -> str:
        return json.dumps(
            solutions,
            ensure_ascii=self._ensure_ascii,
            indent=self._indent,
            default=_to_native,
        )

    def to_file(self, solutions: List[dict], path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_string(solutions))
