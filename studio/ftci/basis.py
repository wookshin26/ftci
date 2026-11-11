"""Local function dictionaries used by FTCI models."""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from .expressions import Expression


class Basis:
    """A vector-valued dictionary evaluated on the complete input vector.

    Expression dictionaries are portable. A callable dictionary is useful for
    local work, but cannot be exported unless equivalent expressions are given.
    Declared dependencies are a contract the callable's author must satisfy.
    """

    def __init__(self, expressions: Sequence[str], name: str = ""):
        if isinstance(expressions, str) or not isinstance(expressions, (list, tuple)):
            raise ValueError("Basis expressions must be a list or tuple of strings.")
        if not 1 <= len(expressions) <= 65536:
            raise ValueError("A basis must contain between 1 and 65536 expressions.")
        self.expressions = tuple(expressions)
        self._expressions = tuple(Expression(text) for text in self.expressions)
        self.input_indices = tuple(sorted({i for expression in self._expressions for i in expression.input_indices}))
        self.dimension = len(self._expressions)
        self.name = str(name)
        self._callable: Callable | None = None

    @classmethod
    def from_callable(cls, fn: Callable, input_indices: Sequence[int], dimension: int,
                      name: str = "", expressions: Sequence[str] | None = None) -> "Basis":
        if not callable(fn):
            raise ValueError("Basis evaluator must be callable.")
        if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)) or not 1 <= dimension <= 65536:
            raise ValueError("Basis dimension must be a positive integer below 65537.")
        indices = tuple(input_indices)
        if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 or i >= 4096 for i in indices):
            raise ValueError("Basis input indices must be nonnegative integers below 4096.")
        indices = tuple(sorted(set(int(i) for i in indices)))
        if expressions is not None:
            result = cls(expressions, name=name)
            if result.dimension != dimension or result.input_indices != indices:
                raise ValueError("Portable expressions must match the declared dimension and dependencies.")
        else:
            result = cls.__new__(cls)
            result.expressions = None
            result._expressions = ()
            result.input_indices = indices
            result.dimension = int(dimension)
            result.name = str(name)
        result._callable = fn
        return result

    @property
    def portable(self) -> bool:
        return self.expressions is not None

    def __call__(self, x: Sequence[float] | np.ndarray) -> np.ndarray:
        if np.iscomplexobj(x):
            raise ValueError("Basis inputs must be real.")
        values = np.asarray(x, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("Basis inputs must be a finite one-dimensional vector.")
        if self.input_indices and self.input_indices[-1] >= len(values):
            raise ValueError(f"Basis {self.name!r} requires input x{self.input_indices[-1]}.")
        if self._callable is None:
            result = np.array([expression(values) for expression in self._expressions], dtype=float)
        else:
            raw = self._callable(values)
            if np.iscomplexobj(raw):
                raise ValueError("Basis evaluators must return real values.")
            result = np.asarray(raw, dtype=float)
        if result.shape != (self.dimension,) or not np.all(np.isfinite(result)):
            raise ValueError(f"Basis {self.name!r} must return {self.dimension} finite real values.")
        return result

    def to_spec(self) -> dict:
        if not self.portable:
            raise ValueError("Callable basis cannot be saved portably. Supply equivalent expressions when constructing it.")
        return {"name": self.name, "expressions": list(self.expressions)}

    @classmethod
    def from_spec(cls, spec: Mapping) -> "Basis":
        if not isinstance(spec, dict) or set(spec) - {"name", "expressions"}:
            raise ValueError("Invalid basis specification.")
        if "expressions" not in spec or not isinstance(spec.get("name", ""), str):
            raise ValueError("A basis specification requires expressions and an optional string name.")
        return cls(spec["expressions"], name=spec.get("name", ""))

    def __repr__(self) -> str:
        return f"Basis(name={self.name!r}, dimension={self.dimension}, input_indices={self.input_indices})"
