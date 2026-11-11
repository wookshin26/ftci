"""Small, serializable real expressions without Python ``eval``.

Expressions use ``x0``, ``x1``, ... for input coordinates and familiar math
functions such as ``sin(x0)``. The syntax is deliberately smaller than Python.
"""
from __future__ import annotations

import ast
import operator
import re
from typing import Sequence

import numpy as np


_FUNCTIONS = {
    "sin": (np.sin, 1), "cos": (np.cos, 1), "tan": (np.tan, 1),
    "exp": (np.exp, 1), "log": (np.log, 1), "sqrt": (np.sqrt, 1),
    "abs": (np.abs, 1), "tanh": (np.tanh, 1),
    "sinh": (np.sinh, 1), "cosh": (np.cosh, 1),
    "asin": (np.arcsin, 1), "acos": (np.arccos, 1),
    "atan": (np.arctan, 1), "atan2": (np.arctan2, 2),
    "arcsin": (np.arcsin, 1), "arccos": (np.arccos, 1),
    "arctan": (np.arctan, 1), "log1p": (np.log1p, 1),
    "expm1": (np.expm1, 1), "minimum": (np.minimum, 2),
    "maximum": (np.maximum, 2),
}
_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: np.power,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_COORDINATE = re.compile(r"x(0|[1-9][0-9]*)\Z")
_CONSTANTS = {"pi": float(np.pi), "e": float(np.e)}


class Expression:
    """A validated scalar expression that is safe to store in a model file."""

    def __init__(self, text: str):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("An expression must be a nonempty string.")
        if len(text) > 4096:
            raise ValueError("An expression may contain at most 4096 characters.")
        self.text = text.strip()
        try:
            self._tree = ast.parse(self.text, mode="eval").body
        except (SyntaxError, RecursionError) as exc:
            raise ValueError(f"Invalid expression: {self.text!r}.") from exc
        self._indices: set[int] = set()
        self._nodes = 0
        self._validate(self._tree, depth=0)
        self.input_indices = tuple(sorted(self._indices))

    def _validate(self, node: ast.AST, depth: int) -> None:
        self._nodes += 1
        if self._nodes > 256 or depth > 48:
            raise ValueError("Expression is too complex (256 nodes / 48 levels).")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("Only real numeric constants are allowed.")
            try:
                value = float(node.value)
            except (OverflowError, ValueError) as exc:
                raise ValueError("Numeric constants must be finite.") from exc
            if not np.isfinite(value):
                raise ValueError("Numeric constants must be finite.")
            return
        if isinstance(node, ast.Name):
            match = _COORDINATE.fullmatch(node.id)
            if match:
                index = int(match.group(1))
                if index >= 4096:
                    raise ValueError("Input coordinate indices must be below 4096.")
                self._indices.add(index)
                return
            if node.id in _CONSTANTS:
                return
            raise ValueError(f"Unknown name {node.id!r}; use x0, x1, ... and math functions.")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            self._validate(node.left, depth + 1)
            self._validate(node.right, depth + 1)
            return
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            self._validate(node.operand, depth + 1)
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            entry = _FUNCTIONS.get(node.func.id)
            if entry is None:
                raise ValueError(f"Function {node.func.id!r} is not supported.")
            if node.keywords or len(node.args) != entry[1]:
                raise ValueError(f"{node.func.id} expects {entry[1]} positional argument(s).")
            for arg in node.args:
                self._validate(arg, depth + 1)
            return
        raise ValueError("Unsupported expression syntax; use arithmetic and named math functions.")

    def _evaluate(self, node: ast.AST, x: np.ndarray) -> np.float64:
        if isinstance(node, ast.Constant):
            return np.float64(node.value)
        if isinstance(node, ast.Name):
            if node.id in _CONSTANTS:
                return np.float64(_CONSTANTS[node.id])
            return np.float64(x[int(node.id[1:])])
        if isinstance(node, ast.BinOp):
            return _BINARY[type(node.op)](self._evaluate(node.left, x), self._evaluate(node.right, x))
        if isinstance(node, ast.UnaryOp):
            return _UNARY[type(node.op)](self._evaluate(node.operand, x))
        function = _FUNCTIONS[node.func.id][0]
        return function(*(self._evaluate(arg, x) for arg in node.args))

    def __call__(self, x: Sequence[float] | np.ndarray) -> float:
        if np.iscomplexobj(x):
            raise ValueError("Expression inputs must be real.")
        values = np.asarray(x, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("Expression inputs must be a finite one-dimensional vector.")
        if self.input_indices and self.input_indices[-1] >= len(values):
            raise ValueError(f"Expression {self.text!r} requires input x{self.input_indices[-1]}.")
        try:
            with np.errstate(all="ignore"):
                result = float(self._evaluate(self._tree, values))
        except (ArithmeticError, OverflowError, ValueError) as exc:
            raise ValueError(f"Expression {self.text!r} is undefined at this input.") from exc
        if not np.isfinite(result):
            raise ValueError(f"Expression {self.text!r} produced a non-finite value.")
        return result

    def __repr__(self) -> str:
        return f"Expression({self.text!r})"
