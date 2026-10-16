"""A model's formula as text, the single place it is written: the same text is fitted (cl_allometry.fit), applied
(Fit.predict) and, later, turned into SQL for other databases. Only arithmetic, powers, exp, log and sqrt of the
form's variables and parameters are allowed, checked when the text is read, so evaluating it runs nothing else.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

import numpy as np

FUNCTIONS = {"exp": np.exp, "log": np.log, "sqrt": np.sqrt}
CONSTANTS = {"pi": np.pi}
_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
          ast.Call, ast.Name, ast.Load, ast.Constant)


class Formula:
    """`text` over `names` (the form's variables and parameters)."""

    def __init__(self, text: str, names: Iterable[str]) -> None:
        self.text = text
        self.names = tuple(names)
        tree = ast.parse(text, mode="eval")
        allowed = set(self.names) | set(CONSTANTS) | set(FUNCTIONS)
        for node in ast.walk(tree):
            if not isinstance(node, _NODES):
                raise ValueError(f"not allowed in a formula: {type(node).__name__} in {text!r}")
            if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS):
                raise ValueError(f"only {', '.join(FUNCTIONS)} may be called: {text!r}")
            if isinstance(node, ast.Name) and node.id not in allowed:
                raise ValueError(f"unknown name {node.id!r} in {text!r}")
            if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
                raise ValueError(f"only numbers are constants: {text!r}")
        self._code = compile(tree, f"<formula {text}>", "eval")

    def __call__(self, **values) -> np.ndarray:
        missing = [n for n in self.names if n not in values]
        if missing:
            raise ValueError(f"{self.text!r} needs {', '.join(missing)}")
        env = {**FUNCTIONS, **CONSTANTS, **{k: np.asarray(v, dtype=float) for k, v in values.items()}}
        with np.errstate(all="ignore"):  # out of a form's domain gives nan/inf, which the fit and the checks catch
            return np.asarray(eval(self._code, {"__builtins__": {}}, env), dtype=float)  # noqa: S307 — checked above

    def __repr__(self) -> str:
        return f"Formula({self.text!r})"
