"""A model in a form any other system can use without Python: a model card (JSON with the formula, coefficients,
units, validity range, statistics and the data it came from), a SQL expression that applies it inside a database
(BigQuery, Postgres), and a table with one row per model (the shape of a future BigQuery mirror of the register).

The SQL is generated from the same formula text the fit used (cl_allometry.expr), walking its syntax tree, so it
cannot drift from the Python model: every operation is parenthesized, powers become POWER(), the coefficients are
written with all their digits, and the columns are cast to floating point (no integer division in Postgres).

    to_sql("height.logistic", {"b0": 21.4, "b1": 6.87, "b2": 0.157}, dialect="bigquery")
    model_card(record, champion)
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping, Sequence

import pandas as pd

from cl_allometry.forms import TARGETS, Form, form as form_by_key
from cl_allometry.registry import Champion, FitRecord

DIALECTS = ("bigquery", "postgres", "sqlite")
_OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
_FUNCS = {"exp": "EXP", "log": "LN", "sqrt": "SQRT"}
_FLOAT = {"bigquery": "FLOAT64", "postgres": "DOUBLE PRECISION", "sqlite": "REAL"}


def _quote(column: str, dialect: str) -> str:
    return f"`{column}`" if dialect == "bigquery" else '"' + column.replace('"', '""') + '"'


def _number(v: float, dialect: str) -> str:
    text = repr(float(v))
    if dialect == "postgres":  # a bare 6e-05 is numeric in Postgres; the model is in double precision
        text = f"CAST({text} AS DOUBLE PRECISION)"
    return f"({text})" if text.startswith("-") else text


def to_sql(form: Form | str, coefficients: Mapping[str, float], *, correction: float = 1.0,
           dialect: str = "bigquery", columns: Mapping[str, str] | None = None, safe: bool = False) -> str:
    """The model as a SQL expression over the table's columns (by default named as the variables: dap_cm, ht_m...).
    safe (BigQuery): SAFE. functions and SAFE_DIVIDE, which give NULL where the model is undefined instead of an
    error, as numpy gives nan."""
    if dialect not in DIALECTS:
        raise ValueError(f"dialect must be one of {', '.join(DIALECTS)}")
    f = form_by_key(form) if isinstance(form, str) else form
    columns = dict(columns or {})
    prefix = "SAFE." if safe and dialect == "bigquery" else ""

    def walk(node: ast.AST) -> str:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.BinOp):
            a, b = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow):
                return f"{prefix}POWER({a}, {b})"
            if isinstance(node.op, ast.Div) and prefix:
                return f"SAFE_DIVIDE({a}, {b})"
            return f"({a} {_OPS[type(node.op)]} {b})"
        if isinstance(node, ast.UnaryOp):
            inner = walk(node.operand)
            return f"(-{inner})" if isinstance(node.op, ast.USub) else inner
        if isinstance(node, ast.Call):
            return f"{prefix}{_FUNCS[node.func.id]}({walk(node.args[0])})"
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return "ACOS(-1)"
            if node.id in f.params:
                return _number(coefficients[node.id], dialect)
            return f"CAST({_quote(columns.get(node.id, node.id), dialect)} AS {_FLOAT[dialect]})"
        if isinstance(node, ast.Constant):
            return _number(node.value, dialect)
        raise ValueError(f"cannot write {type(node).__name__} as SQL")

    f.formula()  # the text is checked to be only arithmetic of its names, as when it is fitted
    sql = walk(ast.parse(f.text, mode="eval"))
    if correction != 1.0:
        sql = f"({sql} * {_number(correction, dialect)})"
    return sql


def model_card(record: FitRecord, champion: Champion | None = None) -> dict:
    """Everything another system needs to use the model and to know where it came from. JSON-ready."""
    f = form_by_key(record.form)
    fit = record.fit
    sql = ({d: to_sql(f, fit.coefficients, correction=fit.correction, dialect=d) for d in ("bigquery", "postgres")}
           if fit.converged else None)
    return {
        "schema": "cl_allometry.model_card/1",
        "fit_id": record.fit_id,
        "target": {"name": record.target, "label": TARGETS.get(record.target, record.target)},
        "form": {"key": f.key, "name": f.name, "formula": f.text, "fitted_as": f.fit_text or f.text,
                 "log_linear": f.log_linear, "variables": list(f.variables), "params": list(f.params),
                 "reference": f.reference},
        "prediction": f"({f.text}) * correction, with the coefficients below and the variables in their units",
        "coefficients": fit.coefficients, "correction": fit.correction,
        "stderr": fit.stderr, "ci95": fit.ci95, "p_value": fit.p_value,
        "n": fit.n, "dropped": fit.dropped, "ranges": fit.ranges,
        "metrics": fit.metrics, "cross_validation": record.cv, "residual_checks": fit.checks,
        "fit_options": {"weights": fit.weights, "seed": fit.seed, "starts": fit.starts},
        "dataset": {"id": record.dataset_id, "sha256": record.dataset_sha256},
        "filter": record.filter,
        "fitted_by": record.author, "commit": record.commit, "created_at": record.created_at,
        "champion": None if champion is None else {
            "scope": str(champion.scope), "event": champion.event.seq, "reason": champion.event.reason,
            "promoted_by": champion.event.author, "promoted_at": champion.event.created_at,
            "valid_range": champion.event.valid_range, "warnings": champion.event.warnings},
        "sql": sql,
    }


def models_table(records: Sequence[FitRecord], champions: Sequence[Champion] = ()) -> pd.DataFrame:
    """One row per fit, flat enough for a spreadsheet or a database table; the champion columns are filled for the
    fits in use."""
    in_use: dict[str, list[Champion]] = {}
    for c in champions:
        in_use.setdefault(c.fit_id, []).append(c)
    rows = []
    for r in records:
        f, fit = form_by_key(r.form), r.fit
        cs = in_use.get(r.fit_id, [])
        rows.append({
            "fit_id": r.fit_id, "target": r.target, "form": r.form, "form_name": f.name, "formula": f.text,
            "coefficients": json.dumps(fit.coefficients, sort_keys=True) if fit.coefficients else None,
            "correction": fit.correction, "converged": fit.converged, "n": fit.n,
            "rmse_pct": fit.metrics.get("rmse_pct"), "bias_pct": fit.metrics.get("bias_pct"),
            "r2_adj": fit.metrics.get("r2_adj"), "aic": fit.metrics.get("aic"),
            "cv_rmse_pct": (r.cv or {}).get("rmse_pct"), "cv_bias_pct": (r.cv or {}).get("bias_pct"),
            "ranges": json.dumps(fit.ranges, sort_keys=True), "dataset_id": r.dataset_id,
            "dataset_sha256": r.dataset_sha256, "filter": json.dumps(r.filter, sort_keys=True, ensure_ascii=False),
            "fitted_by": r.author, "commit": r.commit, "created_at": r.created_at,
            "champion_of": ", ".join(str(c.scope) for c in cs) or None,
            "sql_bigquery": to_sql(f, fit.coefficients, correction=fit.correction) if fit.converged else None,
        })
    return pd.DataFrame(rows)
