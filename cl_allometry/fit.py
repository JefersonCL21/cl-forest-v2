"""Fitting a form to data: nonlinear least squares from several starting points (scipy least_squares), optionally
weighted, with the parameters' standard errors, 95% intervals and t tests, the statistics of cl_allometry.metrics,
and k-fold cross-validation. A log-linear form is fitted in logs and its predictions carry the correction for the
bias of going back (exp(s²/2), Baskerville 1972 / Sprugel 1983).

A fit that does not converge comes back with converged False and no coefficients, never with stale ones. Rows with a
missing or non-positive value in the form's variables or target are left out, and counted (dropped).

    from cl_allometry import fit
    f = fit("height.curtis", df, seed=0)        # df has dap_cm and ht_m
    f.coefficients, f.metrics["rmse_pct"], f.predict(dap_cm=25)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares

from cl_allometry import metrics as M
from cl_allometry.expr import Formula
from cl_allometry.forms import Form, form as form_by_key

# Evaluations allowed from each starting point. On the 9 species × 9 height forms of the inventory, the fits that
# converge took at most 1,075; a start that has not converged by 2,000 is not going to, and letting it run to 20,000
# made one hopeless Richards on 16 trees take 80 s.
MAX_NFEV = 2000


@dataclass
class Fit:
    form: str
    target: str
    converged: bool
    message: str
    n: int
    dropped: int
    coefficients: dict[str, float] | None = None
    correction: float = 1.0  # log-linear forms: exp(s²/2) on the predictions; 1 otherwise
    stderr: dict[str, float] = field(default_factory=dict)
    ci95: dict[str, list[float]] = field(default_factory=dict)
    t: dict[str, float] = field(default_factory=dict)
    p_value: dict[str, float] = field(default_factory=dict)
    covariance: list[list[float]] | None = None
    ranges: dict[str, list[float]] = field(default_factory=dict)  # min and max of each variable and the target used
    metrics: dict[str, float] = field(default_factory=dict)
    checks: dict[str, float] = field(default_factory=dict)
    weights: str | None = None
    seed: int = 0
    starts: int = 1

    def predict(self, **values) -> np.ndarray:
        if not self.coefficients:
            raise ValueError(f"{self.form} has no coefficients: {self.message}")
        return form_by_key(self.form).formula()(**values, **self.coefficients) * self.correction

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Fit:
        return cls(**d)


def _prepare(f: Form, data: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    missing = [c for c in (*f.variables, f.target) if c not in data.columns]
    if missing:
        raise ValueError(f"{f.key} needs columns {', '.join(missing)}")
    cols = data[[*f.variables, f.target]].apply(pd.to_numeric, errors="coerce")
    ok = np.all(np.isfinite(cols.to_numpy(float)) & (cols.to_numpy(float) > 0), axis=1)
    x = {v: cols[v].to_numpy(float)[ok] for v in f.variables}
    return x, cols[f.target].to_numpy(float)[ok], int((~ok).sum())


def fit(form: Form | str, data: pd.DataFrame, weights: str | None = None, starts: int = 8, seed: int = 0,
        start: dict[str, float] | None = None) -> Fit:
    """Fits `form` to `data`. weights: a formula over the form's variables (e.g. "1/dap_cm**2") for residuals whose
    spread grows with the tree; starts: how many starting points (the form's own, or `start` if given, then random
    around it)."""
    f = form_by_key(form) if isinstance(form, str) else form
    x, y, dropped = _prepare(f, data)
    n, p = len(y), len(f.params)
    base = {"form": f.key, "target": f.target, "n": n, "dropped": dropped, "weights": weights, "seed": seed,
            "starts": starts}
    if n <= p:
        return Fit(converged=False, message=f"poucos dados: {n} árvores para {p} parâmetros", **base)

    model = f.fit_formula()
    target = np.log(y) if f.log_linear else y
    w = np.sqrt(Formula(weights, f.variables)(**x)) if weights else np.ones(n)
    if not np.all(np.isfinite(w)):
        return Fit(converged=False, message=f"pesos inválidos: {weights}", **base)

    def residuals(b: np.ndarray) -> np.ndarray:
        r = (model(**x, **dict(zip(f.params, b))) - target) * w
        return np.where(np.isfinite(r), r, 1e6)  # outside the form's domain: a large residual, not a crash

    first = start or f.start(x, y)
    x0 = np.array([first.get(k, 1.0) for k in f.params], dtype=float)
    rng = np.random.default_rng(seed)
    candidates = [x0] + [x0 * np.exp(rng.normal(0, 0.6, p)) for _ in range(max(starts, 1) - 1)]
    best = None
    for c in candidates:
        try:
            r = least_squares(residuals, c, method="trf", x_scale="jac", max_nfev=MAX_NFEV)
        except (ValueError, FloatingPointError):
            continue
        if r.success and np.isfinite(r.cost) and (best is None or r.cost < best.cost):
            best = r
    if best is None or np.any(np.abs(best.fun) >= 1e6):
        return Fit(converged=False, message="o ajuste não convergiu de nenhum ponto de partida", **base)

    b = best.x
    coef = dict(zip(f.params, map(float, b)))
    dof = n - p
    s2 = float(np.sum(best.fun**2)) / dof
    jtj_inv = np.linalg.pinv(best.jac.T @ best.jac)
    cov = jtj_inv * s2
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    tval = np.divide(b, se, out=np.full(p, np.inf), where=se > 0)
    q = stats.t.ppf(0.975, dof)
    leverage = np.einsum("ij,jk,ik->i", best.jac, jtj_inv, best.jac)

    fitted_scale = model(**x, **coef)
    if f.log_linear:
        log_resid = target - fitted_scale
        s2_log = float(np.sum(log_resid**2)) / dof
        correction = float(np.exp(s2_log / 2))
        pred = np.exp(fitted_scale) * correction
        summary = M.summary(y, pred, p, log_syx=float(np.sqrt(s2_log)))
    else:
        correction, pred = 1.0, fitted_scale
        summary = M.summary(y, pred, p)
    return Fit(
        converged=True, message="ok", coefficients=coef, correction=correction,
        stderr=dict(zip(f.params, map(float, se))),
        ci95={k: [float(v - q * s), float(v + q * s)] for k, v, s in zip(f.params, b, se)},
        t=dict(zip(f.params, map(float, tval))),
        p_value={k: float(2 * stats.t.sf(abs(t), dof)) for k, t in zip(f.params, tval)},
        covariance=cov.tolist(),
        ranges={k: [float(v.min()), float(v.max())] for k, v in {**x, f.target: y}.items()},
        metrics=summary, checks=M.residual_checks(y - pred, pred, leverage, p), **base,
    )


def cross_validate(form: Form | str, data: pd.DataFrame, k: int = 5, seed: int = 0, weights: str | None = None,
                   start: dict[str, float] | None = None) -> dict:
    """Out-of-fold statistics: each tree predicted by a fit that did not see it (k folds, shuffled with `seed`).

    Each fold's fit starts from `start` (by default, the coefficients of the fit on all the data): its result is
    still the minimum on the fold's own trees, found in a few steps instead of from every starting point again,
    which made a cross-validation take minutes. Only if that fails are the usual starting points tried."""
    f = form_by_key(form) if isinstance(form, str) else form
    x, y, _ = _prepare(f, data)
    n = len(y)
    order = np.random.default_rng(seed).permutation(n)
    pred = np.full(n, np.nan)
    frame = pd.DataFrame({**x, f.target: y})
    if start is None:
        whole = fit(f, frame, weights=weights, seed=seed)
        start = whole.coefficients
    for fold in np.array_split(order, k):
        train = frame.drop(index=fold)
        fitted = fit(f, train, weights=weights, seed=seed, starts=1, start=start) if start else None
        if fitted is None or not fitted.converged:
            fitted = fit(f, train, weights=weights, seed=seed)
        if not fitted.converged:
            return {"k": k, "seed": seed, "converged": False, "message": fitted.message}
        pred[fold] = fitted.predict(**{v: x[v][fold] for v in f.variables})
    return {"k": k, "seed": seed, "converged": True, **M.summary(y, pred, len(f.params))}
