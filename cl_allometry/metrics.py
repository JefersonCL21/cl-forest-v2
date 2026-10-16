"""The statistics of a fit, defined once, so every screen and report says the same thing.

All of them in the target's own units (a log-linear form is judged on its corrected predictions), with p the number
of fitted parameters:
- bias = mean(observed - predicted): positive, the model underestimates; bias_pct over the observed mean.
- r2 = 1 - SSE/SST (for a nonlinear form, the pseudo-R²) and r2_adj = 1 - (1 - r2)(n - 1)/(n - p).
- rmse = sqrt(SSE/n); syx = sqrt(SSE/(n - p)), the standard error of the estimate; *_pct over the observed mean.
- aic, bic: Gaussian, counting the residual variance as a parameter.
- furnival: syx for a form fitted on the target itself; for one fitted on log(target), the syx in logs times the
  geometric mean of the target (Furnival 1961), so the two can be compared.
Residual checks: Shapiro-Wilk (normality), Breusch-Pagan on the fitted values (constant variance), and the largest
Cook's distance (an observation that alone moves the fit).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def summary(y: np.ndarray, pred: np.ndarray, p: int, log_syx: float | None = None) -> dict[str, float]:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    n = len(y)
    resid = y - pred
    sse, sst, mean = float(np.sum(resid**2)), float(np.sum((y - y.mean()) ** 2)), float(y.mean())
    dof = max(n - p, 1)
    r2 = 1 - sse / sst if sst > 0 else float("nan")
    syx = float(np.sqrt(sse / dof))
    loglik_term = n * np.log(sse / n) if sse > 0 else -np.inf
    return {
        "n": n,
        "p": p,
        "r2": r2,
        "r2_adj": 1 - (1 - r2) * (n - 1) / dof if n > p else float("nan"),
        "rmse": float(np.sqrt(sse / n)),
        "rmse_pct": 100 * float(np.sqrt(sse / n)) / mean,
        "syx": syx,
        "syx_pct": 100 * syx / mean,
        "bias": float(resid.mean()),
        "bias_pct": 100 * float(resid.mean()) / mean,
        "mae": float(np.abs(resid).mean()),
        "aic": float(loglik_term + 2 * (p + 1)),
        "bic": float(loglik_term + (p + 1) * np.log(n)),
        "furnival": float(np.exp(np.mean(np.log(y))) * log_syx) if log_syx is not None else syx,
    }


def residual_checks(resid: np.ndarray, fitted: np.ndarray, leverage: np.ndarray | None, p: int) -> dict[str, float]:
    resid, fitted = np.asarray(resid, float), np.asarray(fitted, float)
    n = len(resid)
    out: dict[str, float] = {}
    sample = resid if n <= 5000 else np.random.default_rng(0).choice(resid, 5000, replace=False)
    out["shapiro_p"] = float(stats.shapiro(sample).pvalue) if n >= 3 else float("nan")
    # Breusch-Pagan: regress the squared residuals on the fitted values; LM = n R², chi² with 1 degree of freedom
    e2 = resid**2
    design = np.column_stack([np.ones(n), fitted])
    coef = np.linalg.lstsq(design, e2, rcond=None)[0]
    ss_tot = float(np.sum((e2 - e2.mean()) ** 2))
    r2 = 1 - float(np.sum((e2 - design @ coef) ** 2)) / ss_tot if ss_tot > 0 else 0.0
    out["breusch_pagan_p"] = float(stats.chi2.sf(n * r2, 1))
    if leverage is not None and n > p:
        s2 = float(np.sum(resid**2)) / (n - p)
        h = np.clip(leverage, 0, 1 - 1e-12)
        cook = resid**2 / (p * s2) * h / (1 - h) ** 2 if s2 > 0 else np.zeros(n)
        out["cook_max"] = float(np.max(cook))
        out["cook_over_4_n"] = int(np.sum(cook > 4 / n))  # the usual rule of thumb for an influential point
    return out
