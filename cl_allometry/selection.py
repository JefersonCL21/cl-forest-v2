"""Choosing among fitted candidates with a criterion, as is usual in forest mensuration: each statistic ranks the
candidates (1 = best, in that statistic's own direction), and the ranks are added with weights. The candidates must
be comparable: the same target, fitted on the same rows of the same data.

The ranking recommends and says why; the decision is a person's, who promotes it in the register with a reason
(cl_allometry.registry). Near ties go to the simpler model, then to the champion already in use: a change of model
should be worth it.

    ranked = rank(records)                      # records: cl_allometry.registry.FitRecord
    rec = recommend(ranked, incumbent=champion_fit_id)
    rec.fit_id, rec.reasons, rec.warnings
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cl_allometry.forms import form as form_by_key
from cl_allometry.registry import FitRecord

# name: (where it is in a record, the better direction, label on screens)
METRICS: dict[str, tuple[str, str, str]] = {
    "cv_rmse_pct": ("cv.rmse_pct", "min", "RMSE% (validação cruzada)"),
    "cv_abs_bias_pct": ("cv.bias_pct", "min_abs", "|Viés%| (validação cruzada)"),
    "rmse_pct": ("metrics.rmse_pct", "min", "RMSE%"),
    "syx_pct": ("metrics.syx_pct", "min", "Syx%"),
    "abs_bias_pct": ("metrics.bias_pct", "min_abs", "|Viés%|"),
    "r2_adj": ("metrics.r2_adj", "max", "R² ajustado"),
    "aic": ("metrics.aic", "min", "AIC"),
    "bic": ("metrics.bic", "min", "BIC"),
    "furnival": ("metrics.furnival", "min", "Índice de Furnival"),
}
# out-of-sample error counts twice: it is the one that says how the model does on trees it did not see
DEFAULT_WEIGHTS = {"cv_rmse_pct": 2, "cv_abs_bias_pct": 1, "r2_adj": 1, "aic": 1}


class MixedCandidates(ValueError):
    """Candidates fitted to different targets, data or rows cannot be ranked together."""


def _value(rec: FitRecord, where: str) -> float:
    source, key = where.split(".")
    d = (rec.cv or {}) if source == "cv" else rec.fit.metrics
    v = d.get(key)
    return float(v) if v is not None else np.nan


def table(records: Sequence[FitRecord]) -> pd.DataFrame:
    """One row per candidate: its statistics, and the warnings a person should see before choosing it."""
    rows = []
    for r in records:
        f = form_by_key(r.form)
        p_values = [v for v in r.fit.p_value.values()]
        rows.append({
            "fit_id": r.fit_id, "form": r.form, "name": f.name, "converged": r.fit.converged, "n": r.fit.n,
            "params": len(f.params),
            **{m: (abs(_value(r, w)) if better == "min_abs" else _value(r, w)) for m, (w, better, _) in METRICS.items()},
            "non_significant": sum(1 for p in p_values if p is None or p > 0.05) if r.fit.converged else np.nan,
            "breusch_pagan_p": r.fit.checks.get("breusch_pagan_p", np.nan),
            "shapiro_p": r.fit.checks.get("shapiro_p", np.nan),
            "influential": r.fit.checks.get("cook_over_4_n", np.nan),
            "cv_converged": bool(r.cv and r.cv.get("converged")),
            "message": r.fit.message,
        })
    return pd.DataFrame(rows)


def _check_comparable(records: Sequence[FitRecord]) -> None:
    for what, key in (("target", lambda r: r.target), ("data", lambda r: r.dataset_sha256),
                      ("rows", lambda r: str(sorted(r.filter.items())))):
        if len({key(r) for r in records}) > 1:
            raise MixedCandidates(f"the candidates differ in {what}: rank fits of the same rows of the same data")


def rank(records: Sequence[FitRecord], weights: Mapping[str, float] | None = None) -> pd.DataFrame:
    """The table of candidates with a rank per weighted statistic, the weighted sum (score) and the final position
    (1 = best). A fit that did not converge is listed without a position."""
    if not records:
        return table([])
    _check_comparable(records)
    weights = {k: v for k, v in (weights or DEFAULT_WEIGHTS).items() if v}
    unknown = set(weights) - set(METRICS)
    if unknown:
        raise ValueError(f"unknown statistics: {', '.join(sorted(unknown))}")
    t = table(records)
    ok = t["converged"]
    score = pd.Series(0.0, index=t.index)
    for m, w in weights.items():
        better = METRICS[m][1]
        values = t.loc[ok, m]
        r = values.rank(method="min", ascending=(better != "max"), na_option="bottom")
        t.loc[ok, f"rank_{m}"] = r
        score.loc[ok] += w * r
    t["score"] = score.where(ok)
    t["position"] = t["score"].rank(method="min")
    return t.sort_values(["position", "params"], na_position="last").reset_index(drop=True)


@dataclass
class Recommendation:
    fit_id: str | None
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    runner_up: str | None = None
    tied: list[str] = field(default_factory=list)


def warnings_for(row: Mapping) -> list[str]:
    """What a person should know about a candidate before promoting it."""
    out = []
    if row["n"] < 20:
        out.append(f"só {int(row['n'])} árvores no ajuste")
    if row.get("non_significant", 0) and row["non_significant"] > 0:
        out.append(f"{int(row['non_significant'])} parâmetro(s) não significativo(s) a 5%")
    if row.get("breusch_pagan_p", 1) < 0.05:
        out.append(f"variância dos resíduos não constante (Breusch-Pagan p = {row['breusch_pagan_p']:.3f}); "
                   "considere ajustar com pesos")
    if row.get("shapiro_p", 1) < 0.05:
        out.append(f"resíduos não normais (Shapiro-Wilk p = {row['shapiro_p']:.3f})")
    if row.get("influential", 0) and row["influential"] > 0:
        out.append(f"{int(row['influential'])} árvore(s) influente(s) (Cook > 4/n)")
    if not row.get("cv_converged", True):
        out.append("a validação cruzada não convergiu")
    return out


def recommend(ranked: pd.DataFrame, tolerance_pct: float = 0.5, incumbent: str | None = None) -> Recommendation:
    """The candidate to promote, with the reasons. Candidates within `tolerance_pct` points of the best RMSE% in
    cross-validation are a tie: among them, the one with fewer parameters, then the champion in use (incumbent),
    then the best score."""
    ok = ranked[ranked["converged"]]
    if ok.empty:
        return Recommendation(None, reasons=["nenhum candidato convergiu"])
    best = ok.iloc[0]
    metric = "cv_rmse_pct" if ok["cv_rmse_pct"].notna().all() else "rmse_pct"
    tied = ok[(ok[metric] - best[metric]).abs() <= tolerance_pct]
    order = tied.assign(is_incumbent=tied["fit_id"] == incumbent).sort_values(
        ["params", "is_incumbent", "score"], ascending=[True, False, True])
    choice = order.iloc[0]
    label = METRICS[metric][2]
    reasons = [f"{best['name']} tem a melhor pontuação ({best['score']:.0f}) com os pesos usados"]
    if choice["fit_id"] != best["fit_id"]:
        why = ("tem menos parâmetros" if choice["params"] < best["params"] else "é o modelo em uso")
        reasons = [f"{choice['name']} empata com {best['name']} (diferença de "
                   f"{abs(choice[metric] - best[metric]):.2f} ponto(s) de {label}) e {why}"]
    elif len(tied) > 1:
        reasons.append(f"empata com {', '.join(tied[tied['fit_id'] != best['fit_id']]['name'])} em {label}, "
                       "mas não há um mais simples")
    reasons.append(f"{label}: {choice[metric]:.2f}%")
    rest = ok[ok["fit_id"] != choice["fit_id"]]
    return Recommendation(fit_id=choice["fit_id"], reasons=reasons, warnings=warnings_for(choice),
                          runner_up=rest.iloc[0]["fit_id"] if len(rest) else None,
                          tied=[x for x in tied["fit_id"] if x != choice["fit_id"]])
