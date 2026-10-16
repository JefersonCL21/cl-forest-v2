"""Applying the champions to trees, so each value says where it came from: a measured value is kept, only the missing
ones are estimated, each estimate carries the id of the fit that made it and the level of the model (its species,
its genus, or the generic one), and an estimate outside the range the model was fitted on is marked, never clipped.

    out = apply(df, "ht_m", registry)          # df with especie and dap_cm (and ht_m where it was measured)
    out[["ht_m", "ht_m_est", "ht_m_final", "ht_m_source", "ht_m_model_id", "ht_m_out_of_range"]]
    predict_one(registry, "ht_m", "Khaya ivorensis", dap_cm=25)

Columns added for a target t: t_est (the champion's estimate wherever it has its inputs), t_final (the measured value
if valid, else t_est), t_source (measured, species, genus, generic or none), t_model_id and t_out_of_range.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cl_allometry.forms import form as form_by_key
from cl_allometry.registry import Champion, Registry

SOURCES = ("measured", "species", "genus", "generic", "none")
Resolver = Callable[[str | None], Champion | None]


def apply(df: pd.DataFrame, target: str, resolve: Registry | Resolver, *, species_col: str = "especie",
          inputs: Mapping[str, str] | None = None, measured_col: str | None = None,
          measured_ok: pd.Series | None = None) -> pd.DataFrame:
    """A copy of `df` with the target estimated by the champion of each tree's species (or genus, or the generic).

    inputs: the column of each variable of the forms, when not named as the variable ({"ht_m": "ht_m_final"} to use
    the heights of an earlier apply in a volume model). measured_col: the measured target (default: the column
    named as the target, if any); measured_ok: which of those to trust (e.g. not flagged implausible)."""
    lookup = resolve.resolver(target) if isinstance(resolve, Registry) else resolve
    inputs = dict(inputs or {})
    out = df.copy()
    n = len(out)
    est = np.full(n, np.nan)
    model_id = np.full(n, None, dtype=object)
    level = np.full(n, None, dtype=object)
    out_of_range = np.zeros(n, dtype=bool)
    species = out[species_col].tolist() if species_col in out else [None] * n
    names = pd.Series([s.strip() if isinstance(s, str) and s.strip() else "\x00" for s in species])
    for sp, pos in names.groupby(names, sort=False).indices.items():  # positions, whatever the index of df
        champ = lookup(None if sp == "\x00" else sp)
        if champ is None:
            continue
        variables = form_by_key(champ.record.form).variables
        missing = [inputs.get(v, v) for v in variables if inputs.get(v, v) not in out]
        if missing:
            raise ValueError(f"the model of {sp} ({champ.record.form}) needs columns {', '.join(missing)}")
        values = {v: pd.to_numeric(out.iloc[pos][inputs.get(v, v)], errors="coerce").to_numpy(float)
                  for v in variables}
        usable = np.all([np.isfinite(x) & (x > 0) for x in values.values()], axis=0)
        if not usable.any():
            continue
        x = {v: a[usable] for v, a in values.items()}
        rows = pos[usable]
        est[rows] = champ.predict(**x)
        model_id[rows] = champ.fit_id
        level[rows] = champ.scope.level
        out_of_range[rows] = ~champ.in_range(**x)

    measured_col = measured_col or (target if target in out else None)
    measured = (pd.to_numeric(out[measured_col], errors="coerce").to_numpy(float) if measured_col
                else np.full(n, np.nan))
    valid = np.isfinite(measured) & (measured > 0)
    if measured_ok is not None:
        valid &= measured_ok.reindex(out.index, fill_value=False).to_numpy(bool)
    has_est = np.isfinite(est)
    out[f"{target}_est"] = est
    out[f"{target}_final"] = np.where(valid, measured, est)
    out[f"{target}_source"] = np.where(valid, "measured", np.where(has_est, level, "none")).astype(object)
    out[f"{target}_model_id"] = model_id
    out[f"{target}_out_of_range"] = out_of_range & has_est
    return out


@dataclass(frozen=True)
class Estimate:
    value: float
    fit_id: str
    form: str
    scope: str
    level: str
    out_of_range: bool


def predict_one(registry: Registry, target: str, especie: str | None, **values: float) -> Estimate:
    """One tree: the estimate of the champion of its species (or genus, or the generic one)."""
    champ = registry.resolve(target, especie)
    if champ is None:
        raise LookupError(f"no model of {target} for {especie or 'trees without species'}")
    x = {k: np.asarray([float(v)]) for k, v in values.items()}
    return Estimate(value=float(champ.predict(**x)[0]), fit_id=champ.fit_id, form=champ.record.form,
                    scope=str(champ.scope), level=champ.scope.level, out_of_range=not bool(champ.in_range(**x)[0]))
