"""The height models of cl-forest-v2 before cl_allometry, migrated as they were: the coefficients of the page's
functions ("code", pages/3_📈Modelos_alometricos.py) and of its export table ("table"), which disagree (for Khaya
grandifoliola every form differs, and every parabolic b2 of the table is wrong). They come from
data/legacy_height_models.json, read out of the page's source at commit 680c086, so nothing was retyped.

Which set is right is a decision to record, not a guess: compare them on the measured heights with compare().
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from cl_allometry import metrics as M
from cl_allometry.fit import Fit
from cl_allometry.forms import form

DATA = Path(__file__).parent / "data" / "legacy_height_models.json"
FORMS = {"logistic": "height.logistic", "gompertz": "height.gompertz", "parabolic": "height.parabolic",
         "curtis": "height.curtis"}


@cache
def _load() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def species() -> list[str]:
    return sorted(_load()["code"]["logistic"])


def model(name: str, sp: str, source: str = "code") -> Fit:
    """A migrated model as a Fit without statistics (they were never kept), ready to predict and to compare."""
    entry = _load()[source][name][sp]
    return Fit(form=FORMS[name], target="ht_m", converged=True, n=0, dropped=0, coefficients=entry["coefficients"],
               message=f"migrado de cl-forest-v2 ({source}, linha {entry['line']} da página, commit 680c086)")


def compare(df: pd.DataFrame, sp: str) -> pd.DataFrame:
    """Every migrated model of `sp`, both sets, judged on the trees of `df` with a measured height (dap_cm, ht_m)."""
    rows = []
    d = df[(df["dap_cm"] > 0) & (df["ht_m"] > 0)]
    for name, key in FORMS.items():
        for source in ("code", "table"):
            pred = model(name, sp, source).predict(dap_cm=d["dap_cm"].to_numpy(float))
            stats = M.summary(d["ht_m"].to_numpy(float), pred, len(form(key).params)) if len(d) else {}
            rows.append({"species": sp, "form": key, "source": source, **{k: stats.get(k, np.nan)
                         for k in ("n", "rmse", "rmse_pct", "bias", "r2")}})
    return pd.DataFrame(rows)
