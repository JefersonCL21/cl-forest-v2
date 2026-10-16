"""The height models of cl-forest-v2 before cl_allometry, migrated as they were: the coefficients of the page's
functions ("code", pages/3_📈Modelos_alometricos.py) and of its export table ("table"), which disagree (for Khaya
grandifoliola every form differs, and every parabolic b2 of the table is wrong). They come from
data/legacy_height_models.json, read out of the page's source at commit 680c086, so nothing was retyped.

Which set is right is a decision to record, not a guess: compare them on the measured heights with compare(). The
decision (16/10/2026) was to refit: bootstrap_v1() puts the refitted models in the register as version 1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from cl_allometry import datasets
from cl_allometry import metrics as M
from cl_allometry.fit import Fit
from cl_allometry.forms import form, forms_for
from cl_allometry.registry import Filter, Registry, Scope
from cl_allometry.store import Store

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


def _max_rel(a: dict[str, float], b: dict[str, float]) -> float:
    return max(abs(a[k] - b[k]) / abs(b[k]) for k in b)


@dataclass
class BootstrapReport:
    dataset_id: str
    sha256: str
    created_version: bool
    species: list[dict] = field(default_factory=list)  # one row per species: n, fits, the checks, the promotion
    promoted: bool = False

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.species)


def bootstrap_v1(store: Store, csv_path: str | Path, *, author: str, decided_by: str, decided_on: str,
                 only: list[str] | None = None, cv_k: int = 5, promote: bool = True, tolerance: float = 1e-3
                 ) -> BootstrapReport:
    """Version 1 of the height models in the register, as decided on 16/10/2026: dados/GERAL_V2.csv imported as
    inventario@v1, every height form refitted on the measured heights of each of the 9 species, and the logistic
    (the form that made HT_Est) promoted for each species.

    Nothing is promoted unless every refitted logistic is within `tolerance` (relative) of the coefficients of the
    table that made HT_Est: the check that version 1 changes no estimate in use. Running it again adds nothing."""
    path = Path(csv_path)
    source = path.read_bytes()
    raw = datasets.read_upload(path.name, source)
    # the version that already holds these values, if any (later versions may exist: importing again would add one)
    sha = datasets.prepare(raw, datasets.INVENTORY_MAPPING)[1]
    held = [v for v in datasets.versions(store, "inventario") if v.sha256 == sha]
    if held:
        dataset_id, created = held[0].dataset_id, False
    else:
        imported = datasets.import_dataset(store, raw, family="inventario", mapping=datasets.INVENTORY_MAPPING,
                                           author=author, source_name=path.name, source_bytes=source,
                                           note="Versão 1: o censo do app (dados/GERAL_V2.csv)")
        dataset_id, created = imported.manifest.dataset_id, imported.created_version
    data, m = datasets.load(store, dataset_id)
    reg = Registry(store)
    report = BootstrapReport(dataset_id=m.dataset_id, sha256=m.sha256, created_version=created)
    keys = [f.key for f in forms_for("ht_m")]
    logistic = {}
    for sp in only or species():
        # every row of the species, nothing excluded: the fit that made the table saw them all
        flt = Filter(especie=(sp,), exclude_flags=())
        recs = {k: reg.run_fit(data, m, k, flt, author=author, seed=0, cv_k=cv_k) for k in keys}
        lg = recs["height.logistic"]
        logistic[sp] = lg
        curtis = recs["height.curtis_log"].fit.coefficients or {}
        report.species.append({
            "species": sp, "n": lg.fit.n,
            "logistic_vs_table": _max_rel(lg.fit.coefficients, model("logistic", sp, "table").coefficients),
            "curtis_log_vs_legacy": min(_max_rel(curtis, model("curtis", sp, s).coefficients)
                                        for s in ("code", "table")) if curtis else float("nan"),
            "logistic_fit_id": lg.fit_id, "fits": len(recs),
            "converged": sum(r.fit.converged for r in recs.values()),
            "rmse_pct": lg.fit.metrics.get("rmse_pct"), "cv_rmse_pct": (lg.cv or {}).get("rmse_pct"),
        })
    worst = max(r["logistic_vs_table"] for r in report.species)
    if not promote or worst > tolerance:
        return report

    for row in report.species:
        sp, lg = row["species"], logistic[row["species"]]
        scope = Scope.species(sp)
        current = reg.champion("ht_m", scope)
        if current is not None and current.fit_id == lg.fit_id:
            row["champion_seq"] = current.event.seq
            continue
        warnings = []
        if lg.fit.n < 20:
            warnings.append(f"só {lg.fit.n} alturas medidas")
        weak = [k for k, p in lg.fit.p_value.items() if p is None or p > 0.05]
        if weak:
            warnings.append(f"parâmetros não significativos a 5%: {', '.join(weak)}")
        reason = (f"Versão 1: logística reajustada pelo cl_allometry nas {lg.fit.n} alturas medidas de {sp} em "
                  f"{m.dataset_id} (sha256 {m.sha256[:12]}). Reproduz os coeficientes que geraram o HT_Est "
                  f"(diferença relativa máxima {row['logistic_vs_table']:.1e}). Decisão de {decided_by} em "
                  f"{decided_on}.")
        e = reg.promote(lg.fit_id, scope, author=author, reason=reason, warnings=warnings,
                        expected_seq=current.event.seq if current else 0)
        row["champion_seq"] = e.seq
    report.promoted = True
    return report


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
