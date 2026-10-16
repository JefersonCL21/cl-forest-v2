"""The catalogue of allometric forms: for each, its formula as text (cl_allometry.expr), its variables with units in
the name (dap_cm, ht_m, crown_m2, rho_gcm3), its parameters, what it estimates, and where a fit starts from.

A log-linear form is fitted as log(target) = fit_text by least squares and predicts exp(fit_text) times the
correction for the bias of going back from logs (cl_allometry.fit); its `text` is that prediction, without the
correction, which is a fitted constant.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from cl_allometry.expr import Formula

Start = Callable[[dict[str, np.ndarray], np.ndarray], dict[str, float]]

TARGETS = {
    "ht_m": "altura total (m)",
    "vol_m3": "volume do fuste (m³)",
    "agb_kg": "biomassa acima do solo (kg)",
    "dap_cm": "DAP (cm)",
}


@dataclass(frozen=True)
class Form:
    key: str
    target: str
    name: str
    text: str
    variables: tuple[str, ...]
    params: tuple[str, ...]
    start: Start
    fit_text: str | None = None
    reference: str = ""

    @property
    def log_linear(self) -> bool:
        return self.fit_text is not None

    def formula(self) -> Formula:
        return Formula(self.text, self.variables + self.params)

    def fit_formula(self) -> Formula:
        return Formula(self.fit_text or self.text, self.variables + self.params)


def _ols(columns: list[np.ndarray], y: np.ndarray, names: tuple[str, ...]) -> dict[str, float]:
    """Least squares for a form linear in its parameters: exact starting values (often the answer itself)."""
    design = np.column_stack(columns)
    ok = np.all(np.isfinite(design), axis=1) & np.isfinite(y)
    coef = np.linalg.lstsq(design[ok], y[ok], rcond=None)[0] if ok.sum() >= len(names) else np.ones(len(names))
    return dict(zip(names, map(float, coef)))


def _one(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x, dtype=float)


def _asymptote(y: np.ndarray) -> float:
    return float(np.nanmax(y)) * 1.1 if np.isfinite(np.nanmax(y)) else 20.0


_FORMS = [
    # height from DBH: dap_cm -> ht_m
    Form("height.curtis", "ht_m", "Curtis", "exp(b0 + b1/dap_cm)", ("dap_cm",), ("b0", "b1"),
         lambda x, y: _ols([_one(x["dap_cm"]), 1 / x["dap_cm"]], np.log(y), ("b0", "b1")), reference="Curtis (1967)"),
    Form("height.logistic", "ht_m", "Logístico", "b0/(1 + b1*exp(-b2*dap_cm))", ("dap_cm",), ("b0", "b1", "b2"),
         lambda x, y: {"b0": _asymptote(y), "b1": 5.0, "b2": 0.1}),
    Form("height.gompertz", "ht_m", "Gompertz", "b0*exp(-exp(b1 - b2*dap_cm))", ("dap_cm",), ("b0", "b1", "b2"),
         lambda x, y: {"b0": _asymptote(y), "b1": 1.0, "b2": 0.1}),
    Form("height.parabolic", "ht_m", "Parabólico", "b0 + b1*dap_cm + b2*dap_cm**2", ("dap_cm",), ("b0", "b1", "b2"),
         lambda x, y: _ols([_one(x["dap_cm"]), x["dap_cm"], x["dap_cm"] ** 2], y, ("b0", "b1", "b2"))),
    Form("height.richards", "ht_m", "Richards", "b0/(1 + exp(b1 - b2*dap_cm))**(1/b3)", ("dap_cm",),
         ("b0", "b1", "b2", "b3"), lambda x, y: {"b0": _asymptote(y), "b1": 1.0, "b2": 0.1, "b3": 1.0},
         reference="Richards (1959)"),
    Form("height.chapman_richards", "ht_m", "Chapman-Richards", "1.3 + b0*(1 - exp(-b1*dap_cm))**b2", ("dap_cm",),
         ("b0", "b1", "b2"), lambda x, y: {"b0": _asymptote(y), "b1": 0.05, "b2": 1.0}),
    Form("height.henricksen", "ht_m", "Henricksen", "b0 + b1*log(dap_cm)", ("dap_cm",), ("b0", "b1"),
         lambda x, y: _ols([_one(x["dap_cm"]), np.log(x["dap_cm"])], y, ("b0", "b1")), reference="Henricksen (1950)"),
    Form("height.stoffels", "ht_m", "Stoffels", "exp(b0 + b1*log(dap_cm))", ("dap_cm",), ("b0", "b1"),
         lambda x, y: _ols([_one(x["dap_cm"]), np.log(x["dap_cm"])], np.log(y), ("b0", "b1")),
         fit_text="b0 + b1*log(dap_cm)", reference="Stoffels & van Soest (1953)"),
    # stem volume: dap_cm, ht_m -> vol_m3
    Form("volume.schumacher_hall", "vol_m3", "Schumacher-Hall", "b0*dap_cm**b1*ht_m**b2", ("dap_cm", "ht_m"),
         ("b0", "b1", "b2"),
         lambda x, y: (lambda c: {"b0": float(np.exp(c["b0"])), "b1": c["b1"], "b2": c["b2"]})(
             _ols([_one(x["dap_cm"]), np.log(x["dap_cm"]), np.log(x["ht_m"])], np.log(y), ("b0", "b1", "b2"))),
         reference="Schumacher & Hall (1933)"),
    Form("volume.schumacher_hall_log", "vol_m3", "Schumacher-Hall (log)", "exp(b0 + b1*log(dap_cm) + b2*log(ht_m))",
         ("dap_cm", "ht_m"), ("b0", "b1", "b2"),
         lambda x, y: _ols([_one(x["dap_cm"]), np.log(x["dap_cm"]), np.log(x["ht_m"])], np.log(y), ("b0", "b1", "b2")),
         fit_text="b0 + b1*log(dap_cm) + b2*log(ht_m)", reference="Schumacher & Hall (1933)"),
    Form("volume.spurr", "vol_m3", "Spurr", "b0 + b1*dap_cm**2*ht_m", ("dap_cm", "ht_m"), ("b0", "b1"),
         lambda x, y: _ols([_one(x["dap_cm"]), x["dap_cm"] ** 2 * x["ht_m"]], y, ("b0", "b1")), reference="Spurr (1952)"),
    Form("volume.husch_log", "vol_m3", "Husch (log)", "exp(b0 + b1*log(dap_cm))", ("dap_cm",), ("b0", "b1"),
         lambda x, y: _ols([_one(x["dap_cm"]), np.log(x["dap_cm"])], np.log(y), ("b0", "b1")),
         fit_text="b0 + b1*log(dap_cm)", reference="Husch (1963)"),
    Form("volume.form_factor", "vol_m3", "Fator de forma", "b0*ht_m*pi*dap_cm**2/40000", ("dap_cm", "ht_m"), ("b0",),
         lambda x, y: _ols([x["ht_m"] * np.pi * x["dap_cm"] ** 2 / 40000], y, ("b0",))),
    # aboveground biomass -> agb_kg
    Form("biomass.power_log", "agb_kg", "Potência do DAP (log)", "exp(b0 + b1*log(dap_cm))", ("dap_cm",), ("b0", "b1"),
         lambda x, y: _ols([_one(x["dap_cm"]), np.log(x["dap_cm"])], np.log(y), ("b0", "b1")),
         fit_text="b0 + b1*log(dap_cm)"),
    Form("biomass.d2h", "agb_kg", "D²H", "b0*(dap_cm**2*ht_m)**b1", ("dap_cm", "ht_m"), ("b0", "b1"),
         lambda x, y: (lambda c: {"b0": float(np.exp(c["b0"])), "b1": c["b1"]})(
             _ols([_one(x["dap_cm"]), np.log(x["dap_cm"] ** 2 * x["ht_m"])], np.log(y), ("b0", "b1")))),
    Form("biomass.chave2014", "agb_kg", "ρD²H (Chave)", "b0*(rho_gcm3*dap_cm**2*ht_m)**b1",
         ("rho_gcm3", "dap_cm", "ht_m"), ("b0", "b1"), lambda x, y: {"b0": 0.0673, "b1": 0.976},
         reference="Chave et al. (2014), Global Change Biology 20:3177-3190"),
    # DBH from the drone's crown area -> dap_cm
    Form("dap.sqrt_crown", "dap_cm", "DAP pela copa", "b0 + b1*sqrt(crown_m2)", ("crown_m2",), ("b0", "b1"),
         lambda x, y: _ols([_one(x["crown_m2"]), np.sqrt(x["crown_m2"])], y, ("b0", "b1"))),
]

FORMS: dict[str, Form] = {f.key: f for f in _FORMS}


def form(key: str) -> Form:
    try:
        return FORMS[key]
    except KeyError:
        raise KeyError(f"unknown form {key!r}; known: {', '.join(FORMS)}") from None


def forms_for(target: str) -> list[Form]:
    return [f for f in _FORMS if f.target == target]
