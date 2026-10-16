"""Formulas evaluate exactly what they say, and only that (cl_allometry.expr, cl_allometry.forms)."""

import math

import numpy as np
import pytest

from cl_allometry import FORMS, form, forms_for
from cl_allometry.expr import Formula


def test_a_formula_evaluates_its_text():
    f = Formula("b0*exp(b1/dap_cm) + sqrt(x)*pi - log(y)", ("dap_cm", "b0", "b1", "x", "y"))
    got = f(dap_cm=25, b0=2, b1=-5, x=4, y=math.e)
    assert got == pytest.approx(2 * math.exp(-0.2) + 2 * math.pi - 1)


@pytest.mark.parametrize("text", ["__import__('os').system('x')", "dap_cm.__class__", "open('f')", "'texto'",
                                  "[1, 2]", "lambda: 1", "b9 + 1"])
def test_a_formula_refuses_anything_but_arithmetic_of_its_names(text):
    with pytest.raises(ValueError):
        Formula(text, ("dap_cm", "b0"))


def test_every_form_reads_and_names_its_target():
    for f in FORMS.values():
        f.formula()
        f.fit_formula()
        assert f.target in ("ht_m", "vol_m3", "agb_kg", "dap_cm") and set(f.params) and set(f.variables)
    assert {f.key for f in forms_for("ht_m")} >= {"height.curtis", "height.logistic", "height.gompertz",
                                                   "height.parabolic", "height.richards"}


# each form against the same expression worked out by hand
CASES = [
    ("height.curtis", {"dap_cm": 25}, {"b0": 3.0, "b1": -5.0}, math.exp(3 - 5 / 25)),
    ("height.curtis_log", {"dap_cm": 25}, {"b0": 3.0, "b1": -5.0}, math.exp(3 - 5 / 25)),
    ("height.logistic", {"dap_cm": 20}, {"b0": 20, "b1": 5, "b2": 0.1}, 20 / (1 + 5 * math.exp(-2))),
    ("height.gompertz", {"dap_cm": 20}, {"b0": 22, "b1": 1, "b2": 0.1}, 22 * math.exp(-math.exp(1 - 2))),
    ("height.parabolic", {"dap_cm": 20}, {"b0": -1, "b1": 1.2, "b2": -0.016}, -1 + 24 - 0.016 * 400),
    ("height.chapman_richards", {"dap_cm": 20}, {"b0": 21, "b1": 0.1, "b2": 2.4},
     1.3 + 21 * (1 - math.exp(-2)) ** 2.4),
    ("volume.schumacher_hall", {"dap_cm": 30, "ht_m": 20}, {"b0": 6e-5, "b1": 1.9, "b2": 0.9},
     6e-5 * 30**1.9 * 20**0.9),
    ("volume.form_factor", {"dap_cm": 30, "ht_m": 20}, {"b0": 0.5}, 0.5 * 20 * math.pi * 900 / 40000),
    ("biomass.chave2014", {"rho_gcm3": 0.6, "dap_cm": 30, "ht_m": 20}, {"b0": 0.0673, "b1": 0.976},
     0.0673 * (0.6 * 900 * 20) ** 0.976),
    ("dap.sqrt_crown", {"crown_m2": 19.4}, {"b0": 3.511, "b1": 3.3326}, 3.511 + 3.3326 * math.sqrt(19.4)),
]


@pytest.mark.parametrize("key,values,coef,expected", CASES)
def test_each_form_against_a_hand_calculation(key, values, coef, expected):
    assert float(form(key).formula()(**values, **coef)) == pytest.approx(expected, rel=1e-12)


def test_the_drone_model_of_cl_forest_gives_its_documented_value():
    # metadados.json of dados/uav/2026-01-27: a 19.4 m² crown is an 18.2 cm DBH
    assert float(form("dap.sqrt_crown").formula()(crown_m2=19.4, b0=3.511, b1=3.3326)) == pytest.approx(18.2, abs=0.05)


def test_an_unknown_form_says_which_exist():
    with pytest.raises(KeyError, match="height.curtis"):
        form("height.nope")
    assert np.isfinite(form("height.curtis").formula()(dap_cm=np.array([10.0, 20.0]), b0=3, b1=-5)).all()
