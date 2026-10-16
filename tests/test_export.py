"""A model used outside Python gives the same values: the SQL generated from the formula equals the Python model for
every form, and the model card carries everything needed to rebuild the predictions (cl_allometry.export)."""

import json
import math
import sqlite3

import numpy as np
import pytest

from cl_allometry import FORMS
from cl_allometry.expr import Formula
from cl_allometry.export import model_card, models_table, to_sql
from cl_allometry.registry import Filter, Scope
from synthetic import WHO, lab

# plausible coefficients for every form, with negative ones where the forms have them
COEF = {
    "height.curtis": {"b0": 3.4, "b1": -12.5},
    "height.curtis_log": {"b0": 3.4, "b1": -12.5},
    "height.logistic": {"b0": 21.4, "b1": 6.87, "b2": 0.157},
    "height.gompertz": {"b0": 22.2, "b1": 1.09, "b2": 0.115},
    "height.parabolic": {"b0": -1.1, "b1": 1.19, "b2": -0.0157},
    "height.richards": {"b0": 22.0, "b1": 1.5, "b2": 0.12, "b3": 0.8},
    "height.chapman_richards": {"b0": 21.2, "b1": 0.10, "b2": 2.37},
    "height.henricksen": {"b0": -4.0, "b1": 6.1},
    "height.stoffels": {"b0": 0.8, "b1": 0.6},
    "volume.schumacher_hall": {"b0": 6e-5, "b1": 1.9, "b2": 0.9},
    "volume.schumacher_hall_log": {"b0": -9.7, "b1": 1.9, "b2": 0.9},
    "volume.spurr": {"b0": 0.01, "b1": 3.5e-5},
    "volume.husch_log": {"b0": -8.1, "b1": 2.4},
    "volume.form_factor": {"b0": 0.5},
    "biomass.power_log": {"b0": -2.1, "b1": 2.5},
    "biomass.d2h": {"b0": 0.06, "b1": 0.95},
    "biomass.chave2014": {"b0": 0.0673, "b1": 0.976},
    "dap.sqrt_crown": {"b0": 3.511, "b1": 3.3326},
}
TREES = [{"dap_cm": 23, "ht_m": 14, "rho_gcm3": 0.6, "crown_m2": 19},  # integers: no integer division
         {"dap_cm": 17.53, "ht_m": 11.2, "rho_gcm3": 0.87, "crown_m2": 7.25}]


def test_every_form_has_coefficients_here():
    assert set(COEF) == set(FORMS)


def sqlite_db():
    db = sqlite3.connect(":memory:")
    for name, fn, n in [("EXP", math.exp, 1), ("LN", math.log, 1), ("SQRT", math.sqrt, 1), ("POWER", math.pow, 2),
                        ("ACOS", math.acos, 1)]:
        db.create_function(name, n, fn, deterministic=True)
    db.execute('CREATE TABLE t (dap_cm, ht_m, rho_gcm3, crown_m2)')
    db.executemany("INSERT INTO t VALUES (?, ?, ?, ?)", [tuple(t.values()) for t in TREES])
    return db


@pytest.mark.parametrize("key", sorted(COEF))
def test_the_sql_gives_the_python_value_for_every_form(key):
    f, coef = FORMS[key], COEF[key]
    correction = 1.0123 if f.log_linear else 1.0
    got = [row[0] for row in sqlite_db().execute(f"SELECT {to_sql(f, coef, correction=correction, dialect='sqlite')}"
                                                 " FROM t")]
    for tree, value in zip(TREES, got):
        expected = float(f.formula()(**{v: tree[v] for v in f.variables}, **coef)) * correction
        assert value == pytest.approx(expected, rel=1e-12), key


def test_each_dialect_uses_its_own_functions_and_quoting():
    bq = to_sql("height.curtis", COEF["height.curtis"])
    pg = to_sql("height.curtis", COEF["height.curtis"], dialect="postgres", columns={"dap_cm": "dap"})
    assert bq == "EXP((3.4 + ((-12.5) / CAST(`dap_cm` AS FLOAT64))))"
    assert '"dap"' in pg and "DOUBLE PRECISION" in pg and "`" not in pg
    safe = to_sql("volume.schumacher_hall_log", COEF["volume.schumacher_hall_log"], safe=True)
    assert "SAFE.EXP(" in safe and "SAFE.LN(" in safe
    assert "SAFE_DIVIDE(" in to_sql("height.curtis", COEF["height.curtis"], safe=True)
    with pytest.raises(ValueError):
        to_sql("height.curtis", COEF["height.curtis"], dialect="oracle")


def test_a_coefficient_is_written_with_all_its_digits():
    b0 = 21.415713184567893
    sql = to_sql("height.logistic", {"b0": b0, "b1": 6.87, "b2": 0.157})
    assert repr(b0) in sql


@pytest.fixture
def champion_card(tmp_path):
    reg, data, m = lab(tmp_path)
    rec = reg.run_fit(data, m, "height.logistic", Filter(especie=("Khaya ivorensis",)), author=WHO)
    reg.promote(rec.fit_id, Scope.species("Khaya ivorensis"), author=WHO, reason="menor RMSE% na validação cruzada")
    champ = reg.champion("ht_m", Scope.species("Khaya ivorensis"))
    return reg, (data, m), rec, champ, model_card(rec, champ)


def test_the_model_card_carries_everything_needed_to_use_the_model(champion_card):
    _, _, rec, champ, card = champion_card
    text = json.dumps(card, allow_nan=False)  # plain JSON
    card = json.loads(text)
    f = card["form"]
    rebuilt = Formula(f["formula"], f["variables"] + f["params"])(dap_cm=25.0, **card["coefficients"])
    assert float(rebuilt) * card["correction"] == pytest.approx(float(champ.predict(dap_cm=25.0)), rel=1e-12)
    assert card["dataset"] == {"id": "inventario@v1", "sha256": rec.dataset_sha256}
    assert card["champion"]["scope"] == "species:Khaya ivorensis" and card["champion"]["valid_range"]["dap_cm"]
    assert card["cross_validation"]["rmse_pct"] > 0 and card["fitted_by"] == WHO and card["sql"]["bigquery"]


def test_the_models_table_has_one_row_per_fit_and_marks_the_ones_in_use(champion_card):
    reg, (data, m), rec, _, _ = champion_card
    other = reg.run_fit(data, m, "height.curtis", Filter(especie=("Khaya ivorensis",)), author=WHO)
    t = models_table(reg.list_fits("ht_m"), reg.champions("ht_m")).set_index("fit_id")
    assert len(t) == 2 and t.loc[rec.fit_id, "champion_of"] == "species:Khaya ivorensis"
    assert t.loc[other.fit_id, "champion_of"] is None
    assert json.loads(t.loc[rec.fit_id, "coefficients"]) == rec.fit.coefficients
