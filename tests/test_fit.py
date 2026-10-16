"""A fit gives back the coefficients the data were made with, its statistics are the textbook ones, and it says when
it failed (cl_allometry.fit, cl_allometry.metrics)."""

import numpy as np
import pandas as pd
import pytest

from cl_allometry import cross_validate, fit, form
from cl_allometry import metrics as M

RNG = np.random.default_rng(42)
N = 400
DAP = RNG.uniform(5, 60, N)
HT = RNG.uniform(5, 30, N)


def synthetic(key, coef, noise=0.03, seed=1):
    """Data made with known coefficients and a small multiplicative noise."""
    rng = np.random.default_rng(seed)
    f = form(key)
    x = {"dap_cm": DAP, "ht_m": HT, "rho_gcm3": rng.uniform(0.4, 0.9, N), "crown_m2": rng.uniform(2, 60, N)}
    x = {v: x[v] for v in f.variables}
    y = f.formula()(**x, **coef) * np.exp(rng.normal(0, noise, N))
    return pd.DataFrame({**x, f.target: y})


TRUE = [
    ("height.curtis", {"b0": 3.4, "b1": -12.5}),
    ("height.curtis_log", {"b0": 3.4, "b1": -12.5}),
    ("height.logistic", {"b0": 21.4, "b1": 6.87, "b2": 0.157}),
    ("height.gompertz", {"b0": 22.2, "b1": 1.09, "b2": 0.115}),
    ("height.parabolic", {"b0": -1.1, "b1": 1.19, "b2": -0.0157}),
    ("height.chapman_richards", {"b0": 21.2, "b1": 0.10, "b2": 2.37}),
    ("height.stoffels", {"b0": 0.8, "b1": 0.6}),
    ("volume.schumacher_hall", {"b0": 6e-5, "b1": 1.9, "b2": 0.9}),
    ("volume.schumacher_hall_log", {"b0": -9.7, "b1": 1.9, "b2": 0.9}),
    ("volume.spurr", {"b0": 0.01, "b1": 3.5e-5}),
    ("biomass.d2h", {"b0": 0.06, "b1": 0.95}),
    ("biomass.chave2014", {"b0": 0.0673, "b1": 0.976}),
    ("dap.sqrt_crown", {"b0": 3.511, "b1": 3.3326}),
]


@pytest.mark.parametrize("key,coef", TRUE)
def test_a_fit_gives_back_the_coefficients_the_data_were_made_with(key, coef):
    df = synthetic(key, coef)
    f = fit(key, df, seed=0)
    assert f.converged, f.message
    for k, v in coef.items():
        # within 4 standard errors of the value the data were made with (an intercept can be loose, never far)
        assert abs(f.coefficients[k] - v) <= 4 * f.stderr[k], (key, k, f.coefficients[k], v, f.stderr[k])
        lo, hi = f.ci95[k]
        assert lo <= f.coefficients[k] <= hi and f.stderr[k] > 0
    # and the fitted curve is the true one: the predictions within 1% of the noiseless values, on average
    x = {v: df[v].to_numpy() for v in form(key).variables}
    truth = form(key).formula()(**x, **coef)
    assert np.mean(np.abs(f.predict(**x) / truth - 1)) < 0.01, key
    assert f.metrics["r2"] > 0.8 and abs(f.metrics["bias_pct"]) < 2
    assert set(f.ranges) == {*form(key).variables, form(key).target}


def test_a_log_linear_fit_corrects_the_bias_of_going_back_from_logs():
    df = synthetic("volume.schumacher_hall_log", {"b0": -9.7, "b1": 1.9, "b2": 0.9}, noise=0.15)
    f = fit("volume.schumacher_hall_log", df)
    s2 = np.log(f.correction) * 2
    assert f.correction > 1 and s2 == pytest.approx(0.15**2, rel=0.2)
    assert abs(f.metrics["bias_pct"]) < 2  # without the correction the volume would come out ~1% short
    assert f.metrics["furnival"] != f.metrics["syx"]  # judged on the same scale as the untransformed forms


def test_the_same_data_and_seed_give_the_same_coefficients():
    df = synthetic("height.logistic", {"b0": 21.4, "b1": 6.87, "b2": 0.157})
    assert fit("height.logistic", df, seed=3).coefficients == fit("height.logistic", df, seed=3).coefficients


def test_a_fit_that_cannot_be_made_says_so_and_keeps_no_coefficients():
    few = pd.DataFrame({"dap_cm": [10.0, 20.0], "ht_m": [8.0, 12.0]})
    f = fit("height.logistic", few)
    assert not f.converged and f.coefficients is None and "poucos dados" in f.message
    with pytest.raises(ValueError, match="no coefficients"):
        f.predict(dap_cm=10)
    with pytest.raises(ValueError, match="ht_m"):
        fit("height.curtis", pd.DataFrame({"dap_cm": [1.0]}))


def test_missing_or_non_positive_values_are_left_out_and_counted():
    df = synthetic("height.curtis", {"b0": 3.4, "b1": -12.5})
    df.loc[:9, "ht_m"] = np.nan
    df.loc[10:14, "dap_cm"] = 0
    f = fit("height.curtis", df)
    assert f.dropped == 15 and f.n == N - 15


def test_a_weighted_fit_uses_the_weights_formula():
    df = synthetic("volume.schumacher_hall", {"b0": 6e-5, "b1": 1.9, "b2": 0.9})
    f = fit("volume.schumacher_hall", df, weights="1/dap_cm**2")
    assert f.converged and f.weights == "1/dap_cm**2"
    curtis = synthetic("height.curtis", {"b0": 3.4, "b1": -12.5})
    assert not fit("height.curtis", curtis, weights="1/(dap_cm-dap_cm)").converged


def test_the_statistics_are_the_textbook_ones():
    y = np.array([10.0, 12.0, 15.0, 9.0, 14.0])
    pred = np.array([11.0, 11.5, 14.0, 10.0, 13.0])
    m = M.summary(y, pred, p=2)
    resid = y - pred
    sse = (resid**2).sum()
    assert m["bias"] == pytest.approx(resid.mean())  # positive: the model underestimates
    assert m["rmse"] == pytest.approx(np.sqrt(sse / 5)) and m["syx"] == pytest.approx(np.sqrt(sse / 3))
    assert m["r2"] == pytest.approx(1 - sse / ((y - y.mean()) ** 2).sum())
    assert m["rmse_pct"] == pytest.approx(100 * m["rmse"] / y.mean())
    assert m["aic"] == pytest.approx(5 * np.log(sse / 5) + 2 * 3)
    assert M.summary(y, pred, 2, log_syx=0.1)["furnival"] == pytest.approx(np.exp(np.log(y).mean()) * 0.1)


def test_the_residual_checks_catch_a_spreading_variance_and_an_influential_tree():
    x = np.linspace(1, 50, 300)
    noise = np.random.default_rng(0).normal(0, 1, 300)
    spreading = M.residual_checks(noise * x / 10, x, None, 2)
    steady = M.residual_checks(noise, x, None, 2)
    assert spreading["breusch_pagan_p"] < 0.01 < steady["breusch_pagan_p"]
    df = synthetic("height.curtis", {"b0": 3.4, "b1": -12.5}, noise=0.02)
    df.loc[0, ["dap_cm", "ht_m"]] = [59.0, 2.0]  # a big tree measured 2 m tall
    assert fit("height.curtis", df).checks["cook_max"] > 1


def test_cross_validation_judges_each_tree_by_a_fit_that_did_not_see_it():
    df = synthetic("height.logistic", {"b0": 21.4, "b1": 6.87, "b2": 0.157})
    cv = cross_validate("height.logistic", df, k=5, seed=0)
    inside = fit("height.logistic", df).metrics
    assert cv["converged"] and cv["n"] == N and cv["rmse"] >= inside["rmse"] * 0.95
