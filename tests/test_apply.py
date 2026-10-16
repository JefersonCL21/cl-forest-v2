"""Applying the champions: a measured value is kept and only the missing ones are estimated, each value says where it
came from and which fit made it, and an estimate outside the fitted range is marked, never clipped
(cl_allometry.apply)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cl_allometry import datasets as D
from cl_allometry.apply import apply, predict_one
from cl_allometry.registry import Filter, Registry, Scope
from cl_allometry.store import LocalStore
from synthetic import WHO, lab

CSV = Path(__file__).parents[1] / "dados" / "GERAL_V2.csv"


@pytest.fixture
def promoted(tmp_path):
    """Khaya grandifoliola has its own model, the genus Khaya one too, and there is no generic model."""
    reg, data, m = lab(tmp_path)
    sp = reg.run_fit(data, m, "height.logistic", Filter(especie=("Khaya grandifoliola",)), author=WHO)
    genus = reg.run_fit(data, m, "height.logistic", Filter(genus="Khaya"), author=WHO)
    reg.promote(sp.fit_id, Scope.species("Khaya grandifoliola"), author=WHO, reason="ajuste da espécie")
    reg.promote(genus.fit_id, Scope.genus("Khaya"), author=WHO, reason="ajuste do gênero")
    return reg, sp, genus


TREES = pd.DataFrame({
    "especie": ["Khaya grandifoliola", "Khaya grandifoliola", "Khaya ivorensis", "Dalbergia nigra",
                "Khaya grandifoliola", "Khaya grandifoliola", None],
    "dap_cm": [20.0, 20.0, 20.0, 20.0, np.nan, 80.0, 20.0],
    "ht_m": [13.1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
}, index=[10, 11, 12, 13, 14, 15, 15])  # any index, even repeated


def test_a_measured_height_is_kept_and_only_the_missing_ones_are_estimated(promoted):
    reg, sp, _ = promoted
    out = apply(TREES, "ht_m", reg)
    assert out["ht_m_final"].iloc[0] == 13.1 and out["ht_m_source"].iloc[0] == "measured"
    assert out["ht_m_est"].iloc[0] == pytest.approx(float(sp.fit.predict(dap_cm=20.0)))  # still estimated, to compare
    assert out["ht_m_final"].iloc[1] == pytest.approx(float(sp.fit.predict(dap_cm=20.0)))
    assert list(out.index) == list(TREES.index) and out["ht_m"].equals(TREES["ht_m"])  # nothing else changed


def test_each_tree_says_where_its_value_came_from(promoted):
    reg, sp, genus = promoted
    out = apply(TREES, "ht_m", reg)
    assert list(out["ht_m_source"]) == ["measured", "species", "genus", "none", "none", "species", "none"]
    assert list(out["ht_m_model_id"].iloc[:3]) == [sp.fit_id, sp.fit_id, genus.fit_id]
    assert out["ht_m_model_id"].iloc[3] is None and np.isnan(out["ht_m_final"].iloc[3])  # no Dalbergia model


def test_a_tree_without_its_inputs_gets_no_estimate(promoted):
    out = apply(TREES, "ht_m", promoted[0])
    assert np.isnan(out["ht_m_est"].iloc[4]) and out["ht_m_source"].iloc[4] == "none"


def test_out_of_range_is_marked_not_clipped(promoted):
    reg, sp, _ = promoted
    out = apply(TREES, "ht_m", reg)
    assert bool(out["ht_m_out_of_range"].iloc[5]) and not out["ht_m_out_of_range"].iloc[1]
    assert out["ht_m_est"].iloc[5] == pytest.approx(float(sp.fit.predict(dap_cm=80.0)))


def test_a_measured_value_can_be_distrusted(promoted):
    flagged = pd.Series([True, False, False, False, False, False, False], index=TREES.index)
    out = apply(TREES, "ht_m", promoted[0], measured_ok=~flagged)
    assert out["ht_m_source"].iloc[0] == "species"


def test_predict_one_gives_what_apply_gives(promoted):
    reg = promoted[0]
    out = apply(TREES, "ht_m", reg)
    one = predict_one(reg, "ht_m", "Khaya ivorensis", dap_cm=20.0)
    assert one.value == pytest.approx(out["ht_m_est"].iloc[2]) and one.level == "genus" and not one.out_of_range
    assert predict_one(reg, "ht_m", "Khaya grandifoliola", dap_cm=80).out_of_range
    with pytest.raises(LookupError):
        predict_one(reg, "ht_m", "Dalbergia nigra", dap_cm=20)


def test_apply_works_for_any_target_and_chains(tmp_path):
    """A volume model fed with the heights of the height models: measured where there is one, estimated elsewhere."""
    reg, data, m = lab(tmp_path)
    genus = reg.run_fit(data, m, "height.logistic", Filter(genus="Khaya"), author=WHO)
    reg.promote(genus.fit_id, Scope.genus("Khaya"), author=WHO, reason="ajuste do gênero")
    rng = np.random.default_rng(5)
    dap, ht = rng.uniform(10, 40, 150), rng.uniform(8, 22, 150)
    vol = 6e-5 * dap**1.9 * ht**0.9 * np.exp(rng.normal(0, 0.05, 150))
    cubed = pd.DataFrame({"Especie": "Khaya ivorensis", "DAP": dap.round(2).astype(str),
                          "HT": ht.round(2).astype(str), "Volume": vol.round(5).astype(str)})
    r = D.import_dataset(reg.store, cubed, family="cubagem", mapping=D.suggest_mapping(cubed.columns), author=WHO,
                         source_name="cubagem.csv")
    v = reg.run_fit(*D.load(reg.store, r.manifest.dataset_id), "volume.schumacher_hall", Filter(), author=WHO)
    reg.promote(v.fit_id, Scope.generic(), author=WHO, reason="Schumacher-Hall da cubagem")

    heights = apply(TREES, "ht_m", reg)
    out = apply(heights, "vol_m3", reg, inputs={"ht_m": "ht_m_final"})
    assert out["vol_m3_final"].iloc[0] == pytest.approx(float(v.fit.predict(dap_cm=20.0, ht_m=13.1)))  # measured
    ht_est = float(genus.fit.predict(dap_cm=20.0))
    assert out["vol_m3_final"].iloc[1] == pytest.approx(float(v.fit.predict(dap_cm=20.0, ht_m=ht_est)))
    assert out["vol_m3_source"].iloc[1] == "generic" and out["vol_m3_model_id"].iloc[1] == v.fit_id
    assert list(out["vol_m3_source"].iloc[3:5]) == ["none", "none"]  # no height for Dalbergia; no DAP


@pytest.fixture(scope="module")
def census():
    if not CSV.is_file():
        pytest.skip("dados/GERAL_V2.csv not here")
    return D.read_upload(CSV.name, CSV.read_bytes())


def test_the_champion_of_version_1_gives_the_ht_est_of_the_census(census, tmp_path):
    """Applying the refitted logistic of Khaya grandifoliola reproduces the HT_Est the app uses today, and gives one
    to the 88 trees of talhão 13 that were measured after HT_Est was computed."""
    store = LocalStore(tmp_path / "store")
    r = D.import_dataset(store, census, family="inventario", mapping=D.INVENTORY_MAPPING, author=WHO,
                         source_name="GERAL_V2.csv")
    data, m = D.load(store, r.manifest.dataset_id)
    reg = Registry(store)
    rec = reg.run_fit(data, m, "height.logistic", Filter(especie=("Khaya grandifoliola",), exclude_flags=()),
                      author=WHO, cv_k=None)
    reg.promote(rec.fit_id, Scope.species("Khaya grandifoliola"), author=WHO, reason="versão 1 da espécie")
    out = apply(data, "ht_m", reg)
    k = out[(out["especie"] == "Khaya grandifoliola") & out["dap_cm"].notna()]
    ht_est = pd.to_numeric(k["HT_Est"])
    assert (k["ht_m_source"].isin(["measured", "species"])).all() and k["ht_m_est"].notna().all()
    had = ht_est.notna()
    assert had.sum() == 1939 and np.max(np.abs(k.loc[had, "ht_m_est"] / ht_est[had] - 1)) < 1e-3
    assert (~had).sum() == 88 and set(k.loc[~had, "talhao"]) == {"13"}
    assert (out.loc[out["especie"] != "Khaya grandifoliola", "ht_m_source"].isin(["measured", "none"])).all()
