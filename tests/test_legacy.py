"""The height models before cl_allometry, migrated as they were, and the evidence of which coefficients were fitted
to the field data (cl_allometry.legacy)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cl_allometry import fit, legacy

CSV = Path(__file__).parents[1] / "dados" / "GERAL_V2.csv"
KHAYA = "Khaya grandifoliola"


def test_the_nine_species_and_four_forms_of_both_sets_were_migrated():
    assert len(legacy.species()) == 9
    for sp in legacy.species():
        for name in legacy.FORMS:
            for source in ("code", "table"):
                assert legacy.model(name, sp, source).coefficients


def test_the_two_sets_disagree_where_the_review_found():
    code, table = legacy.model("logistic", KHAYA, "code"), legacy.model("logistic", KHAYA, "table")
    assert code.coefficients["b0"] == pytest.approx(21.76844341)
    assert table.coefficients["b0"] == pytest.approx(21.41571318)
    for sp in legacy.species():  # every parabolic b2 of the table is wrong
        assert (legacy.model("parabolic", sp, "table").coefficients["b2"]
                != legacy.model("parabolic", sp, "code").coefficients["b2"])
    for sp in set(legacy.species()) - {KHAYA}:  # the logistic is the same in both, but for Khaya grandifoliola
        code, table = legacy.model("logistic", sp, "code").coefficients, legacy.model("logistic", sp, "table").coefficients
        assert code == pytest.approx(table, rel=1e-8)  # the table rounds to 9 decimals


@pytest.fixture(scope="module")
def inventory():
    if not CSV.is_file():
        pytest.skip("dados/GERAL_V2.csv not here")
    df = pd.read_csv(CSV)
    return df.assign(dap_cm=pd.to_numeric(df["DAP"], errors="coerce"), ht_m=pd.to_numeric(df["HT"], errors="coerce"))


def test_ht_est_came_from_the_table_for_khaya_and_from_either_set_for_the_rest(inventory):
    for sp in legacy.species():
        d = inventory[(inventory["Especie"] == sp) & inventory["dap_cm"].notna() & inventory["HT_Est"].notna()]
        same = {s: np.abs(legacy.model("logistic", sp, s).predict(dap_cm=d["dap_cm"].to_numpy()) - d["HT_Est"]) < 1e-6
                for s in ("code", "table")}
        assert same["table"].all(), sp
        assert same["code"].all() == (sp != KHAYA), sp


def test_refitting_on_the_measured_heights_gives_the_table_not_the_code(inventory):
    """The evidence for the decision: the table's logistic of Khaya grandifoliola is the fit to its 448 measured
    heights; the code holds an older one. Its parabolic b2, though, is a typo: the code's is the right one."""
    d = inventory[(inventory["Especie"] == KHAYA) & (inventory["dap_cm"] > 0) & (inventory["ht_m"] > 0)]
    f = fit("height.logistic", d, seed=0)
    assert f.n == 448
    for k, v in legacy.model("logistic", KHAYA, "table").coefficients.items():
        assert f.coefficients[k] == pytest.approx(v, rel=1e-3)
    ranked = legacy.compare(d, KHAYA).set_index(["form", "source"])["rmse_pct"]
    assert ranked["height.parabolic", "table"] > 1000 > ranked["height.parabolic", "code"]
