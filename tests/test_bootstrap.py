"""Version 1 of the height models in the register changes no estimate in use: the refitted logistic of each species
is the one that made HT_Est, it is promoted with its reason, and doing it again adds nothing
(cl_allometry.legacy.bootstrap_v1, the decision of 16/10/2026)."""

from pathlib import Path

import pytest

from cl_allometry import datasets, legacy
from cl_allometry.registry import Registry, Scope
from cl_allometry.store import LocalStore

CSV = Path(__file__).parents[1] / "dados" / "GERAL_V2.csv"
MEASURED = {"Khaya grandifoliola": 448, "Khaya ivorensis": 248, "Plathymenia reticulata": 109, "Dalbergia nigra": 90,
            "Handroanthus serratifolius": 72, "Bowdichia virgilioides": 37, "Handroanthus chrysotrichus": 16,
            "Cariniana legalis": 13, "Caesalpinia echinata": 9}


@pytest.fixture(scope="module")
def boot(tmp_path_factory):
    if not CSV.is_file():
        pytest.skip("dados/GERAL_V2.csv not here")
    store = LocalStore(tmp_path_factory.mktemp("store"))
    report = legacy.bootstrap_v1(store, CSV, author="dev@local", decided_by="Jeferson", decided_on="16/10/2026")
    return store, report


def test_version_1_reproduces_the_coefficients_that_made_ht_est(boot):
    store, report = boot
    assert report.promoted and report.dataset_id == "inventario@v1"
    reg = Registry(store)
    for row in report.species:
        assert row["logistic_vs_table"] < 1e-3, row
        champ = reg.champion("ht_m", Scope.species(row["species"]))
        assert champ.record.form == "height.logistic" and champ.fit_id == row["logistic_fit_id"]
        table = legacy.model("logistic", row["species"], "table")
        assert champ.predict(dap_cm=20.0) == pytest.approx(table.predict(dap_cm=20.0), rel=1e-3)


def test_every_species_is_fitted_on_all_its_measured_heights(boot):
    report = boot[1]
    assert {r["species"]: r["n"] for r in report.species} == MEASURED
    assert all(r["fits"] == len([f for f in legacy.forms_for("ht_m")]) for r in report.species)


def test_the_promotion_says_why_and_warns_about_small_samples(boot):
    store, report = boot
    reg = Registry(store)
    for sp, n in MEASURED.items():
        e = reg.champion("ht_m", Scope.species(sp)).event
        assert report.sha256[:12] in e.reason and "Jeferson" in e.reason and "16/10/2026" in e.reason
        assert (f"só {n} alturas medidas" in e.warnings) == (n < 20), (sp, e.warnings)


def test_curtis_fitted_in_logs_is_the_curtis_of_before(boot):
    """The old Curtis was the straight line of ln(HT) on 1/DAP: height.curtis_log gives its coefficients back."""
    assert all(r["curtis_log_vs_legacy"] < 1e-6 for r in boot[1].species)


def test_running_it_again_adds_nothing(boot):
    store, _ = boot
    before = store.list("")
    again = legacy.bootstrap_v1(store, CSV, author="dev@local", decided_by="Jeferson", decided_on="16/10/2026")
    assert not again.created_version and again.promoted
    assert store.list("") == before


def test_it_finds_version_1_even_after_newer_versions_were_imported(tmp_path):
    if not CSV.is_file():
        pytest.skip("dados/GERAL_V2.csv not here")
    store = LocalStore(tmp_path / "store")
    kw = dict(author="dev@local", decided_by="Jeferson", decided_on="16/10/2026", only=["Caesalpinia echinata"],
              promote=False)
    legacy.bootstrap_v1(store, CSV, **kw)
    raw = datasets.read_upload(CSV.name, CSV.read_bytes())
    raw.loc[0, "HT"] = "18.0"  # a height measured later: inventario@v2
    datasets.import_dataset(store, raw, family="inventario", mapping=datasets.INVENTORY_MAPPING, author="dev@local",
                            source_name="GERAL_V2.csv")
    again = legacy.bootstrap_v1(store, CSV, **kw)
    assert again.dataset_id == "inventario@v1" and not again.created_version
    assert [v.version for v in datasets.versions(store, "inventario")] == [1, 2]
