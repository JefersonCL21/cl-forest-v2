"""The Model Lab page runs in every tab and its main flows work, in local mode, on a small register (the page,
pages/5_🧪Laboratorio_de_Modelos.py). Needs Streamlit, so CI (which installs only the package's dependencies) skips
it; run it with the app's environment."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from cl_allometry.registry import Filter, Scope  # noqa: E402
from synthetic import WHO, lab  # noqa: E402

PAGE = str(Path(__file__).parents[1] / "pages" / "5_🧪Laboratorio_de_Modelos.py")


@pytest.fixture
def page(tmp_path, monkeypatch):
    reg, data, m = lab(tmp_path)
    rec = reg.run_fit(data, m, "height.logistic", Filter(especie=("Khaya ivorensis",)), author=WHO)
    reg.promote(rec.fit_id, Scope.species("Khaya ivorensis"), author=WHO, reason="modelo da espécie")
    monkeypatch.setenv("ALLOMETRY_STORE", str(reg.store.root))
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.chdir(Path(__file__).parents[1])  # the page reads imagens/ relative to the app
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.run()
    return at, reg


def test_every_tab_runs_without_an_error(page):
    at, _ = page
    assert not at.exception and not at.error
    assert len(at.tabs) == 5
    assert any("Recomendado" in s.value for s in at.success)


def test_fitting_from_the_page_keeps_the_fits(page):
    at, reg = page
    at.multiselect(key="fit_species").set_value(["Dalbergia nigra"])
    at.multiselect(key="fit_forms").set_value(["height.logistic", "height.curtis"])
    at.run()
    at.button(key="fit_go").click().run()
    assert not at.exception and not at.error
    fitted = [r for r in reg.list_fits("ht_m") if r.filter["especie"] == ["Dalbergia nigra"]]
    assert {r.form for r in fitted} == {"height.logistic", "height.curtis"}
    assert all(r.author == "dev@local" for r in fitted)


def test_applying_from_the_page_estimates_and_offers_the_file(page):
    at, _ = page
    at.radio(key="many_src").set_value("uma versão registrada").run()
    at.button(key="many_go").click().run()
    assert not at.exception and not at.error
    assert any(b.label.startswith("Baixar com as estimativas") for b in at.get("download_button"))
    assert [m.label for m in at.metric] == ["altura total (m)"]  # the one-tree calculator
