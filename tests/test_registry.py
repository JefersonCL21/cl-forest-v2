"""Every fit is kept with what it saw and who made it; the model in use for a species is the one a person last
promoted, with a reason; nothing in the history is ever rewritten (cl_allometry.registry)."""

import json
import threading

import numpy as np
import pytest

from cl_allometry.registry import ConcurrentPromotion, Filter, Registry, Scope
from cl_allometry.store import LocalStore
from synthetic import SPECIES, WHO, lab


@pytest.fixture
def setup(tmp_path):
    return lab(tmp_path)


def species_fit(setup, sp="Khaya grandifoliola", form="height.logistic", **kw):
    reg, data, m = setup
    return reg.run_fit(data, m, form, Filter(especie=(sp,)), author=WHO, **kw)


def test_a_fit_record_keeps_the_data_hash_the_filter_the_cv_the_author_and_the_commit(setup):
    reg, _, m = setup
    rec = species_fit(setup)
    assert rec.fit.converged and rec.fit.n == 120 and rec.n_filtered == 120
    assert rec.dataset_id == "inventario@v1" and rec.dataset_sha256 == m.sha256
    assert rec.filter["especie"] == ["Khaya grandifoliola"] and "flag_duplicate_row" in rec.filter["exclude_flags"]
    assert rec.cv["converged"] and rec.cv["k"] == 5 and rec.cv["rmse_pct"] > 0
    assert rec.author == WHO and rec.commit and rec.created_at.startswith("2026-10-16")
    for k, v in zip(("b0", "b1", "b2"), SPECIES["Khaya grandifoliola"]):  # within 4 standard errors of the truth
        assert abs(rec.fit.coefficients[k] - v) <= 4 * rec.fit.stderr[k]
    assert reg.get_fit(rec.fit_id) == rec


def test_fitting_the_same_thing_again_finds_the_record_instead_of_adding_one(setup):
    reg = setup[0]
    first = species_fit(setup)
    again = species_fit(setup)
    assert again.fit_id == first.fit_id and again.created_at == first.created_at
    assert len(reg.store.list("fits/")) == 1
    other = species_fit(setup, seed=1)  # a different spec is a different fit
    assert other.fit_id != first.fit_id and len(reg.list_fits("ht_m")) == 2


def test_records_are_plain_json_any_system_reads(setup):
    reg = setup[0]
    rec = species_fit(setup)
    text = reg.store.get(reg._fit_path(rec.fit_id)).decode()
    record = json.loads(text, parse_constant=lambda c: pytest.fail(f"{c} in a record"))
    assert record["fit"]["coefficients"] == rec.fit.coefficients


def test_the_champion_is_the_latest_promotion_and_a_rollback_brings_back_the_one_before(setup):
    reg = setup[0]
    scope = Scope.species("Khaya grandifoliola")
    a, b = species_fit(setup), species_fit(setup, form="height.gompertz")
    assert reg.champion("ht_m", scope) is None
    reg.promote(a.fit_id, scope, author=WHO, reason="melhor RMSE na validação cruzada")
    reg.promote(b.fit_id, scope, author="bia@courageousland.com", reason="resíduos mais homogêneos")
    assert reg.champion("ht_m", scope).fit_id == b.fit_id
    reg.promote(a.fit_id, scope, author=WHO, reason="volta: o Gompertz superestima as árvores grandes")
    history = reg.history("ht_m", scope)
    assert [e.fit_id for e in history] == [a.fit_id, b.fit_id, a.fit_id] and [e.seq for e in history] == [1, 2, 3]
    assert reg.champion("ht_m", scope).fit_id == a.fit_id and history[1].author == "bia@courageousland.com"


def test_a_tree_gets_its_species_model_else_its_genus_else_the_generic(setup):
    reg, data, m = setup
    sp = species_fit(setup)
    genus = reg.run_fit(data, m, "height.logistic", Filter(genus="Khaya"), author=WHO)
    generic = reg.run_fit(data, m, "height.logistic", Filter(), author=WHO)
    reg.promote(sp.fit_id, Scope.species("Khaya grandifoliola"), author=WHO, reason="ajuste da espécie")
    reg.promote(genus.fit_id, Scope.genus("Khaya"), author=WHO, reason="ajuste do gênero")
    reg.promote(generic.fit_id, Scope.generic(), author=WHO, reason="ajuste de todas as espécies")
    lookup = reg.resolver("ht_m")
    for especie, expected in [("Khaya grandifoliola", sp), ("Khaya ivorensis", genus), ("Khaya senegalensis", genus),
                              ("Dalbergia nigra", generic), (None, generic), ("", generic)]:
        assert reg.resolve("ht_m", especie).fit_id == expected.fit_id, especie
        assert lookup(especie).fit_id == expected.fit_id, especie


def test_retiring_a_species_model_sends_its_trees_to_the_genus_model(setup):
    reg, data, m = setup
    sp = species_fit(setup)
    genus = reg.run_fit(data, m, "height.logistic", Filter(genus="Khaya"), author=WHO)
    reg.promote(sp.fit_id, Scope.species("Khaya grandifoliola"), author=WHO, reason="ajuste da espécie")
    reg.promote(genus.fit_id, Scope.genus("Khaya"), author=WHO, reason="ajuste do gênero")
    reg.retire("ht_m", Scope.species("Khaya grandifoliola"), author=WHO, reason="dados da espécie com erro")
    assert reg.champion("ht_m", Scope.species("Khaya grandifoliola")) is None
    assert reg.resolve("ht_m", "Khaya grandifoliola").fit_id == genus.fit_id
    assert [c.scope for c in reg.champions("ht_m")] == [Scope.genus("Khaya")]


def test_a_promotion_needs_a_reason_a_converged_fit_and_a_fit_of_its_scope(setup):
    reg, data, m = setup
    rec = species_fit(setup)
    with pytest.raises(ValueError, match="reason"):
        reg.promote(rec.fit_id, Scope.species("Khaya grandifoliola"), author=WHO, reason="ok")
    with pytest.raises(ValueError, match="cannot be the model"):  # a Khaya fit is not a Dalbergia model
        reg.promote(rec.fit_id, Scope.species("Dalbergia nigra"), author=WHO, reason="engano de espécie")
    with pytest.raises(ValueError, match="cannot be the model"):
        reg.promote(rec.fit_id, Scope.genus("Dalbergia"), author=WHO, reason="engano de gênero")
    few = reg.run_fit(data.head(2), m, "height.richards", Filter(), author=WHO)
    assert not few.fit.converged and few.cv is None  # kept, so the attempt is on record
    with pytest.raises(ValueError, match="did not converge"):
        reg.promote(few.fit_id, Scope.generic(), author=WHO, reason="não deveria passar")
    reg.promote(rec.fit_id, Scope.species("Khaya grandifoliola"), author=WHO, reason="ajuste da espécie")
    with pytest.raises(ValueError, match="already the champion"):
        reg.promote(rec.fit_id, Scope.species("Khaya grandifoliola"), author=WHO, reason="de novo, por engano")


def test_a_decision_made_on_a_stale_screen_is_refused(setup):
    reg = setup[0]
    scope = Scope.species("Khaya grandifoliola")
    a, b = species_fit(setup), species_fit(setup, form="height.gompertz")
    reg.promote(a.fit_id, scope, author=WHO, reason="primeira campeã", expected_seq=0)
    # Bia's screen still shows no champion (event 0) when she promotes
    with pytest.raises(ConcurrentPromotion):
        reg.promote(b.fit_id, scope, author="bia@courageousland.com", reason="achei melhor", expected_seq=0)
    assert reg.promote(b.fit_id, scope, author="bia@courageousland.com", reason="achei melhor", expected_seq=1).seq == 2


def test_of_two_people_promoting_at_once_only_one_wins(setup):
    reg = setup[0]
    scope = Scope.species("Khaya grandifoliola")
    recs = [species_fit(setup, form=f) for f in ("height.logistic", "height.gompertz", "height.curtis",
                                                 "height.chapman_richards")]
    barrier, outcome = threading.Barrier(len(recs)), []

    def go(rec):
        r = Registry(reg.store)  # each person has their own session
        barrier.wait()
        try:
            r.promote(rec.fit_id, scope, author=WHO, reason="promovido ao mesmo tempo")
            outcome.append(rec.fit_id)
        except ConcurrentPromotion:
            outcome.append(None)

    threads = [threading.Thread(target=go, args=(r,)) for r in recs]
    [t.start() for t in threads]
    [t.join() for t in threads]
    events = reg.history("ht_m", scope)
    assert [e.seq for e in events] == list(range(1, len(events) + 1))  # no number taken twice
    assert len(events) == sum(o is not None for o in outcome) >= 1


def test_the_validity_range_is_the_fitted_data_range_unless_given(setup):
    reg = setup[0]
    rec = species_fit(setup)
    scope = Scope.species("Khaya grandifoliola")
    e = reg.promote(rec.fit_id, scope, author=WHO, reason="faixa padrão do ajuste")
    assert e.valid_range == {"dap_cm": rec.fit.ranges["dap_cm"]}
    champ = reg.champion("ht_m", scope)
    lo, hi = rec.fit.ranges["dap_cm"]
    assert champ.in_range(dap_cm=np.array([lo, hi, lo - 0.1, hi + 0.1])).tolist() == [True, True, False, False]
    other = species_fit(setup, form="height.gompertz")
    e2 = reg.promote(other.fit_id, scope, author=WHO, reason="faixa restrita", valid_range={"dap_cm": (5, 30)})
    assert e2.valid_range == {"dap_cm": [5.0, 30.0]}


def test_the_register_survives_a_new_session(setup):
    """What a person sees comes from the store, not from the memory of the session that wrote it."""
    reg = setup[0]
    rec = species_fit(setup)
    reg.promote(rec.fit_id, Scope.species("Khaya grandifoliola"), author=WHO, reason="ajuste da espécie")
    fresh = Registry(LocalStore(reg.store.root))
    champ = fresh.resolve("ht_m", "Khaya grandifoliola")
    assert champ.fit_id == rec.fit_id
    assert champ.predict(dap_cm=25.0) == pytest.approx(rec.fit.predict(dap_cm=25.0))
