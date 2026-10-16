"""Candidates are ranked statistic by statistic in each one's own direction, the weights decide what matters, near
ties go to the simpler model and then to the one in use, and the recommendation says why
(cl_allometry.selection)."""

import pytest

from cl_allometry import selection as S
from cl_allometry.fit import Fit
from cl_allometry.forms import form
from cl_allometry.registry import Filter, FitRecord
from synthetic import WHO, lab

SHA = "a" * 64


def candidate(key, cv_rmse, r2_adj=0.8, aic=100.0, n=200, bias=0.1, p=0.001, bp=0.5, sha=SHA, converged=True,
              especie=("Khaya grandifoliola",)):
    """A fit record with the statistics given (the rest plausible), to rank without fitting."""
    params = form(key).params
    fit = Fit(form=key, target=form(key).target, converged=converged, message="ok" if converged else "não convergiu",
              n=n, dropped=0, coefficients={k: 1.0 for k in params} if converged else None,
              p_value={k: p for k in params} if converged else {},
              metrics={"rmse_pct": cv_rmse - 0.3, "syx_pct": cv_rmse - 0.2, "bias_pct": bias, "r2_adj": r2_adj,
                       "aic": aic, "bic": aic + 5, "furnival": 1.0} if converged else {},
              checks={"breusch_pagan_p": bp, "shapiro_p": 0.4, "cook_over_4_n": 0} if converged else {})
    return FitRecord(fit_id=f"ht_m.{sha[:12]}.{key}", target=fit.target, form=key, fit=fit,
                     cv={"converged": True, "rmse_pct": cv_rmse, "bias_pct": bias} if converged else None,
                     dataset_id="inventario@v1", dataset_sha256=sha, filter=Filter(especie=especie).to_dict(),
                     spec={}, n_filtered=n, author=WHO, commit="abc", created_at="2026-10-16T12:00:00+00:00")


def test_each_statistic_ranks_in_its_own_direction():
    a = candidate("height.logistic", cv_rmse=10, r2_adj=0.90, aic=120, bias=-3)
    b = candidate("height.gompertz", cv_rmse=12, r2_adj=0.95, aic=110, bias=1)
    t = S.rank([a, b]).set_index("form")
    assert t.loc["height.logistic", "rank_cv_rmse_pct"] == 1 and t.loc["height.gompertz", "rank_r2_adj"] == 1
    assert t.loc["height.gompertz", "rank_aic"] == 1
    assert t.loc["height.gompertz", "rank_cv_abs_bias_pct"] == 1  # |−3| is a bigger bias than |1|


def test_the_weights_decide_what_matters():
    a = candidate("height.logistic", cv_rmse=10, r2_adj=0.90)
    b = candidate("height.gompertz", cv_rmse=12, r2_adj=0.95)
    assert S.rank([a, b], {"cv_rmse_pct": 1}).iloc[0]["form"] == "height.logistic"
    assert S.rank([a, b], {"r2_adj": 1}).iloc[0]["form"] == "height.gompertz"
    with pytest.raises(ValueError, match="unknown"):
        S.rank([a, b], {"elegance": 1})


def test_the_recommendation_says_why():
    a = candidate("height.logistic", cv_rmse=10, aic=90)
    b = candidate("height.gompertz", cv_rmse=14, aic=120)
    rec = S.recommend(S.rank([a, b]))
    assert rec.fit_id == a.fit_id and rec.runner_up == b.fit_id and rec.tied == []
    assert "Logístico" in rec.reasons[0] and any("10.00%" in r for r in rec.reasons)


def test_a_near_tie_goes_to_the_simpler_model_then_to_the_one_in_use():
    logistic = candidate("height.logistic", cv_rmse=13.64, aic=100)
    gompertz = candidate("height.gompertz", cv_rmse=13.63, aic=99)
    curtis = candidate("height.curtis", cv_rmse=13.71, aic=101)
    ranked = S.rank([logistic, gompertz, curtis])
    assert ranked.iloc[0]["form"] == "height.gompertz"
    rec = S.recommend(ranked)
    assert rec.fit_id == curtis.fit_id and "menos parâmetros" in rec.reasons[0]
    same_size = S.rank([logistic, gompertz])
    assert S.recommend(same_size).fit_id == gompertz.fit_id
    assert S.recommend(same_size, incumbent=logistic.fit_id).fit_id == logistic.fit_id  # a change should be worth it
    far = S.rank([logistic, candidate("height.curtis", cv_rmse=15.0)])
    assert S.recommend(far).fit_id == logistic.fit_id


def test_candidates_of_different_data_or_rows_are_not_ranked_together():
    a = candidate("height.logistic", cv_rmse=10)
    with pytest.raises(S.MixedCandidates, match="data"):
        S.rank([a, candidate("height.curtis", cv_rmse=11, sha="b" * 64)])
    with pytest.raises(S.MixedCandidates, match="rows"):
        S.rank([a, candidate("height.curtis", cv_rmse=11, especie=("Khaya ivorensis",))])


def test_a_fit_that_did_not_converge_is_listed_but_not_ranked():
    a = candidate("height.logistic", cv_rmse=10)
    failed = candidate("height.richards", cv_rmse=0, converged=False)
    t = S.rank([failed, a])
    assert list(t["form"]) == ["height.logistic", "height.richards"]
    assert t.iloc[1]["position"] != t.iloc[1]["position"]  # NaN: no position
    assert S.recommend(S.rank([failed])).fit_id is None


def test_the_warnings_a_person_should_see_before_promoting():
    small = candidate("height.logistic", cv_rmse=10, n=12, p=0.3, bp=0.01)
    w = S.recommend(S.rank([small])).warnings
    assert any("só 12 árvores" in x for x in w) and any("não significativo" in x for x in w)
    assert any("Breusch-Pagan" in x and "pesos" in x for x in w)


def test_ranking_real_fits_end_to_end(tmp_path):
    reg, data, m = lab(tmp_path)
    flt = Filter(especie=("Khaya ivorensis",))
    recs = [reg.run_fit(data, m, k, flt, author=WHO) for k in ("height.logistic", "height.curtis",
                                                               "height.gompertz", "height.henricksen")]
    ranked = S.rank(recs)
    rec = S.recommend(ranked)
    assert rec.fit_id in {r.fit_id for r in recs} and ranked["position"].notna().all()
    # the data were made with a logistic: the straight line in log(DAP) cannot be the best
    assert ranked.iloc[-1]["form"] in ("height.henricksen", "height.curtis")
