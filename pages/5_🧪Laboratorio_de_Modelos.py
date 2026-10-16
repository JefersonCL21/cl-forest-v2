"""Laboratório de Modelos: import field data as versions, fit allometric models, compare them, promote the one to use,
and apply it, with everything recorded (cl_allometry). Reading and applying are open; writing needs a Google login
of the company (allometry_app.auth)."""

import json
from itertools import product

import pandas as pd
import streamlit as st

import adicionarLogo
from allometry_app import auth, charts, services, widgets
from cl_allometry import datasets as D
from cl_allometry import selection as S
from cl_allometry.apply import apply
from cl_allometry.export import model_card, models_table, to_sql
from cl_allometry.forms import TARGETS, form as form_by_key, forms_for
from cl_allometry.registry import ConcurrentPromotion, Filter, Scope, genus_of

st.set_page_config(page_title="CL Forest Biometrics", page_icon=":seedling:", layout="wide")
adicionarLogo.add_logo()
widgets.sidebar_background()
st.markdown("<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}</style>",
            unsafe_allow_html=True)

CFG = auth.settings()
WHO = auth.identity(CFG)
auth.account_box(CFG, WHO)
if not CFG.store_uri:
    st.error("O armazenamento dos modelos não está configurado (ALLOMETRY_STORE).")
    st.stop()
URI = CFG.store_uri


@st.cache_data(max_entries=2, show_spinner="Lendo a planilha…")
def read_file(name: str, data: bytes, sheet: str | None) -> pd.DataFrame:
    return D.read_upload(name, data, sheet)


@st.cache_data(max_entries=2, show_spinner="Conferindo os dados…")
def prepare(raw: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, str]:
    return D.prepare(raw, mapping)


def upload(key: str) -> tuple[str, bytes, pd.DataFrame] | None:
    up = st.file_uploader("Planilha (csv ou xlsx)", type=["csv", "txt", "xlsx", "xls"], key=f"{key}_file")
    if not up:
        return None
    data = up.getvalue()
    sheet = None
    if up.name.lower().endswith((".xlsx", ".xls")):
        names = D.sheets(data)
        sheet = st.selectbox("Aba da planilha", names, key=f"{key}_sheet") if len(names) > 1 else None
    return up.name, data, read_file(up.name, data, sheet)


def resolver(champs, target):
    """The champion of a tree's species, else its genus', else the generic one (from the champions already read)."""
    table = {str(c.scope): c for c in champs if c.target == target}
    return lambda sp: next((table[str(s)] for s in Scope.chain(sp) if str(s) in table), None)


def fmt_coefs(coefs: dict | None) -> str:
    return "  ".join(f"{k} = {v:.6g}" for k, v in (coefs or {}).items())


def fmt_range(rng: dict | None) -> str:
    return "; ".join(f"{charts.LABELS.get(k, k)} {lo:.3g}–{hi:.3g}" for k, (lo, hi) in (rng or {}).items())


# ── Dados ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def tab_data() -> None:
    m = widgets.dataset_picker(URI, "data")
    if m is not None:
        vs = services.versions(URI, m.family)
        st.dataframe(pd.DataFrame([{
            "Versão": f"v{v.version}", "Data": v.created_at[:16].replace("T", " "), "Autor": v.author,
            "Linhas": v.rows, "Alertas": sum(v.quality.values()), "sha256": v.sha256[:12], "Arquivo": v.source_name,
            "Nota": v.note} for v in reversed(vs)]), hide_index=True, use_container_width=True)
        df, _ = services.dataset(URI, m.dataset_id)
        flagged = {k: v for k, v in m.quality.items() if v}
        with st.expander(f"Alertas de qualidade de {m.dataset_id}"):
            if not flagged:
                st.caption("Nenhum alerta.")
            else:
                st.dataframe(widgets.quality_table(flagged), hide_index=True)
                flag = st.selectbox("Ver as linhas com", list(flagged), format_func=widgets.flag_label,
                                    key="data_flag")
                st.dataframe(df[df[flag]].drop(columns=[c for c in df if c.startswith("flag_")]), hide_index=True)
        if len(vs) > 1:
            with st.expander("Comparar duas versões"):
                labels = [f"v{v.version}" for v in vs]
                c1, c2 = st.columns(2)
                a = c1.selectbox("De", labels, index=len(labels) - 2, key="data_diff_a")
                b = c2.selectbox("Para", labels, index=len(labels) - 1, key="data_diff_b")
                if a != b:
                    d = D.diff(services.dataset(URI, f"{m.family}@{a}")[0], services.dataset(URI, f"{m.family}@{b}")[0])
                    cols = st.columns(4)
                    for col, (label, value) in zip(cols, [("Novas", d.added), ("Removidas", d.removed),
                                                          ("Alteradas", "—" if d.changed is None else d.changed),
                                                          ("Iguais", d.unchanged)]):
                        col.metric(label, value)
                    if d.columns_added or d.columns_removed:
                        st.caption(f"Colunas novas: {', '.join(d.columns_added) or '—'} · "
                                   f"removidas: {', '.join(d.columns_removed) or '—'}")
                    if not d.examples.empty:
                        st.dataframe(d.examples.drop(columns=[c for c in d.examples if c.startswith("flag_")]),
                                     hide_index=True)

    with st.expander("Importar nova versão", expanded=not services.families(URI)):
        if auth.writer_gate(CFG, WHO, "importar dados", key="import"):
            import_form()


def import_form() -> None:
    got = upload("import")
    if got is None:
        return
    name, data, raw = got
    fams = services.families(URI)
    c1, c2 = st.columns(2)
    new = "nova família…"
    choice = c1.selectbox("Família", [*fams, new], key="import_family")
    family = c2.text_input("Nome da família", placeholder="ex.: cubagem-sucupira", key="import_new_family") \
        if choice == new else choice
    st.caption("Uma família reúne as versões de um mesmo conjunto de dados: o censo, uma cubagem, uma coleta.")
    st.markdown("**O que é cada coluna**")
    mapping = widgets.mapping_editor(raw.columns, key=f"import_map_{name}")
    if not mapping:
        st.warning("Diga o que é cada coluna: ao menos a espécie e o DAP (ou o CAP).")
        return
    try:
        prepared, sha = prepare(raw, mapping)
    except ValueError as e:
        st.error(str(e))
        return
    flags = {c: int(prepared[c].sum()) for c in prepared if c.startswith("flag_")}
    c1, c2, c3 = st.columns(3)
    c1.metric("Linhas", f"{len(prepared):,}".replace(",", "."))
    c2.metric("Linhas com alerta", int(prepared[[c for c in flags]].any(axis=1).sum()) if flags else 0)
    c3.metric("sha256", sha[:12])
    if any(flags.values()):
        st.dataframe(widgets.quality_table(flags), hide_index=True)
        st.caption("As linhas com alerta são guardadas assim mesmo, marcadas; cada ajuste escolhe o que deixar de fora.")
    latest = services.versions(URI, family)[-1] if family in fams and services.versions(URI, family) else None
    if latest is not None:
        if latest.sha256 == sha:
            st.info(f"Esses dados já estão em {latest.dataset_id}: nada a gravar.")
            return
        d = D.diff(services.dataset(URI, latest.dataset_id)[0], prepared)
        st.caption(f"Em relação a {latest.dataset_id}: {d.added} novas, {d.removed} removidas"
                   + (f", {d.changed} alteradas" if d.changed is not None else "") + f", {d.unchanged} iguais.")
    with st.expander("Prévia"):
        st.dataframe(prepared.drop(columns=list(flags)).head(50), hide_index=True)
    note = st.text_input("Nota da versão", placeholder="o que mudou e de onde vieram os dados", key="import_note")
    if st.button("Gravar versão", type="primary", disabled=not family, key="import_go"):
        try:
            r = D.import_dataset(services.registry(URI).store, raw, family=family, mapping=mapping, author=WHO.email,
                                 source_name=name, source_bytes=data, note=note)
        except ValueError as e:
            st.error(str(e))
            return
        services.written()
        st.session_state["lab_family"] = family
        if r.created_version:
            st.success(f"Gravado como {r.manifest.dataset_id} (sha256 {r.manifest.sha256[:12]}).")
        else:
            st.info(f"Esses dados já estavam em {r.manifest.dataset_id}.")


# ── Ajustar ────────────────────────────────────────────────────────────────────────────────────────────────────────
def tab_fit() -> None:
    if not auth.writer_gate(CFG, WHO, "ajustar modelos", key="fit"):
        return
    m = widgets.dataset_picker(URI, "fit")
    if m is None:
        return
    df, _ = services.dataset(URI, m.dataset_id)
    targets = [t for t in TARGETS if t in df and any(set(f.variables) <= set(df) for f in forms_for(t))]
    if not targets or "especie" not in df:
        st.warning("Estes dados não têm espécie e um alvo para ajustar (altura, volume, biomassa ou DAP pela copa).")
        return
    c1, c2 = st.columns([1, 2])
    target = c1.selectbox("Alvo", targets, format_func=TARGETS.get, key="fit_target")
    forms = [f for f in forms_for(target) if set(f.variables) <= set(df)]
    cols = [target, *forms[0].variables]
    ok = (df[cols] > 0).all(axis=1) & ~df[[c for c in D.DEFAULT_EXCLUDE if c in df]].any(axis=1)
    counts = df[ok].groupby("especie").size().sort_values(ascending=False)
    chosen = c2.multiselect("Espécies", list(counts.index), format_func=lambda s: f"{s} ({counts[s]})",
                            key="fit_species", placeholder="escolha as espécies (entre parênteses, as árvores com "
                                                           f"{TARGETS[target].split(' (')[0]})")
    grouping = st.radio("Ajustar", ["um modelo por espécie", "um modelo por gênero", "um modelo para todas juntas"],
                        horizontal=True, key="fit_grouping")
    keys = st.multiselect("Formas", [f.key for f in forms], default=[f.key for f in forms],
                          format_func=lambda k: form_by_key(k).name, key="fit_forms")
    flags = [c for c in df if c.startswith("flag_")]
    with st.expander("Opções"):
        o1, o2, o3, o4 = st.columns(4)
        weights = o1.selectbox("Pesos", [None, "1/dap_cm", "1/dap_cm**2"], key="fit_weights",
                               format_func=lambda w: "sem pesos" if w is None else w) \
            if all("dap_cm" in form_by_key(k).variables for k in keys) else None
        k = o2.number_input("Partições da validação cruzada", 2, 10, 5, key="fit_k")
        seed = o3.number_input("Semente", 0, 1_000_000, 0, key="fit_seed")
        starts = o4.number_input("Pontos de partida", 1, 32, 8, key="fit_starts")
        exclude = st.multiselect("Deixar de fora as linhas com", flags, format_func=widgets.flag_label,
                                 default=[f for f in D.DEFAULT_EXCLUDE if f in flags], key="fit_exclude")
        talhoes = st.multiselect("Só os talhões", sorted(df["talhao"].dropna().unique()), key="fit_talhoes") \
            if "talhao" in df else []
        idade = None
        if "idade" in df and df["idade"].notna().any():
            lo, hi = float(df["idade"].min()), float(df["idade"].max())
            if hi > lo:
                idade = st.slider("Idade (anos)", lo, hi, (lo, hi), key="fit_idade")

    common = dict(talhao=tuple(talhoes) or None, exclude_flags=tuple(exclude),
                  idade_min=idade[0] if idade and idade[0] > df["idade"].min() else None,
                  idade_max=idade[1] if idade and idade[1] < df["idade"].max() else None)
    if grouping.endswith("por espécie"):
        filters = [Filter(especie=(sp,), **common) for sp in chosen]
    elif grouping.endswith("por gênero"):
        filters = [Filter(genus=g, **common) for g in sorted({genus_of(sp) for sp in chosen})]
    else:
        filters = [Filter(especie=tuple(sorted(chosen)) or None, **common)]
    n = len(filters) * len(keys) if (chosen or grouping.endswith("juntas")) else 0
    if st.button(f"Ajustar {n} modelo(s)", type="primary", disabled=n == 0, key="fit_go"):
        reg = services.registry(URI)
        bar = st.progress(0.0)
        done = []
        for i, (flt, key) in enumerate(product(filters, keys)):
            bar.progress(i / n, text=f"{form_by_key(key).name} · {widgets.filter_label(flt.to_dict())}")
            done.append(reg.run_fit(df, m, key, flt, author=WHO.email, weights=weights, seed=int(seed),
                                    starts=int(starts), cv_k=int(k)).fit_id)
        bar.empty()
        services.written()
        st.session_state["lab_last_fits"] = done
    last = st.session_state.get("lab_last_fits")
    if last:
        reg = services.registry(URI)
        recs = [reg.get_fit(f) for f in last]
        st.dataframe(pd.DataFrame([{
            "Dados": widgets.filter_label(r.filter), "Forma": form_by_key(r.form).name, "n": r.fit.n,
            "RMSE%": r.fit.metrics.get("rmse_pct"), "RMSE% (VC)": (r.cv or {}).get("rmse_pct"),
            "Convergiu": "sim" if r.fit.converged else r.fit.message} for r in recs]),
            hide_index=True, use_container_width=True,
            column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ("RMSE%", "RMSE% (VC)")})
        st.caption("Os ajustes ficam guardados. Compare-os e promova o melhor na aba Comparar.")


# ── Comparar ───────────────────────────────────────────────────────────────────────────────────────────────────────
def tab_compare() -> None:
    m = widgets.dataset_picker(URI, "cmp")
    if m is None:
        return
    targets = [t for t in TARGETS if services.fits(URI, t, m.sha256)]
    if not targets:
        st.info("Ainda não há ajustes nestes dados. Ajuste na aba Ajustar.")
        return
    c1, c2 = st.columns([1, 2])
    target = c1.selectbox("Alvo", targets, format_func=TARGETS.get, key="cmp_target")
    recs = services.fits(URI, target, m.sha256)
    groups: dict[str, list] = {}
    for r in recs:
        groups.setdefault(json.dumps(r.filter, sort_keys=True), []).append(r)
    last = set(st.session_state.get("lab_last_fits", []))
    keys = sorted(groups, key=lambda g: (not any(r.fit_id in last for r in groups[g]), widgets.filter_label(
        json.loads(g))))
    g = c2.selectbox("Ajustes de", keys, key="cmp_group",
                     format_func=lambda g: f"{widgets.filter_label(json.loads(g))} ({len(groups[g])} ajustes)")
    cands = groups[g]
    flt = Filter.from_dict(cands[0].filter)
    scope = flt.suggested_scope()
    champ = next((c for c in services.champions(URI, target) if c.scope == scope), None)
    incumbent = champ.fit_id if champ else None

    with st.expander("Pesos do ranking"):
        cols = st.columns(len(S.METRICS))
        weights = {k: col.number_input(widgets.metric_label(k), 0, 5, S.DEFAULT_WEIGHTS.get(k, 0), key=f"cmp_w_{k}")
                   for col, k in zip(cols, S.METRICS)}
    ranked = S.rank(cands, weights)
    rec = S.recommend(ranked, incumbent=incumbent)
    view = pd.DataFrame({
        "Posição": ranked["position"], "Forma": ranked["name"],
        "Em uso": ["★" if f == incumbent else "" for f in ranked["fit_id"]],
        "RMSE% (VC)": ranked["cv_rmse_pct"], "|Viés%| (VC)": ranked["cv_abs_bias_pct"], "R² aj.": ranked["r2_adj"],
        "AIC": ranked["aic"], "n": ranked["n"], "Parâmetros": ranked["params"],
        "Avisos": [len(S.warnings_for(row)) if row["converged"] else "não convergiu" for _, row in ranked.iterrows()],
    })
    st.dataframe(view, hide_index=True, use_container_width=True, column_config={
        "Posição": st.column_config.NumberColumn(format="%d"),
        **{c: st.column_config.NumberColumn(format="%.2f") for c in ("RMSE% (VC)", "|Viés%| (VC)", "R² aj.")},
        "AIC": st.column_config.NumberColumn(format="%.1f")})
    by_id = {r.fit_id: r for r in cands}
    if rec.fit_id:
        st.success(f"**Recomendado: {form_by_key(by_id[rec.fit_id].form).name}.** " + "; ".join(rec.reasons) + ".")
        for w in rec.warnings:
            st.caption(f"⚠️ {w}")
    st.caption(f"Em uso para {widgets.scope_label(scope)}: "
               + (f"{form_by_key(champ.record.form).name} (desde {champ.event.created_at[:10]})" if champ
                  else "nenhum modelo"))

    converged = [r.fit_id for r in cands if r.fit.converged]
    default = [f for f in dict.fromkeys([rec.fit_id, incumbent, rec.runner_up]) if f in converged][:3]
    shown = st.multiselect("No gráfico", converged, default=default, max_selections=4, key=f"cmp_shown_{g}",
                           format_func=lambda f: form_by_key(by_id[f].form).name)
    rows = flt.apply(services.dataset(URI, m.dataset_id)[0])
    if shown:
        st.plotly_chart(charts.curves(rows, [by_id[f] for f in shown]), use_container_width=True)

    detail = st.selectbox("Detalhes do ajuste", [r.fit_id for r in cands], key=f"cmp_detail_{g}",
                          index=[r.fit_id for r in cands].index(rec.fit_id) if rec.fit_id else 0,
                          format_func=lambda f: form_by_key(by_id[f].form).name)
    fit_details(by_id[detail], rows)

    if WHO is None:
        st.caption("Para promover um modelo, entre com sua conta (aba Dados ou barra lateral).")
    elif not auth.may_promote(CFG, WHO):
        st.caption("Só quem está na lista de promotores pode promover um modelo.")
    elif converged and st.button("Promover um modelo…", type="primary", key="cmp_promote"):
        promote_dialog(cands, ranked, rec.fit_id or converged[0], scope, target)


def fit_details(r, rows: pd.DataFrame) -> None:
    f = form_by_key(r.form)
    st.markdown(f"`{f.text}`" + (f" · ajustada em log, com correção {r.fit.correction:.4f}" if f.log_linear else ""))
    if not r.fit.converged:
        st.warning(r.fit.message)
        return
    c1, c2 = st.columns([3, 2])
    c1.dataframe(pd.DataFrame([{
        "Parâmetro": k, "Estimativa": v, "Erro-padrão": r.fit.stderr.get(k), "IC 95% inf.": r.fit.ci95[k][0],
        "IC 95% sup.": r.fit.ci95[k][1], "t": r.fit.t.get(k), "p": r.fit.p_value.get(k)}
        for k, v in r.fit.coefficients.items()]), hide_index=True, use_container_width=True,
        column_config={c: st.column_config.NumberColumn(format="%.6g") for c in
                       ("Estimativa", "Erro-padrão", "IC 95% inf.", "IC 95% sup.")}
        | {"t": st.column_config.NumberColumn(format="%.2f"), "p": st.column_config.NumberColumn(format="%.4f")})
    names = {"r2": "R²", "r2_adj": "R² ajustado", "rmse": "RMSE", "rmse_pct": "RMSE%", "syx": "Syx", "syx_pct": "Syx%",
             "bias": "Viés", "bias_pct": "Viés%", "mae": "MAE", "aic": "AIC", "bic": "BIC", "furnival": "Furnival"}
    c2.dataframe(pd.DataFrame({"Estatística": list(names.values()),
                               "Ajuste": [r.fit.metrics.get(k) for k in names],
                               "Validação cruzada": [(r.cv or {}).get(k) for k in names]}),
                 hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.4g") for c in ("Ajuste", "Validação cruzada")})
    st.plotly_chart(charts.residuals(rows, r), use_container_width=True)
    ch = r.fit.checks
    st.caption(f"Shapiro-Wilk p = {ch.get('shapiro_p', float('nan')):.3f} · Breusch-Pagan p = "
               f"{ch.get('breusch_pagan_p', float('nan')):.3f} · Cook máx. = {ch.get('cook_max', float('nan')):.3f} "
               f"({ch.get('cook_over_4_n', 0)} acima de 4/n) · n = {r.fit.n} ({r.fit.dropped} sem dados)")
    with st.expander("Origem"):
        st.caption(f"Ajuste `{r.fit_id}` · dados {r.dataset_id} (sha256 {r.dataset_sha256[:12]}) · "
                   f"{widgets.filter_label(r.filter)} · pesos {r.fit.weights or 'nenhum'} · semente {r.fit.seed} · "
                   f"{r.fit.starts} pontos de partida · por {r.author} em {r.created_at[:16].replace('T', ' ')} · "
                   f"código {r.commit}")


@st.dialog("Promover modelo", width="large")
def promote_dialog(cands, ranked, default: str, scope: Scope, target: str) -> None:
    reg = services.registry(URI)
    converged = [r for r in cands if r.fit.converged]
    ids = [r.fit_id for r in converged]
    by_id = {r.fit_id: r for r in converged}
    choice = st.selectbox("Modelo", ids, index=ids.index(default) if default in ids else 0,
                          format_func=lambda f: f"{form_by_key(by_id[f].form).name} · RMSE% (VC) "
                                                f"{(by_id[f].cv or {}).get('rmse_pct', float('nan')):.2f}")
    scopes = [scope] + ([Scope.generic()] if scope.level != "generic" else [])
    sc = st.radio("Para", scopes, format_func=widgets.scope_label, horizontal=True)
    r = by_id[choice]
    history = reg.history(target, sc)  # read now, not from the cache: the decision is made on the latest state
    current = history[-1] if history else None
    if current and current.action == "promote":
        st.caption(f"Em uso agora: {form_by_key(reg.get_fit(current.fit_id).form).name}, promovido por "
                   f"{current.author} em {current.created_at[:10]}.")
    rng = {}
    for v, (lo, hi) in r.valid_range().items():
        c1, c2 = st.columns(2)
        rng[v] = (c1.number_input(f"{charts.LABELS.get(v, v)} mínimo", value=float(lo), key=f"rng_lo_{v}"),
                  c2.number_input(f"{charts.LABELS.get(v, v)} máximo", value=float(hi), key=f"rng_hi_{v}"))
    st.caption("Fora dessa faixa a estimativa sai marcada. Por padrão, é a faixa dos dados do ajuste.")
    reason = st.text_area("Justificativa", placeholder="ex.: menor RMSE% na validação cruzada e resíduos sem tendência")
    row = ranked[ranked["fit_id"] == choice].iloc[0]
    for w in S.warnings_for(row):
        st.caption(f"⚠️ {w}")
    if st.button("Promover", type="primary", disabled=len(reason.strip()) < 10):
        try:
            reg.promote(choice, sc, author=WHO.email, reason=reason, valid_range=rng,
                        expected_seq=current.seq if current else 0, warnings=S.warnings_for(row))
        except ConcurrentPromotion:
            services.written()
            st.error("Alguém mudou o modelo em uso enquanto você decidia. Feche e confira de novo.")
            return
        except ValueError as e:
            st.error(str(e))
            return
        services.written()
        st.rerun()


# ── Modelos ────────────────────────────────────────────────────────────────────────────────────────────────────────
def tab_models() -> None:
    champs = services.champions(URI)
    if champs:
        st.dataframe(pd.DataFrame([{
            "Alvo": TARGETS[c.target], "Para": widgets.scope_label(c.scope), "Forma": form_by_key(c.record.form).name,
            "Coeficientes": fmt_coefs(c.record.fit.coefficients), "n": c.record.fit.n,
            "RMSE%": c.record.fit.metrics.get("rmse_pct"), "RMSE% (VC)": (c.record.cv or {}).get("rmse_pct"),
            "Faixa": fmt_range(c.event.valid_range), "Dados": c.record.dataset_id, "Promovido por": c.event.author,
            "Em": c.event.created_at[:10], "Justificativa": c.event.reason} for c in champs]),
            hide_index=True, use_container_width=True,
            column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ("RMSE%", "RMSE% (VC)")})
    else:
        st.info("Nenhum modelo promovido ainda.")

    scopes = sorted({(c.target, str(c.scope)) for c in champs})
    if champs:
        labels = {f"{TARGETS[t]} · {widgets.scope_label(s)}": (t, s) for t, s in scopes}
        target, scope = labels[st.selectbox("Modelo", list(labels), key="mod_pick")]
        champ = next(c for c in champs if c.target == target and str(c.scope) == scope)
        model_panel(champ)

    with st.expander("Todos os ajustes guardados"):
        t = st.selectbox("Alvo", list(TARGETS), format_func=TARGETS.get, key="mod_all_target")
        table = models_table(services.fits(URI, t), [c for c in champs if c.target == t])
        if table.empty:
            st.caption("Nenhum ajuste.")
        else:
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.download_button("Baixar CSV", table.to_csv(index=False).encode("utf-8-sig"), f"modelos_{t}.csv",
                               "text/csv", key="mod_all_csv")


def model_panel(champ) -> None:
    rec = champ.record
    c1, c2 = st.columns([1, 1])
    card = model_card(rec, champ)
    c1.download_button("Baixar o model card (JSON)", json.dumps(card, ensure_ascii=False, indent=1),
                       f"{rec.fit_id}.json", "application/json", key="mod_card")
    with c2.popover("SQL para outros bancos"):
        st.caption("BigQuery")
        st.code(to_sql(rec.form, rec.fit.coefficients, correction=rec.fit.correction), language="sql")
        st.caption("Postgres")
        st.code(to_sql(rec.form, rec.fit.coefficients, correction=rec.fit.correction, dialect="postgres"),
                language="sql")
    history = services.history(URI, champ.target, str(champ.scope))
    reg = services.registry(URI)
    st.dataframe(pd.DataFrame([{
        "Evento": e.seq, "Ação": "promoção" if e.action == "promote" else "aposentadoria",
        "Forma": form_by_key(reg.get_fit(e.fit_id).form).name if e.fit_id else "—", "Ajuste": e.fit_id or "—",
        "Por": e.author, "Em": e.created_at[:16].replace("T", " "), "Justificativa": e.reason,
        "Avisos": "; ".join(e.warnings)} for e in reversed(history)]), hide_index=True, use_container_width=True)
    if not auth.may_promote(CFG, WHO):
        return
    with st.expander("Voltar a um modelo anterior ou aposentar"):
        earlier = list(dict.fromkeys(e.fit_id for e in reversed(history) if e.fit_id and e.fit_id != champ.fit_id))
        reason = st.text_input("Justificativa", key="mod_reason")
        last_seq = history[-1].seq if history else 0
        c1, c2 = st.columns(2)
        if earlier:
            back = c1.selectbox("Voltar para", earlier, key="mod_back",
                                format_func=lambda f: f"{form_by_key(reg.get_fit(f).form).name} · {f}")
            if c1.button("Voltar para este", disabled=len(reason.strip()) < 10, key="mod_back_go"):
                act(lambda: reg.promote(back, champ.scope, author=WHO.email, reason=reason, expected_seq=last_seq))
        if c2.button("Aposentar o modelo em uso", disabled=len(reason.strip()) < 10, key="mod_retire"):
            act(lambda: reg.retire(champ.target, champ.scope, author=WHO.email, reason=reason,
                                   expected_seq=last_seq))
        st.caption("Aposentado, as árvores desta espécie passam a usar o modelo do gênero ou o genérico.")


def act(write) -> None:
    try:
        write()
    except ConcurrentPromotion:
        st.error("Alguém mudou o modelo em uso enquanto você decidia. Confira de novo.")
    except ValueError as e:
        st.error(str(e))
    else:
        services.written()
        st.rerun()


# ── Aplicar ────────────────────────────────────────────────────────────────────────────────────────────────────────
def tab_apply() -> None:
    champs = services.champions(URI)
    if not champs:
        st.info("Nenhum modelo promovido ainda.")
        return
    targets = [t for t in TARGETS if any(c.target == t for c in champs)]
    with st.expander("Uma árvore", expanded=True):
        c1, c2 = st.columns([1, 2])
        target = c1.selectbox("Alvo", targets, format_func=TARGETS.get, key="one_target")
        known = sorted({c.scope.name for c in champs if c.target == target and c.scope.level == "species"})
        sp = c2.selectbox("Espécie", [*known, "outra…"], key="one_species")
        if sp == "outra…":
            sp = c2.text_input("Nome científico", key="one_species_text")
        champ = resolver(champs, target)(sp)
        if champ is None:
            st.warning("Não há modelo para essa espécie, nem do gênero, nem genérico.")
        else:
            f = form_by_key(champ.record.form)
            rng = champ.event.valid_range or {}
            cols = st.columns(len(f.variables) + 1)
            x = {v: col.number_input(charts.LABELS.get(v, v), min_value=0.0, key=f"one_{v}",
                                     value=float(round(sum(rng.get(v, [10.0, 30.0])) / 2, 1)))
                 for col, v in zip(cols, f.variables)}
            if all(val > 0 for val in x.values()):
                value = float(champ.predict(**{k: [v] for k, v in x.items()})[0])
                inside = bool(champ.in_range(**{k: [v] for k, v in x.items()})[0])
                cols[-1].metric(TARGETS[target], f"{value:.2f}")
                st.caption(f"{f.name} · {widgets.SOURCE_LABELS[champ.scope.level]} "
                           f"({widgets.scope_label(champ.scope)}) · ajuste `{champ.fit_id}`"
                           + ("" if inside else f" · ⚠️ fora da faixa ajustada ({fmt_range(champ.event.valid_range)})"))

    with st.expander("Uma planilha", expanded=True):
        source = st.radio("Dados", ["enviar uma planilha", "uma versão registrada"], horizontal=True, key="many_src")
        if source.startswith("enviar"):
            got = upload("apply")
            if got is None:
                return
            name, _, raw = got
            mapping = widgets.mapping_editor(raw.columns, key=f"apply_map_{name}")
            if not mapping:
                st.warning("Diga o que é cada coluna: ao menos a espécie e o DAP (ou o CAP).")
                return
            try:
                df, invalid = D.canonicalize(raw, mapping)
            except ValueError as e:
                st.error(str(e))
                return
            # the same quality rules as an import: an implausible measured value is not kept over the estimate
            df = pd.concat([df, D.quality_flags(df, invalid)], axis=1)
            base = name.rsplit(".", 1)[0]
        else:
            m = widgets.dataset_picker(URI, "apply")
            if m is None:
                return
            df, base = services.dataset(URI, m.dataset_id)[0], m.dataset_id.replace("@", "_")
        chosen = st.multiselect("Estimar", targets, default=targets[:1], format_func=TARGETS.get, key="many_targets")
        if not chosen or not st.button("Aplicar", type="primary", key="many_go"):
            return
        out = df
        try:
            for t in sorted(chosen, key=lambda t: t != "ht_m"):  # heights first: volume and biomass use them
                inputs = {"ht_m": "ht_m_final"} if t != "ht_m" and "ht_m_final" in out else None
                flag = f"flag_{t}_implausible"
                out = apply_target(out, t, champs, inputs, ~out[flag] if flag in out else None)
        except ValueError as e:
            st.error(str(e))
            return
        for t in chosen:
            counts = out[f"{t}_source"].map(widgets.SOURCE_LABELS).value_counts()
            st.markdown(f"**{TARGETS[t]}**")
            st.plotly_chart(charts.sources(counts), use_container_width=True)
            n_out = int(out[f"{t}_out_of_range"].sum())
            if n_out:
                st.caption(f"⚠️ {n_out} estimativa(s) fora da faixa em que o modelo foi ajustado "
                           f"(coluna {t}_out_of_range).")
        st.download_button("Baixar com as estimativas (CSV)",
                           out.drop(columns=[c for c in out if c.startswith("flag_")]).to_csv(index=False)
                           .encode("utf-8-sig"), f"{base}_estimado.csv", "text/csv", key="many_csv")


def apply_target(df, target, champs, inputs, measured_ok):
    return apply(df, target, resolver(champs, target), inputs=inputs, measured_ok=measured_ok)


tabs = st.tabs(["Dados", "Ajustar", "Comparar", "Modelos", "Aplicar"])
for tab, render in zip(tabs, (tab_data, tab_fit, tab_compare, tab_models, tab_apply)):
    with tab:
        render()
