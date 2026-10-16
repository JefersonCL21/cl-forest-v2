"""Pieces the tabs of the Model Lab share: picking a dataset version, mapping a file's columns, and the Portuguese
labels of flags, scopes, filters and statistics."""

from __future__ import annotations

import base64
from collections.abc import Iterable, Mapping

import pandas as pd
import streamlit as st

from cl_allometry import datasets as D
from cl_allometry.registry import Filter, Scope
from cl_allometry.selection import METRICS
from allometry_app import services

KEEP = "— manter como está —"
SOURCE_LABELS = {"measured": "medida", "species": "modelo da espécie", "genus": "modelo do gênero",
                 "generic": "modelo genérico", "none": "sem estimativa"}


def flag_label(col: str) -> str:
    rest = col.removeprefix("flag_")
    if rest in D.FLAG_LABELS:
        return D.FLAG_LABELS[rest]
    for name, c in D.COLUMNS.items():
        if rest.startswith(name + "_") and rest[len(name) + 1:] in D.FLAG_LABELS:
            return f"{c.label}: {D.FLAG_LABELS[rest[len(name) + 1:]]}"
    return rest


def quality_table(counts: Mapping[str, int]) -> pd.DataFrame:
    rows = [{"Alerta": flag_label(k), "Linhas": int(v)} for k, v in counts.items() if v]
    return pd.DataFrame(rows, columns=["Alerta", "Linhas"])


def scope_label(scope: Scope | str) -> str:
    scope = Scope.parse(scope) if isinstance(scope, str) else scope
    if scope.level == "species":
        return scope.name
    if scope.level == "genus":
        return f"gênero {scope.name}"
    return "genérico (as demais espécies)"


def filter_label(flt: Mapping) -> str:
    f = Filter.from_dict(flt)
    parts = []
    if f.especie:
        parts.append(", ".join(f.especie) if len(f.especie) <= 3 else f"{len(f.especie)} espécies")
    if f.genus:
        parts.append(f"gênero {f.genus}")
    if not f.especie and not f.genus:
        parts.append("todas as espécies")
    if f.talhao:
        parts.append("talhões " + ", ".join(f.talhao))
    if f.idade_min is not None or f.idade_max is not None:
        parts.append(f"idade {f.idade_min or 0:g}–{f.idade_max if f.idade_max is not None else '∞'}")
    if not f.exclude_flags:
        parts.append("sem excluir alertas")
    return " · ".join(parts)


def metric_label(name: str) -> str:
    return METRICS[name][2] if name in METRICS else name


def version_label(m: D.Manifest) -> str:
    return f"v{m.version} · {m.created_at[:10]} · {m.rows:,} linhas".replace(",", ".")


def dataset_picker(uri: str, key: str, label: str = "Dados") -> D.Manifest | None:
    """A family and one of its versions (the latest by default); the choice is shared by the tabs."""
    fams = services.families(uri)
    if not fams:
        st.info("Ainda não há dados no laboratório. Importe uma planilha na aba Dados.")
        return None
    c1, c2 = st.columns([1, 2])
    fam = c1.selectbox(f"{label}: família", fams, key=f"{key}_family",
                       index=fams.index(st.session_state.get("lab_family", fams[0]))
                       if st.session_state.get("lab_family") in fams else 0)
    vs = services.versions(uri, fam)
    if not vs:
        return None
    labels = [version_label(m) for m in reversed(vs)]
    chosen = c2.selectbox("Versão", labels, key=f"{key}_version")
    m = list(reversed(vs))[labels.index(chosen)]
    st.session_state["lab_family"] = fam
    return m


def mapping_editor(columns: Iterable[str], key: str) -> dict[str, str]:
    """{column of the file: canonical name}, suggested from the column names and editable."""
    columns = [str(c) for c in columns]
    suggested = D.suggest_mapping(columns)
    labels = {c.label: name for name, c in D.COLUMNS.items()}
    table = pd.DataFrame({"Coluna do arquivo": columns,
                          "Significa": [D.COLUMNS[suggested[c]].label if c in suggested else KEEP for c in columns]})
    edited = st.data_editor(
        table, key=key, hide_index=True, use_container_width=True, disabled=["Coluna do arquivo"],
        column_config={"Significa": st.column_config.SelectboxColumn(options=[KEEP, *labels], required=True)})
    mapping = {row["Coluna do arquivo"]: labels[row["Significa"]] for _, row in edited.iterrows()
               if row["Significa"] in labels}
    repeated = [k for k in set(mapping.values()) if list(mapping.values()).count(k) > 1]
    if repeated:
        st.error("Duas colunas com o mesmo significado: " + ", ".join(D.COLUMNS[r].label for r in repeated))
        return {}
    return mapping


def sidebar_background(path: str = "imagens/image.png") -> None:
    """The sidebar image of the other pages (the CSS of adicionarPlanoFundoSideBar, without that module's imports of
    geopandas, folium, seaborn... that this page does not use)."""
    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode()
    st.markdown(f"""<style>
        [data-testid="stSidebar"] > div:first-child {{
            background: url(data:image/png;base64,{data}) no-repeat center top / cover;
        }}
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] {{ position: relative; z-index: 1; }}
        </style>""", unsafe_allow_html=True)
