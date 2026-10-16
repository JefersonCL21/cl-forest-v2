"""The register as the page reads it, cached. Records never change, so a dataset or a fit read once is kept; the
listings (versions, fits, champions, histories) can change when anyone writes, so they live 60 seconds here and
are cleared at once after a write in this session (another instance of the app sees it within a minute).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from cl_allometry import datasets as D
from cl_allometry.registry import Champion, Event, FitRecord, Registry, Scope
from cl_allometry.store import open_store


@st.cache_resource(show_spinner=False)
def registry(uri: str) -> Registry:
    return Registry(open_store(uri))


@st.cache_data(ttl=60, show_spinner=False)
def families(uri: str) -> list[str]:
    return D.families(registry(uri).store)


@st.cache_data(ttl=60, show_spinner=False)
def versions(uri: str, family: str) -> list[D.Manifest]:
    return D.versions(registry(uri).store, family)


@st.cache_data(max_entries=6, show_spinner="Lendo os dados…")
def dataset(uri: str, dataset_id: str) -> tuple[pd.DataFrame, D.Manifest]:
    return D.load(registry(uri).store, dataset_id)


@st.cache_data(ttl=60, show_spinner=False)
def fits(uri: str, target: str, dataset_sha256: str | None = None) -> list[FitRecord]:
    return registry(uri).list_fits(target, dataset_sha256)


@st.cache_data(ttl=60, show_spinner=False)
def champions(uri: str, target: str | None = None) -> list[Champion]:
    return registry(uri).champions(target)


@st.cache_data(ttl=60, show_spinner=False)
def history(uri: str, target: str, scope: str) -> list[Event]:
    return registry(uri).history(target, Scope.parse(scope))


def written() -> None:
    """After a write: the listings are read again."""
    for f in (families, versions, fits, champions, history):
        f.clear()
