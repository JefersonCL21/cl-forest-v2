"""Login for the Model Lab with Streamlit's own Google login (st.login, st.experimental_user in Streamlit 1.44).
Only writing needs it: who imports, fits or promotes is recorded by e-mail. The rules are in policy.py.

Configuration, in this order: environment variables (ALLOMETRY_STORE, ALLOMETRY_PROMOTERS), then secrets.toml:
    [auth]          redirect_uri, cookie_secret, client_id, client_secret, server_metadata_url
    [allometry]     store = "gs://...", allowed_domain = "courageousland.com", promoters = ["..."]
Without any of it, running locally, the store is the folder .allometry-local and the page writes as dev@local.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from allometry_app.policy import (DEFAULT_DOMAIN, Identity, can_promote, domain_allowed, host_matches, main_address,
                                  promoters, write_mode)

LOCAL_STORE = Path(__file__).resolve().parents[1] / ".allometry-local"


@dataclass(frozen=True)
class Settings:
    store_uri: str | None
    mode: str
    redirect_uri: str | None
    domain: str
    promoters: frozenset[str]


def _secrets() -> dict:
    try:
        return st.secrets.to_dict()
    except Exception:  # no secrets.toml at all: a plain local run
        return {}


def settings() -> Settings:
    secrets = _secrets()
    lab = secrets.get("allometry", {})
    on_cloud_run = bool(os.environ.get("K_SERVICE"))
    store = os.environ.get("ALLOMETRY_STORE") or lab.get("store") or (None if on_cloud_run else str(LOCAL_STORE))
    auth = secrets.get("auth") or {}
    return Settings(store_uri=store, mode=write_mode(bool(auth), store, on_cloud_run),
                    redirect_uri=auth.get("redirect_uri"), domain=lab.get("allowed_domain", DEFAULT_DOMAIN),
                    promoters=promoters(lab.get("promoters"), os.environ.get("ALLOMETRY_PROMOTERS")))


def _claims() -> dict:
    user = st.experimental_user
    return dict(user) if user.get("is_logged_in") else {}


def _host() -> str | None:
    return st.context.headers.get("Host")


def identity(s: Settings) -> Identity | None:
    """The person who may write now, or None."""
    if s.mode == "dev":
        return Identity(os.environ.get("ALLOMETRY_DEV_USER", "dev@local"), "modo local", "dev")
    if s.mode != "google":
        return None
    claims = _claims()
    if not claims or not domain_allowed(claims, s.domain):
        return None
    return Identity(claims["email"].strip().lower(), claims.get("name") or claims["email"], "google")


def account_box(s: Settings, who: Identity | None) -> None:
    with st.sidebar:
        if s.mode == "dev":
            st.caption(f"🧪 Modo local · gravando como **{who.email}** em `{s.store_uri}`")
        elif s.mode == "read_only":
            st.caption("🔒 Somente leitura")
        elif _claims():
            st.caption(f"👤 {_claims().get('email')}")
            if st.button("Sair", key="lab_logout"):
                st.logout()
        elif host_matches(s.redirect_uri, _host()):
            if st.button("Entrar com Google", key="lab_login"):
                st.login()


def writer_gate(s: Settings, who: Identity | None, what: str, key: str) -> bool:
    """True if `who` may write; otherwise says why, with the way to log in, and returns False."""
    if who is not None:
        return True
    if s.mode == "read_only":
        st.info(f"Somente leitura neste ambiente: não é possível {what}.")
    elif not host_matches(s.redirect_uri, _host()):
        st.info(f"Para {what}, use o endereço principal do app: {main_address(s.redirect_uri)}")
    elif _claims():
        st.warning(f"Para {what}, entre com uma conta @{s.domain} (você entrou como {_claims().get('email')}).")
        if st.button("Sair e trocar de conta", key=f"{key}_switch"):
            st.logout()
    else:
        st.info(f"Para {what}, entre com sua conta Google @{s.domain}.")
        if st.button("Entrar com Google", key=f"{key}_login", type="primary"):
            st.login()
    return False


def may_promote(s: Settings, who: Identity | None) -> bool:
    return can_promote(who, s.promoters)
