"""Who may write in the Model Lab, decided without Streamlit so it can be tested: reading and applying models is
open; importing data and fitting need a verified Google account of the company's domain; promoting needs, besides,
to be in the list of promoters.

Modes of the page:
- google: [auth] is configured; a person logs in with Google;
- dev: running locally, with no auth and a local store; the author is ALLOMETRY_DEV_USER (default dev@local);
- read_only: anything else (on Cloud Run without auth, or a bucket without auth): nobody writes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from urllib.parse import urlparse

DEFAULT_DOMAIN = "courageousland.com"


@dataclass(frozen=True)
class Identity:
    email: str
    name: str
    mode: str  # "google" or "dev"


def domain_allowed(claims: Mapping, domain: str = DEFAULT_DOMAIN) -> bool:
    """A verified e-mail of the domain, and the hosted-domain claim Google only sets for Workspace accounts (an
    address that merely ends in the domain is not enough)."""
    email = str(claims.get("email") or "").strip().lower()
    domain = domain.strip().lower()
    return (claims.get("email_verified") is True and email.endswith("@" + domain)
            and str(claims.get("hd") or "").strip().lower() == domain)


def promoters(listed: Iterable[str] | None = None, env: str | None = None) -> frozenset[str]:
    """The e-mails that may promote: those in the secrets and those in ALLOMETRY_PROMOTERS (comma separated)."""
    emails = list(listed or []) + [e for e in (env or "").split(",")]
    return frozenset(e.strip().lower() for e in emails if e and e.strip())


def can_promote(identity: Identity | None, allowed: frozenset[str]) -> bool:
    if identity is None:
        return False
    return identity.mode == "dev" or identity.email.strip().lower() in allowed


def write_mode(auth_configured: bool, store_uri: str | None, on_cloud_run: bool) -> str:
    if not store_uri:
        return "read_only"
    if auth_configured:
        return "google"
    if not on_cloud_run and not store_uri.startswith("gs://"):
        return "dev"
    return "read_only"


def host_matches(redirect_uri: str | None, host: str | None) -> bool:
    """Login only works on the address of redirect_uri (Streamlit ties its cookie to that origin): on any other
    address of the same app (the tagged test revision, the other *.run.app name) the page is read-only."""
    if not redirect_uri or not host:
        return False
    return urlparse(redirect_uri).netloc.lower() == host.strip().lower()


def main_address(redirect_uri: str | None) -> str | None:
    if not redirect_uri:
        return None
    u = urlparse(redirect_uri)
    return f"{u.scheme}://{u.netloc}"
