"""Who may write in the Model Lab: only verified accounts of the company's Workspace import and fit, only listed
people promote, nobody writes where login cannot work, and a local run writes as dev (allometry_app.policy)."""

import pytest

from allometry_app.policy import (Identity, can_promote, domain_allowed, host_matches, main_address, promoters,
                                  write_mode)

ANA = {"email": "Ana@CourageousLand.com", "email_verified": True, "hd": "courageousland.com"}


def test_a_verified_account_of_the_company_workspace_may_write():
    assert domain_allowed(ANA)


@pytest.mark.parametrize("claims", [
    {**ANA, "email_verified": False},                       # not verified
    {**ANA, "hd": None},                                    # a consumer account that uses the company address
    {**ANA, "hd": "other.com"},
    {"email": "ana@gmail.com", "email_verified": True},     # someone else's account
    {"email": "ana@courageousland.com.evil.com", "email_verified": True, "hd": "courageousland.com"},
    {},
])
def test_anyone_else_may_not(claims):
    assert not domain_allowed(claims)


def test_promoters_come_from_the_secrets_and_the_environment_whatever_the_case():
    allowed = promoters(["Jeferson@courageousland.com "], "bia@courageousland.com, ,")
    assert allowed == {"jeferson@courageousland.com", "bia@courageousland.com"}
    assert can_promote(Identity("JEFERSON@courageousland.com", "J", "google"), allowed)
    assert not can_promote(Identity("ana@courageousland.com", "Ana", "google"), allowed)
    assert not can_promote(None, allowed)
    assert can_promote(Identity("dev@local", "dev", "dev"), frozenset())  # a local run is its owner's
    assert can_promote(Identity("Ana", "Ana", "open"), frozenset())  # login off: whoever says who they are


@pytest.mark.parametrize("auth,store,cloud_run,open_writes,mode", [
    (True, "gs://bucket", True, False, "google"),
    (True, ".allometry-local", False, False, "google"),
    (False, ".allometry-local", False, False, "dev"),
    (False, "gs://bucket", False, False, "read_only"),  # a real bucket is never written without a login...
    (False, "gs://bucket", True, False, "read_only"),
    (False, "/tmp/store", True, False, "read_only"),  # Cloud Run without auth: nobody writes
    (False, "gs://bucket", True, True, "open"),  # ...unless open writing is turned on, on purpose
    (True, "gs://bucket", True, True, "google"),  # a configured login always wins
    (False, None, True, True, "read_only"),
])
def test_the_mode_of_the_page(auth, store, cloud_run, open_writes, mode):
    assert write_mode(auth, store, cloud_run, open_writes) == mode


def test_login_only_works_on_the_address_of_the_redirect():
    uri = "https://cl-biometrics-abc-uc.a.run.app/oauth2callback"
    assert host_matches(uri, "cl-biometrics-abc-uc.a.run.app")
    assert not host_matches(uri, "test---cl-biometrics-abc-uc.a.run.app")  # the tagged test revision
    assert not host_matches(uri, None) and not host_matches(None, "x")
    assert main_address(uri) == "https://cl-biometrics-abc-uc.a.run.app"
