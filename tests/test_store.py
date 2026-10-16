"""A record, once stored, is never replaced: the store only creates, the same bytes again are a no-op, and of
writers racing for a path only one wins (cl_allometry.store). The same tests run on a folder, on a simulated bucket
and, with ALLOMETRY_TEST_GCS=gs://bucket/prefix set, on a real one."""

import json
import os
import threading
import uuid

import numpy as np
import pytest
from google.api_core.exceptions import NotFound as GcsNotFound
from google.api_core.exceptions import PreconditionFailed

from cl_allometry.store import AlreadyExists, GCSStore, LocalStore, NotFound, get_json, open_store, put_json


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    def upload_from_string(self, data, content_type=None, if_generation_match=None):
        self.bucket.calls.append(if_generation_match)
        with self.bucket.lock:
            if if_generation_match == 0 and self.name in self.bucket.objects:
                raise PreconditionFailed("exists")
            self.bucket.objects[self.name] = bytes(data)

    def download_as_bytes(self):
        try:
            return self.bucket.objects[self.name]
        except KeyError:
            raise GcsNotFound("missing") from None

    def exists(self):
        return self.name in self.bucket.objects


class FakeBucket:
    """Cloud Storage as far as the store uses it: create-only uploads refused with 412 when the object exists."""

    def __init__(self):
        self.objects, self.calls, self.lock = {}, [], threading.Lock()

    def blob(self, name):
        return FakeBlob(self, name)

    def list_blobs(self, prefix=""):
        return [FakeBlob(self, n) for n in sorted(self.objects) if n.startswith(prefix)]


class FakeClient:
    def __init__(self):
        self.buckets = {}

    def bucket(self, name):
        return self.buckets.setdefault(name, FakeBucket())


def _stores():
    yield "local"
    yield "fake-gcs"
    if os.environ.get("ALLOMETRY_TEST_GCS"):
        yield "gcs"


@pytest.fixture(params=list(_stores()))
def store(request, tmp_path):
    if request.param == "local":
        return LocalStore(tmp_path / "store")
    if request.param == "fake-gcs":
        return GCSStore("gs://bucket/prefix", client=FakeClient())
    return GCSStore(f"{os.environ['ALLOMETRY_TEST_GCS'].rstrip('/')}/test-{uuid.uuid4().hex}")


def test_a_record_cannot_be_overwritten(store):
    assert store.put("fits/a.json", b"first") is True
    with pytest.raises(AlreadyExists):
        store.put("fits/a.json", b"second")
    assert store.get("fits/a.json") == b"first"


def test_writing_the_same_bytes_again_is_a_no_op(store):
    """So every write can be retried, including one whose first attempt did succeed."""
    assert store.put("d/x.bin", b"\x00\x01") is True
    assert store.put("d/x.bin", b"\x00\x01") is False
    assert store.exists("d/x.bin") and not store.exists("d/y.bin")


def test_of_writers_racing_for_a_path_only_one_wins(store):
    results, barrier = [], threading.Barrier(8)

    def write(i):
        barrier.wait()
        try:
            results.append(("created", i) if store.put("promotions/x/000001.json", f"{i}".encode()) else ("same", i))
        except AlreadyExists:
            results.append(("refused", i))

    threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    winners = [i for kind, i in results if kind == "created"]
    assert len(winners) == 1 and sum(kind == "refused" for kind, _ in results) == 7
    assert store.get("promotions/x/000001.json") == f"{winners[0]}".encode()


def test_a_missing_object_says_so(store):
    with pytest.raises(NotFound):
        store.get("nothing/here.json")


def test_list_is_sorted_and_scoped_to_its_prefix(store):
    for p in ["families/b/v000002.json", "families/b/v000001.json", "families/a/v000001.json", "fits/x.json"]:
        store.put(p, b"{}")
    assert store.list("families/b/") == ["families/b/v000001.json", "families/b/v000002.json"]
    assert store.list("families/") == ["families/a/v000001.json", "families/b/v000001.json",
                                       "families/b/v000002.json"]
    assert store.list("promotions/") == []


def test_paths_cannot_leave_the_store(store):
    for bad in ["../x.json", "/abs.json", "a\\b.json", ""]:
        with pytest.raises(ValueError):
            store.put(bad, b"x")


def test_the_bucket_itself_is_asked_to_refuse_replacing_an_object():
    """if_generation_match=0 on every upload: even a bug in the code that skipped the checks cannot replace."""
    client = FakeClient()
    s = GCSStore("gs://bucket", client=client)
    s.put("a.json", b"1")
    s.put("a.json", b"1")
    assert client.buckets["bucket"].calls == [0, 0]
    assert list(client.buckets["bucket"].objects) == ["a.json"]


def test_a_bucket_prefix_keeps_the_store_paths_relative():
    client = FakeClient()
    s = GCSStore("gs://bucket/some/prefix/", client=client)
    s.put("fits/a.json", b"1")
    assert list(client.buckets["bucket"].objects) == ["some/prefix/fits/a.json"]
    assert s.list("fits/") == ["fits/a.json"] and s.uri == "gs://bucket/some/prefix"


def test_json_records_are_plain_json_any_system_reads(store):
    put_json(store, "r.json", {"r2": float("nan"), "t": float("inf"), "n": np.int64(3), "x": np.float64(0.5)})
    text = store.get("r.json").decode()
    json.loads(text, parse_constant=lambda c: pytest.fail(f"{c} in a record"))
    assert get_json(store, "r.json") == {"r2": None, "t": None, "n": 3, "x": 0.5}


def test_a_folder_uri_opens_a_local_store(tmp_path):
    s = open_store(tmp_path / "s")
    assert isinstance(s, LocalStore)
    s.put("a/b.json", b"{}")
    assert s.list("a/") == ["a/b.json"]
