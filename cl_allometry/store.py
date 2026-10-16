"""Where datasets, fits and promotions are kept: a store that only creates. A record, once written, is never
overwritten or deleted by this code (and in production the app's service account has no delete permission on the
bucket either), so the history of a model cannot be rewritten.

Writing the same bytes again is a no-op, which makes every write safe to retry; writing different bytes to a path
that exists raises AlreadyExists, which is what orders concurrent writers (two people promoting at once: one wins,
the other reloads).

    from cl_allometry.store import open_store
    store = open_store("gs://bucket/prefix")     # or a local folder, for development and tests
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np


class AlreadyExists(Exception):
    """Different bytes are already stored at this path."""


class NotFound(Exception):
    pass


class Store(Protocol):
    uri: str

    def put(self, path: str, data: bytes, content_type: str = "application/octet-stream") -> bool:
        """Creates `path`. True if created, False if the same bytes were already there; AlreadyExists otherwise."""

    def get(self, path: str) -> bytes: ...

    def exists(self, path: str) -> bool: ...

    def list(self, prefix: str) -> list[str]:
        """The paths that start with `prefix`, sorted."""


def _check(path: str) -> str:
    p = PurePosixPath(path)
    if not path or p.is_absolute() or ".." in p.parts or "\\" in path:
        raise ValueError(f"not a store path: {path!r}")
    return str(p)


class LocalStore:
    """A folder. The file is written in full under a temporary name and then linked to its final name, which is
    atomic and fails if the name exists: a reader never sees half a record, and two writers cannot both win."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)
        self.uri = str(self.root)

    def put(self, path: str, data: bytes, content_type: str = "application/octet-stream") -> bool:
        final = self.root / _check(path)
        final.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=final.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            try:
                os.link(tmp, final)
            except FileExistsError:
                if final.read_bytes() == data:
                    return False
                raise AlreadyExists(path) from None
        finally:
            os.unlink(tmp)
        return True

    def get(self, path: str) -> bytes:
        try:
            return (self.root / _check(path)).read_bytes()
        except FileNotFoundError:
            raise NotFound(path) from None

    def exists(self, path: str) -> bool:
        return (self.root / _check(path)).is_file()

    def list(self, prefix: str) -> list[str]:
        base = self.root / prefix.rsplit("/", 1)[0] if "/" in prefix else self.root
        if not base.is_dir():
            return []
        found = (p.relative_to(self.root).as_posix() for p in base.rglob("*") if p.is_file())
        return sorted(p for p in found if p.startswith(prefix) and not p.rsplit("/", 1)[-1].startswith(".tmp-"))


class GCSStore:
    """A Cloud Storage bucket (gs://bucket/prefix). Each write carries if_generation_match=0, so the bucket itself
    refuses to replace an object; a retried request whose first attempt did succeed comes back as the same bytes."""

    def __init__(self, uri: str, client: Any = None) -> None:
        if not uri.startswith("gs://"):
            raise ValueError(f"not a gs:// uri: {uri!r}")
        bucket, _, prefix = uri[len("gs://"):].partition("/")
        prefix = prefix.strip("/")
        self.uri = f"gs://{bucket}" + (f"/{prefix}" if prefix else "")
        self._prefix = f"{prefix}/" if prefix else ""
        if client is None:
            from google.cloud import storage  # only needed when the store is a bucket

            client = storage.Client()
        self._bucket = client.bucket(bucket)

    def put(self, path: str, data: bytes, content_type: str = "application/octet-stream") -> bool:
        from google.api_core.exceptions import PreconditionFailed

        blob = self._bucket.blob(self._prefix + _check(path))
        try:
            blob.upload_from_string(data, content_type=content_type, if_generation_match=0)
        except PreconditionFailed:
            if self.get(path) == data:
                return False
            raise AlreadyExists(path) from None
        return True

    def get(self, path: str) -> bytes:
        from google.api_core.exceptions import NotFound as Missing

        try:
            return self._bucket.blob(self._prefix + _check(path)).download_as_bytes()
        except Missing:
            raise NotFound(path) from None

    def exists(self, path: str) -> bool:
        return bool(self._bucket.blob(self._prefix + _check(path)).exists())

    def list(self, prefix: str) -> list[str]:
        n = len(self._prefix)
        return sorted(b.name[n:] for b in self._bucket.list_blobs(prefix=self._prefix + prefix))


def open_store(uri: str | os.PathLike) -> Store:
    uri = str(uri)
    return GCSStore(uri) if uri.startswith("gs://") else LocalStore(uri)


def _plain(obj: Any) -> Any:
    """JSON that any system reads: no NaN or infinity (they become null) and no numpy types."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def dumps(obj: Any) -> bytes:
    return json.dumps(_plain(obj), sort_keys=True, ensure_ascii=False, indent=1, allow_nan=False).encode("utf-8")


def put_json(store: Store, path: str, obj: Any) -> bool:
    return store.put(path, dumps(obj), content_type="application/json")


def get_json(store: Store, path: str) -> Any:
    return json.loads(store.get(path).decode("utf-8"))
