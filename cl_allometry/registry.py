"""The register of fits and of the models in use: every fit kept as an immutable record (the coefficients, every
statistic, the data it saw by sha256, the filter, who fitted it, with which code and when), and, per target and
scope (a species, a genus, or generic), the champion: the fit a person promoted, with a written reason.

Promotions are events that are only ever added (promotions/<target>/<scope>/000001.json, 000002.json, ...): the
champion is the latest one, a rollback is promoting an older fit again, and the history is never rewritten. The
event numbers are claimed with writes that fail if the number is taken, so two people promoting at once cannot both
win; the second is told to reload (ConcurrentPromotion).

A fit's id comes from what was fitted (the data's sha256, the form and its formula, the filter, weights, seed,
starts, folds and the fitting algorithm's version): fitting the same thing again finds the record instead of
adding another.

    reg = Registry(open_store("gs://..."))
    rec = reg.run_fit(df, manifest, "height.logistic", Filter(especie=("Khaya grandifoliola",)), author=email)
    reg.promote(rec.fit_id, Scope.species("Khaya grandifoliola"), author=email, reason="...")
    reg.resolve("ht_m", "Khaya grandifoliola").predict(dap_cm=25)
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from cl_allometry.datasets import DEFAULT_EXCLUDE, Manifest
from cl_allometry.fit import Fit, cross_validate, fit
from cl_allometry.forms import Form, form as form_by_key
from cl_allometry.store import AlreadyExists, NotFound, Store, dumps, get_json, open_store, put_json

FIT_ALGORITHM_VERSION = 1  # bump when fit.py changes what a fit returns for the same data
LEVELS = ("species", "genus", "generic")


@dataclass(frozen=True)
class Scope:
    """Whom a champion is for: one species, a genus (the first word of the species name), or everything else."""
    level: str
    name: str | None = None

    def __post_init__(self) -> None:
        if self.level not in LEVELS or (self.level == "generic") != (self.name is None):
            raise ValueError(f"not a scope: {self.level} {self.name!r}")

    @classmethod
    def species(cls, name: str) -> Scope:
        return cls("species", name.strip())

    @classmethod
    def genus(cls, name: str) -> Scope:
        return cls("genus", name.strip())

    @classmethod
    def generic(cls) -> Scope:
        return cls("generic")

    @classmethod
    def parse(cls, text: str) -> Scope:
        level, _, name = text.partition(":")
        return cls(level, name or None)

    @classmethod
    def chain(cls, especie: str | None) -> list[Scope]:
        """Where to look for a model of a tree, most specific first."""
        especie = (especie or "").strip()
        out = [cls.species(especie), cls.genus(especie.split()[0])] if especie else []
        return out + [cls.generic()]

    def __str__(self) -> str:
        return self.level if self.name is None else f"{self.level}:{self.name}"

    @property
    def slug(self) -> str:
        if self.name is None:
            return self.level
        ascii_name = unicodedata.normalize("NFKD", self.name).encode("ascii", "ignore").decode().lower()
        return f"{self.level}.{re.sub(r'[^a-z0-9]+', '-', ascii_name).strip('-')}"


def genus_of(especie: str) -> str:
    return especie.strip().split()[0]


@dataclass(frozen=True)
class Filter:
    """Which rows of a dataset a fit uses. Rows with any of `exclude_flags` set are left out (see
    cl_allometry.datasets.DEFAULT_EXCLUDE); the fit itself also leaves out rows missing its variables."""
    especie: tuple[str, ...] | None = None
    genus: str | None = None
    talhao: tuple[str, ...] | None = None
    idade_min: float | None = None
    idade_max: float | None = None
    exclude_flags: tuple[str, ...] = DEFAULT_EXCLUDE

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        keep = pd.Series(True, index=df.index)
        if self.especie is not None:
            keep &= df["especie"].isin(self.especie)
        if self.genus is not None:
            keep &= df["especie"].fillna("").str.split().str[0] == self.genus
        if self.talhao is not None:
            keep &= df["talhao"].isin(self.talhao)
        if self.idade_min is not None:
            keep &= df["idade"] >= self.idade_min
        if self.idade_max is not None:
            keep &= df["idade"] <= self.idade_max
        for flag in self.exclude_flags:
            if flag in df:
                keep &= ~df[flag].astype(bool)
        return df[keep]

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("especie", "talhao", "exclude_flags"):
            if d[k] is not None:
                d[k] = sorted(d[k])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Filter:
        d = dict(d)
        for k in ("especie", "talhao", "exclude_flags"):
            if d.get(k) is not None:
                d[k] = tuple(d[k])
        return cls(**d)

    def fits_scope(self, scope: Scope) -> bool:
        """Whether a fit on these rows can be the model of `scope`: a species model is fitted on that species only,
        a genus model on that genus (or on species of it); a generic model on anything."""
        if scope.level == "species":
            return self.especie == (scope.name,) and self.genus in (None, genus_of(scope.name))
        if scope.level == "genus":
            if self.genus is not None:
                return self.genus == scope.name and (self.especie is None or
                                                     all(genus_of(e) == scope.name for e in self.especie))
            return bool(self.especie) and all(genus_of(e) == scope.name for e in self.especie)
        return True

    def suggested_scope(self) -> Scope:
        if self.especie is not None and len(self.especie) == 1:
            return Scope.species(self.especie[0])
        if self.genus is not None:
            return Scope.genus(self.genus)
        if self.especie and len({genus_of(e) for e in self.especie}) == 1:
            return Scope.genus(genus_of(self.especie[0]))
        return Scope.generic()


@dataclass(frozen=True)
class FitRecord:
    fit_id: str
    target: str
    form: str
    fit: Fit
    cv: dict | None
    dataset_id: str
    dataset_sha256: str
    filter: dict
    spec: dict
    n_filtered: int  # rows the filter kept; the fit's n leaves out those missing a variable
    author: str
    commit: str
    created_at: str
    schema_version: int = 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fit"] = self.fit.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FitRecord:
        return cls(**{**d, "fit": Fit.from_dict(d["fit"])})

    @property
    def flt(self) -> Filter:
        return Filter.from_dict(self.filter)

    def valid_range(self) -> dict[str, list[float]]:
        """The data range of the form's variables: where the model was fitted, so where it can be trusted."""
        return {v: r for v, r in self.fit.ranges.items() if v in form_by_key(self.form).variables}


@dataclass(frozen=True)
class Event:
    seq: int
    action: str  # "promote" or "retire"
    target: str
    scope: str
    fit_id: str | None
    reason: str
    author: str
    created_at: str
    valid_range: dict[str, list[float]] | None = None
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> Event:
        return cls(**d)


@dataclass(frozen=True)
class Champion:
    target: str
    scope: Scope
    event: Event
    record: FitRecord

    @property
    def fit_id(self) -> str:
        return self.record.fit_id

    def predict(self, **values) -> np.ndarray:
        return self.record.fit.predict(**values)

    def in_range(self, **values) -> np.ndarray:
        """True where every variable is inside the validity range of the promotion."""
        ok = None
        for v, (lo, hi) in (self.event.valid_range or {}).items():
            x = np.asarray(values[v], dtype=float)
            inside = (x >= lo) & (x <= hi)
            ok = inside if ok is None else ok & inside
        return np.ones(np.shape(next(iter(values.values()))), bool) if ok is None else ok


class ConcurrentPromotion(Exception):
    """Someone changed the champion of this scope since it was read; reload and decide again."""


@lru_cache(maxsize=1)
def code_version() -> str:
    """The commit of the code that fits: GIT_COMMIT in the deployed app, else asked of git (with -dirty when the
    working tree has changes, because then the commit alone does not say what ran)."""
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    here = Path(__file__).parent
    try:
        head = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=here, capture_output=True, text=True,
                              timeout=5, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=here,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _sha(obj) -> str:
    return hashlib.sha256(dumps(obj)).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Registry:
    def __init__(self, store: Store, now: Callable[[], str] = _now) -> None:
        self.store = store
        self._now = now
        self._fits: dict[str, FitRecord] = {}  # records never change, so they can be kept in memory

    @classmethod
    def from_env(cls) -> Registry:
        uri = os.environ.get("ALLOMETRY_STORE")
        if not uri:
            raise RuntimeError("set ALLOMETRY_STORE to the store (gs://bucket/prefix or a folder)")
        return cls(open_store(uri))

    # fits ------------------------------------------------------------------------------------------------------
    @staticmethod
    def spec(dataset_sha256: str, f: Form, flt: Filter, weights: str | None, seed: int, starts: int,
             cv_k: int | None) -> dict:
        return {"dataset_sha256": dataset_sha256, "form": f.key, "text": f.text, "fit_text": f.fit_text,
                "params": list(f.params), "filter": flt.to_dict(), "weights": weights, "seed": seed,
                "starts": starts, "cv_k": cv_k, "algorithm": FIT_ALGORITHM_VERSION}

    @staticmethod
    def fit_id_of(spec: dict, target: str) -> str:
        return f"{target}.{spec['dataset_sha256'][:12]}.{_sha(spec)[:16]}"

    @staticmethod
    def _fit_path(fit_id: str) -> str:
        target, sha12, spec16 = fit_id.split(".")
        return f"fits/{target}/{sha12}/{spec16}.json"

    def run_fit(self, data: pd.DataFrame, manifest: Manifest, form: Form | str, flt: Filter, *, author: str,
                weights: str | None = None, seed: int = 0, starts: int = 8, cv_k: int | None = 5) -> FitRecord:
        """Fits `form` to the rows of `data` (the dataset of `manifest`) that `flt` keeps, with k-fold
        cross-validation, and keeps the record. A fit that was already made is returned, not made again."""
        f = form_by_key(form) if isinstance(form, str) else form
        spec = self.spec(manifest.sha256, f, flt, weights, seed, starts, cv_k)
        fit_id = self.fit_id_of(spec, f.target)
        try:
            return self.get_fit(fit_id)
        except NotFound:
            pass
        rows = flt.apply(data)
        result = fit(f, rows, weights=weights, starts=starts, seed=seed)
        cv = (cross_validate(f, rows, k=cv_k, seed=seed, weights=weights, start=result.coefficients)
              if cv_k and result.converged else None)
        record = FitRecord(fit_id=fit_id, target=f.target, form=f.key, fit=result, cv=cv,
                           dataset_id=manifest.dataset_id, dataset_sha256=manifest.sha256, filter=flt.to_dict(),
                           spec=spec, n_filtered=len(rows), author=author, commit=code_version(),
                           created_at=self._now())
        try:
            put_json(self.store, self._fit_path(fit_id), record.to_dict())
        except AlreadyExists:  # someone made the same fit at the same moment: theirs is the record
            return self.get_fit(fit_id)
        return self.get_fit(fit_id)  # as stored (NaN as None), so a fresh record and a loaded one are the same

    def get_fit(self, fit_id: str) -> FitRecord:
        if fit_id not in self._fits:
            self._fits[fit_id] = FitRecord.from_dict(get_json(self.store, self._fit_path(fit_id)))
        return self._fits[fit_id]

    def list_fits(self, target: str | None = None, dataset_sha256: str | None = None) -> list[FitRecord]:
        prefix = "fits/" + (f"{target}/" if target else "")
        if target and dataset_sha256:
            prefix += f"{dataset_sha256[:12]}/"
        out = []
        for path in self.store.list(prefix):
            _, t, sha12, name = path.split("/")
            out.append(self.get_fit(f"{t}.{sha12}.{name.removesuffix('.json')}"))
        if dataset_sha256:
            out = [r for r in out if r.dataset_sha256 == dataset_sha256]
        return sorted(out, key=lambda r: r.created_at)

    # promotions ------------------------------------------------------------------------------------------------
    @staticmethod
    def _events_prefix(target: str, scope: Scope) -> str:
        return f"promotions/{target}/{scope.slug}/"

    def history(self, target: str, scope: Scope) -> list[Event]:
        return [Event.from_dict(get_json(self.store, p)) for p in self.store.list(self._events_prefix(target, scope))]

    def _append(self, target: str, scope: Scope, event: dict, expected_seq: int | None) -> Event:
        last = self.history(target, scope)
        seq = last[-1].seq + 1 if last else 1
        if expected_seq is not None and expected_seq != seq - 1:
            raise ConcurrentPromotion(f"{target} {scope}: the champion changed (event {seq - 1}) since it was read "
                                      f"(event {expected_seq})")
        e = Event(seq=seq, target=target, scope=str(scope), created_at=self._now(), **event)
        try:
            put_json(self.store, f"{self._events_prefix(target, scope)}{seq:06d}.json", asdict(e))
        except AlreadyExists:
            raise ConcurrentPromotion(f"{target} {scope}: someone else promoted at the same time") from None
        return e

    def promote(self, fit_id: str, scope: Scope, *, author: str, reason: str,
                valid_range: dict[str, Sequence[float]] | None = None, expected_seq: int | None = None,
                warnings: Sequence[str] = ()) -> Event:
        """Makes the fit the champion of `scope`. expected_seq: the last event number the person saw (0 for none),
        so a decision made on a stale screen is refused instead of silently undoing someone else's."""
        rec = self.get_fit(fit_id)
        if not rec.fit.converged:
            raise ValueError(f"{fit_id} did not converge: {rec.fit.message}")
        if len(reason.strip()) < 10:
            raise ValueError("a promotion needs a reason (at least 10 characters)")
        if not rec.flt.fits_scope(scope):
            raise ValueError(f"{fit_id} was fitted on {rec.filter.get('especie') or 'all species'}, "
                             f"so it cannot be the model of {scope}")
        current = self.champion(rec.target, scope)
        if current is not None and current.fit_id == fit_id:
            raise ValueError(f"{fit_id} is already the champion of {rec.target} {scope}")
        rng = {k: [float(v[0]), float(v[1])] for k, v in (valid_range or rec.valid_range()).items()}
        return self._append(rec.target, scope, {"action": "promote", "fit_id": fit_id, "reason": reason.strip(),
                                                "author": author, "valid_range": rng,
                                                "warnings": list(warnings)}, expected_seq)

    def retire(self, target: str, scope: Scope, *, author: str, reason: str, expected_seq: int | None = None
               ) -> Event:
        """Leaves `scope` without a champion: its trees fall back to the genus or the generic model."""
        if self.champion(target, scope) is None:
            raise ValueError(f"{target} {scope} has no champion to retire")
        if len(reason.strip()) < 10:
            raise ValueError("retiring a model needs a reason (at least 10 characters)")
        return self._append(target, scope, {"action": "retire", "fit_id": None, "reason": reason.strip(),
                                            "author": author}, expected_seq)

    def _champion_of(self, events: list[Event], target: str, scope: Scope) -> Champion | None:
        if not events or events[-1].action != "promote":
            return None
        return Champion(target=target, scope=scope, event=events[-1], record=self.get_fit(events[-1].fit_id))

    def champion(self, target: str, scope: Scope) -> Champion | None:
        return self._champion_of(self.history(target, scope), target, scope)

    def champions(self, target: str | None = None) -> list[Champion]:
        last: dict[str, str] = {}
        for path in self.store.list("promotions/" + (f"{target}/" if target else "")):
            last[path.rsplit("/", 1)[0]] = path  # paths come sorted, so the last one per scope is the latest
        out = []
        for path in last.values():
            e = Event.from_dict(get_json(self.store, path))
            if e.action == "promote":
                out.append(Champion(target=e.target, scope=Scope.parse(e.scope), event=e,
                                    record=self.get_fit(e.fit_id)))
        return sorted(out, key=lambda c: (c.target, LEVELS.index(c.scope.level), c.scope.name or ""))

    def resolve(self, target: str, especie: str | None) -> Champion | None:
        """The model for a tree of `especie`: its species' champion, else its genus', else the generic one."""
        for scope in Scope.chain(especie):
            c = self.champion(target, scope)
            if c is not None:
                return c
        return None

    def resolver(self, target: str) -> Callable[[str | None], Champion | None]:
        """resolve() for many trees: the champions read once, then looked up in memory."""
        table = {(str(c.scope)): c for c in self.champions(target)}

        def lookup(especie: str | None) -> Champion | None:
            return next((table[str(s)] for s in Scope.chain(especie) if str(s) in table), None)
        return lookup
