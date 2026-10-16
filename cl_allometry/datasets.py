"""Field data as immutable versions: each import is read into named columns with units (dap_cm, ht_m, ...), checked
for quality, and kept under the sha256 of its content, in a family of versions (inventario@v1, inventario@v2, ...)
where each version points to the one before.

The content hash does not depend on the order of the rows or of the columns, nor on how the file was written
(csv or xlsx, comma or point as decimal), only on the values: the same trees give the same hash. It is computed on
canonical text of each value, never on the parquet bytes, which change with the pyarrow version. Rows are stored
sorted by their own hash, so the same content is always in the same order (and a cross-validation over it gives the
same folds).

Problem rows are marked (flag_* columns), never dropped: what to leave out of a fit is the fit's choice, and it is
recorded with the fit.

    from cl_allometry import datasets
    raw = datasets.read_upload("censo.xlsx", data)
    result = datasets.import_dataset(store, raw, family="inventario", mapping=datasets.suggest_mapping(raw.columns),
                                     author="jeferson@courageousland.com", source_name="censo.xlsx")
    df, manifest = datasets.load(store, "inventario@v1")
"""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cl_allometry.store import AlreadyExists, NotFound, Store, get_json, put_json

CANON_VERSION = 1  # how values are written for the hash; changing it changes every hash
QUALITY_RULES_VERSION = "1"


@dataclass(frozen=True)
class Column:
    name: str
    kind: str  # "text" or "number"
    label: str
    plausible: tuple[float, float] | None = None


COLUMNS: dict[str, Column] = {c.name: c for c in [
    Column("especie", "text", "Espécie"),
    Column("talhao", "text", "Talhão"),
    Column("linha", "text", "Linha"),
    Column("arv", "text", "Árvore"),
    Column("idade", "number", "Idade (anos)", (0, 200)),
    Column("dap_cm", "number", "DAP (cm)", (0.5, 250)),  # a CAP of 3 cm is a real small tree, not an error
    Column("cap_cm", "number", "CAP (cm)", (1.5, 785)),
    Column("ht_m", "number", "Altura total (m)", (1.3, 60)),  # a tree with a DAP is taller than 1.3 m
    Column("crown_m2", "number", "Área da copa (m²)", (0.05, 1500)),
    Column("rho_gcm3", "number", "Densidade da madeira (g/cm³)", (0.1, 1.5)),
    Column("vol_m3", "number", "Volume (m³)", (1e-6, 60)),
    Column("agb_kg", "number", "Biomassa acima do solo (kg)", (1e-3, 2e5)),
]}

_ALIASES = {
    "especie": ("especie", "species", "nomecientifico", "sp"),
    "talhao": ("talhao", "stand"),
    "linha": ("linha", "row"),
    "arv": ("arv", "arvore", "tree", "narv"),
    "idade": ("idade", "idadeanos", "age"),
    "dap_cm": ("dapcm", "dap", "dbhcm", "dbh"),
    "cap_cm": ("capcm", "cap", "cbhcm", "cbh"),
    "ht_m": ("htm", "ht", "alturatotal", "altura", "height", "h"),
    "crown_m2": ("crownm2", "areacopam2", "areadacopa", "areacopa", "crown", "copa"),
    "rho_gcm3": ("rhogcm3", "densidadebasica", "densidade", "rho", "wd"),
    "vol_m3": ("volm3", "volume", "vol"),
    "agb_kg": ("agbkg", "biomassakg", "biomassa", "agb"),
}

# The mapping of the census kept in dados/GERAL_V2.csv.
INVENTORY_MAPPING = {"Especie": "especie", "Talhao": "talhao", "Linha": "linha", "Arv": "arv", "idade": "idade",
                     "DAP": "dap_cm", "CAP (cm)": "cap_cm", "HT": "ht_m"}

FLAG_LABELS = {
    "invalid": "valor que não é número",
    "implausible": "fora da faixa plausível",
    "ht_without_dap": "altura sem DAP",
    "cap_dap_mismatch": "CAP e DAP não batem",
    "duplicate_row": "linha repetida",
    "duplicate_key": "mesma árvore (talhão, linha, árvore) mais de uma vez",
    "species_missing": "sem espécie",
    "dap_cm_outlier": "DAP muito diferente dos outros da espécie",
}

# What a fit leaves out unless told otherwise: values that cannot be right. Duplicate keys and outliers are only
# warnings (a repeated tree number happens in the field; a big tree may be real).
DEFAULT_EXCLUDE = ("flag_dap_cm_invalid", "flag_ht_m_invalid", "flag_dap_cm_implausible", "flag_ht_m_implausible",
                   "flag_cap_dap_mismatch", "flag_duplicate_row")

_NA = "\x00"  # how a missing value is written for the hash


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s)


def suggest_mapping(columns: Iterable[str]) -> dict[str, str]:
    """{column of the file: canonical name} for the columns whose names say what they are (DAP, CAP (cm), HT...)."""
    columns = list(columns)
    normed = {c: _norm(c) for c in columns}
    out: dict[str, str] = {}
    for canon, aliases in _ALIASES.items():
        for alias in aliases:  # the first alias that matches wins, so "dapcm" beats "dap"
            hit = next((c for c in columns if normed[c] == alias and c not in out), None)
            if hit is not None:
                out[hit] = canon
                break
    return out


def sheets(data: bytes) -> list[str]:
    return pd.ExcelFile(io.BytesIO(data)).sheet_names


def read_upload(name: str, data: bytes, sheet: str | int | None = None) -> pd.DataFrame:
    """A csv or xlsx as text, untouched: what each value means is decided by canonicalize()."""
    if name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet or 0, dtype=str)
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    first = text.split("\n", 1)[0]
    sep = ";" if first.count(";") > first.count(",") else ("\t" if first.count("\t") > first.count(",") else ",")
    return pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False, na_values=[""])


def _text(s: pd.Series) -> pd.Series:
    def one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        v = unicodedata.normalize("NFC", str(v)).strip()
        if not v:
            return None
        return re.sub(r"^(-?\d+)\.0+$", r"\1", v)  # 3.0 from a spreadsheet cell is the tree number 3
    return pd.Series([one(v) for v in s.tolist()], index=s.index, dtype=object)


def _numbers(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Float values, and a mask of the cells that had text that is not a number (kept as missing, and flagged)."""
    values, invalid = [], []
    for v in _text(s).tolist():
        if v is None:
            values.append(np.nan)
            invalid.append(False)
            continue
        t = v.replace(",", ".") if ("," in v and "." not in v) else v  # 12,5 is 12.5
        try:
            values.append(float(t))
            invalid.append(not np.isfinite(values[-1]))
            if invalid[-1]:
                values[-1] = np.nan
        except ValueError:
            values.append(np.nan)
            invalid.append(True)
    return pd.Series(values, index=s.index, dtype=float), pd.Series(invalid, index=s.index, dtype=bool)


def _is_row_index(s: pd.Series) -> bool:
    nums, invalid = _numbers(s)
    n = len(s)
    return not invalid.any() and (nums.isna().all() or nums.tolist() in (list(range(n)), list(range(1, n + 1))))


def canonicalize(raw: pd.DataFrame, mapping: Mapping[str, str], dap_from_cap: bool = True
                 ) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """The file's columns renamed to canonical names (`mapping`: {column of the file: canonical name}), with numbers
    as floats and text trimmed; the other columns kept as text under their own names. A nameless column that is only
    the row number (a pandas index saved by mistake) is left out. Returns the table and, per numeric column, the mask
    of cells whose text was not a number."""
    unknown = sorted(set(mapping.values()) - set(COLUMNS))
    if unknown:
        raise ValueError(f"unknown canonical columns: {', '.join(unknown)}")
    targets = list(mapping.values())
    if len(targets) != len(set(targets)):
        raise ValueError("two columns of the file mapped to the same canonical column")
    missing = [c for c in mapping if c not in raw.columns]
    if missing:
        raise ValueError(f"columns not in the file: {', '.join(missing)}")
    clash = [c for c in raw.columns if c in COLUMNS and c not in mapping and c in targets]
    if clash:
        raise ValueError(f"the file already has columns named {', '.join(clash)}")

    out: dict[str, pd.Series] = {}
    invalid: dict[str, pd.Series] = {}
    for col in raw.columns:
        name = mapping.get(col, col)
        if col not in mapping and str(col).startswith("Unnamed:") and _is_row_index(raw[col]):
            continue
        if name in COLUMNS and COLUMNS[name].kind == "number":
            out[name], invalid[name] = _numbers(raw[col])
        else:
            out[str(name)] = _text(raw[col])
    df = pd.DataFrame(out, index=raw.index).reset_index(drop=True)
    invalid = {k: v.reset_index(drop=True) for k, v in invalid.items()}
    if dap_from_cap and "cap_cm" in df:
        if "dap_cm" not in df:
            df["dap_cm"] = np.nan
        fill = df["dap_cm"].isna() & df["cap_cm"].notna()
        df.loc[fill, "dap_cm"] = df.loc[fill, "cap_cm"] / np.pi
    return df, invalid


def _hashed_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if not c.startswith("flag_"))


def _kind(s: pd.Series) -> str:
    return "number" if pd.api.types.is_float_dtype(s) else "text"


def _cells(s: pd.Series) -> list[str]:
    if _kind(s) == "number":
        return [repr(float(v)) if v == v else _NA for v in s.tolist()]
    return [_NA if v is None or (isinstance(v, float) and v != v) else str(v) for v in s.tolist()]


def row_hashes(df: pd.DataFrame) -> pd.Series:
    cols = _hashed_columns(df)
    cells = [_cells(df[c]) for c in cols]
    rows = ["\x1f".join(r) for r in zip(*cells)] if cols else [""] * len(df)
    return pd.Series([hashlib.sha256(r.encode("utf-8")).hexdigest() for r in rows], index=df.index, dtype=object)


def content_sha256(df: pd.DataFrame) -> str:
    """The hash of the values: a line with the columns and their kinds, then the sorted row hashes."""
    cols = _hashed_columns(df)
    schema = f"canon={CANON_VERSION};" + ",".join(f"{c}:{_kind(df[c])}" for c in cols)
    body = "\n".join(sorted(row_hashes(df).tolist()))
    return hashlib.sha256(f"{schema}\n{body}".encode("utf-8")).hexdigest()


def quality_flags(df: pd.DataFrame, invalid: Mapping[str, pd.Series] | None = None) -> pd.DataFrame:
    """One boolean column per problem found (flag_*), for the rows of `df` in their order; a flag that no row has is
    still a column, so every version of a family has the same flags."""
    flags: dict[str, pd.Series] = {}
    false = pd.Series(False, index=df.index)
    for name, col in COLUMNS.items():
        if col.kind != "number" or name not in df:
            continue
        flags[f"flag_{name}_invalid"] = (invalid or {}).get(name, false).reindex(df.index, fill_value=False)
        lo, hi = col.plausible
        flags[f"flag_{name}_implausible"] = df[name].notna() & ((df[name] < lo) | (df[name] > hi))
    if "ht_m" in df:
        dap = df["dap_cm"] if "dap_cm" in df else pd.Series(np.nan, index=df.index)
        flags["flag_ht_without_dap"] = df["ht_m"].notna() & dap.isna()
    if {"cap_cm", "dap_cm"} <= set(df):
        flags["flag_cap_dap_mismatch"] = (df["cap_cm"] / np.pi - df["dap_cm"]).abs() > 0.5
    cols = _hashed_columns(df)
    flags["flag_duplicate_row"] = df.duplicated(subset=cols, keep="first") if cols else false
    key = [k for k in ("talhao", "linha", "arv") if k in df]
    if len(key) == 3:  # among the rows that are not exact copies: a copy is already flagged as such
        distinct = df[~flags["flag_duplicate_row"]]
        repeated = distinct[key].notna().all(axis=1) & distinct[key].duplicated(keep=False)
        flags["flag_duplicate_key"] = repeated.reindex(df.index, fill_value=False)
    if "especie" in df:
        flags["flag_species_missing"] = df["especie"].isna()
    if {"especie", "dap_cm"} <= set(df):
        flags["flag_dap_cm_outlier"] = _outliers(df["dap_cm"], df["especie"])
    return pd.DataFrame({k: v.astype(bool) for k, v in flags.items()}, index=df.index)


def _outliers(x: pd.Series, group: pd.Series, z: float = 6.0, min_n: int = 10) -> pd.Series:
    """Far from the rest of its species: a robust z (median and MAD) above `z`, in species with min_n trees."""
    out = pd.Series(False, index=x.index)
    for _, idx in x.groupby(group).groups.items():
        v = x.loc[idx].dropna()
        if len(v) < min_n:
            continue
        mad = float((v - v.median()).abs().median()) * 1.4826
        if mad > 0:
            out.loc[v.index] = (v - v.median()).abs() / mad > z
    return out


@dataclass(frozen=True)
class Manifest:
    """A version of a family (family@vN) and the content it holds."""
    family: str
    version: int
    sha256: str
    parent_sha256: str | None
    author: str
    created_at: str
    source_name: str
    source_bytes_sha256: str | None
    mapping: dict[str, str]
    note: str
    rows: int
    columns: dict[str, str]
    quality: dict[str, int]
    rules_version: str
    canon_version: int

    @property
    def dataset_id(self) -> str:
        return f"{self.family}@v{self.version}"

    def to_dict(self) -> dict:
        return {**asdict(self), "dataset_id": self.dataset_id}


@dataclass(frozen=True)
class ImportResult:
    manifest: Manifest
    created_version: bool  # False: the family's latest version already holds exactly these values
    data: pd.DataFrame = field(repr=False)


class IntegrityError(Exception):
    """The stored data no longer has the hash it was stored under."""


def _family_ok(family: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", family):
        raise ValueError(f"family names are lowercase letters, digits, - and _: {family!r}")
    return family


def _data_path(sha: str) -> str:
    return f"datasets/{sha}/data.parquet"


def _content_path(sha: str) -> str:
    return f"datasets/{sha}/manifest.json"


def _version_path(family: str, version: int) -> str:
    return f"families/{family}/v{version:06d}.json"


def prepare(raw: pd.DataFrame, mapping: Mapping[str, str], dap_from_cap: bool = True) -> tuple[pd.DataFrame, str]:
    """What would be stored: the canonical table with its flags, sorted by row hash, and its content hash."""
    df, invalid = canonicalize(raw, mapping, dap_from_cap=dap_from_cap)
    order = row_hashes(df).sort_values(kind="mergesort").index
    df = df.loc[order].reset_index(drop=True)
    invalid = {k: v.loc[order].reset_index(drop=True) for k, v in invalid.items()}
    df = pd.concat([df, quality_flags(df, invalid)], axis=1)
    return df, content_sha256(df)


def _to_parquet(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    return buf.getvalue()


def import_dataset(store: Store, raw: pd.DataFrame, *, family: str, mapping: Mapping[str, str], author: str,
                   source_name: str, source_bytes: bytes | None = None, note: str = "", dap_from_cap: bool = True,
                   now: datetime | None = None) -> ImportResult:
    """Stores `raw` as the next version of `family`, unless the family's latest version already holds the same
    values (then nothing is written). Order of writes: the data, its content manifest, then the version claim, which
    is the commit point; data without a claim is harmless. Two people importing at once get consecutive versions."""
    _family_ok(family)
    df, sha = prepare(raw, mapping, dap_from_cap=dap_from_cap)
    created_at = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    quality = {c: int(df[c].sum()) for c in df.columns if c.startswith("flag_")}
    content = {"sha256": sha, "rows": len(df), "columns": {c: _kind(df[c]) for c in _hashed_columns(df)},
               "quality": quality, "rules_version": QUALITY_RULES_VERSION, "canon_version": CANON_VERSION}
    if not store.exists(_data_path(sha)):  # content-addressed: whatever is there already holds these values
        try:
            store.put(_data_path(sha), _to_parquet(df))
        except AlreadyExists:
            pass
    if not store.exists(_content_path(sha)):
        try:
            put_json(store, _content_path(sha), content)
        except AlreadyExists:
            pass

    for _ in range(20):
        known = versions(store, family)
        if known and known[-1].sha256 == sha:
            return ImportResult(known[-1], created_version=False, data=df)
        version = known[-1].version + 1 if known else 1
        claim = {"family": family, "version": version, "sha256": sha,
                 "parent_sha256": known[-1].sha256 if known else None, "author": author, "created_at": created_at,
                 "source_name": source_name,
                 "source_bytes_sha256": hashlib.sha256(source_bytes).hexdigest() if source_bytes else None,
                 "mapping": dict(mapping), "note": note}
        try:
            put_json(store, _version_path(family, version), claim)
        except AlreadyExists:
            continue  # someone else took this version number: look again
        return ImportResult(_manifest(claim, content), created_version=True, data=df)
    raise RuntimeError(f"could not claim a version of {family} after 20 tries")


def _manifest(claim: dict, content: dict) -> Manifest:
    return Manifest(**{k: claim[k] for k in ("family", "version", "sha256", "parent_sha256", "author", "created_at",
                                             "source_name", "source_bytes_sha256", "mapping", "note")},
                    **{k: content[k] for k in ("rows", "columns", "quality", "rules_version", "canon_version")})


def families(store: Store) -> list[str]:
    return sorted({p.split("/")[1] for p in store.list("families/") if p.count("/") == 2})


def versions(store: Store, family: str) -> list[Manifest]:
    out = []
    for path in store.list(f"families/{_family_ok(family)}/"):
        claim = get_json(store, path)
        out.append(_manifest(claim, get_json(store, _content_path(claim["sha256"]))))
    return sorted(out, key=lambda m: m.version)


def manifest(store: Store, dataset_id: str) -> Manifest:
    family, _, v = dataset_id.partition("@v")
    try:
        claim = get_json(store, _version_path(_family_ok(family), int(v)))
    except (NotFound, ValueError):
        raise NotFound(dataset_id) from None
    return _manifest(claim, get_json(store, _content_path(claim["sha256"])))


def load(store: Store, dataset_id: str) -> tuple[pd.DataFrame, Manifest]:
    """The data of `dataset_id` (family@vN), after checking it still has its hash."""
    m = manifest(store, dataset_id)
    df = pd.read_parquet(io.BytesIO(store.get(_data_path(m.sha256))), engine="pyarrow")
    for c, kind in m.columns.items():  # parquet keeps an all-missing text column as object of None: the same
        if kind == "text" and c in df:
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    if content_sha256(df) != m.sha256:
        raise IntegrityError(f"{dataset_id}: the stored data does not match its sha256 {m.sha256}")
    return df, m


@dataclass
class DatasetDiff:
    added: int
    removed: int
    changed: int | None  # rows whose key is in both but whose values differ; None without a key
    unchanged: int
    columns_added: list[str]
    columns_removed: list[str]
    examples: pd.DataFrame = field(repr=False)


def diff(old: pd.DataFrame, new: pd.DataFrame, key: Sequence[str] | None = ("talhao", "linha", "arv")) -> DatasetDiff:
    """What changed from `old` to `new`. Rows are compared by their hash (a row that changed is one removed and one
    added); with a key that is unique in both, those pairs are counted as changed instead."""
    cols_old, cols_new = set(_hashed_columns(old)), set(_hashed_columns(new))
    common = sorted(cols_old & cols_new)
    h_old, h_new = row_hashes(old[common]), row_hashes(new[common])
    c_old, c_new = h_old.value_counts(), h_new.value_counts()
    both = c_old.index.intersection(c_new.index)
    unchanged = int(np.minimum(c_old[both], c_new[both]).sum())
    removed, added = len(old) - unchanged, len(new) - unchanged
    changed = None
    examples = new[~h_new.isin(set(h_old))].head(20)
    key = list(key or ())
    if key and set(key) <= set(common) and not old.duplicated(key).any() and not new.duplicated(key).any():
        gone = old[~h_old.isin(set(h_new))].set_index(key)
        came = new[~h_new.isin(set(h_old))].set_index(key)
        changed = int(gone.index.isin(came.index).sum())
        removed, added = removed - changed, added - changed
    return DatasetDiff(added=added, removed=removed, changed=changed, unchanged=unchanged,
                       columns_added=sorted(cols_new - cols_old), columns_removed=sorted(cols_old - cols_new),
                       examples=examples)
