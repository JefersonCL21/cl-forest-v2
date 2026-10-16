"""Field data become immutable versions whose hash says exactly which values they hold, whatever the file looked
like; problem rows are marked, never dropped (cl_allometry.datasets)."""

import io
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cl_allometry import datasets as D
from cl_allometry.store import LocalStore

CSV = Path(__file__).parents[1] / "dados" / "GERAL_V2.csv"


def field_sheet():
    """A small census as it comes from the field: text cells, a blank height, names with accents and units."""
    return pd.DataFrame({
        "Talhão": ["T1", "T1", "T1", "T2", "T2"],
        "Linha": ["1", "1", "2", "1", "1"],
        "Arv": ["1", "2", "1", "1", "2"],
        "Espécie": ["Khaya grandifoliola", "Khaya grandifoliola", "Dalbergia nigra", "Khaya ivorensis",
                    "Dalbergia nigra"],
        "CAP (cm)": ["62.8", "40.1", "31.4", "75.4", "28.3"],
        "HT": ["14.2", " ", "9.8", "16.0", "8.1"],
        "Obs": ["", "torta", "", "", ""],
    })


def stored(raw, mapping=None):
    return D.prepare(raw, mapping or D.suggest_mapping(raw.columns))


def test_columns_are_read_into_canonical_names_with_units():
    raw = field_sheet()
    assert D.suggest_mapping(raw.columns) == {"Talhão": "talhao", "Linha": "linha", "Arv": "arv",
                                              "Espécie": "especie", "CAP (cm)": "cap_cm", "HT": "ht_m"}
    df, _ = stored(raw)
    assert df["ht_m"].dtype == float and df["especie"].dtype == object
    assert "Obs" in df  # what is not mapped is kept, as text
    assert D.suggest_mapping(["DBH (cm)", "Height", "Species", "Crown area"]) == {
        "DBH (cm)": "dap_cm", "Height": "ht_m", "Species": "especie"}


def test_the_dap_comes_from_the_cap_when_it_is_missing():
    df, _ = stored(field_sheet())
    row = df[df["cap_cm"] == 62.8].iloc[0]
    assert row["dap_cm"] == pytest.approx(62.8 / np.pi)


def test_a_csv_with_semicolons_and_decimal_commas_is_read():
    text = "Especie;DAP;HT\nKhaya ivorensis;12,5;9,1\nKhaya ivorensis;20;14\n"
    raw = D.read_upload("campo.csv", text.encode("latin-1"))
    df, invalid = D.canonicalize(raw, D.suggest_mapping(raw.columns))
    assert sorted(df["dap_cm"]) == [12.5, 20.0] and sorted(df["ht_m"]) == [9.1, 14.0]
    assert not invalid["dap_cm"].any()


def test_a_csv_and_an_xlsx_of_the_same_values_have_the_same_hash():
    raw = field_sheet()
    buf = io.BytesIO()
    raw.to_excel(buf, index=False)
    from_xlsx = D.read_upload("censo.xlsx", buf.getvalue())
    from_csv = D.read_upload("censo.csv", raw.to_csv(index=False).encode("utf-8"))
    assert stored(from_xlsx)[1] == stored(from_csv)[1] == stored(raw)[1]


def test_problem_rows_are_flagged_and_kept():
    raw = pd.concat([field_sheet(), pd.DataFrame({
        "Talhão": ["T9", "T9", "T1"], "Linha": ["1", "2", "1"], "Arv": ["1", "1", "1"],
        "Espécie": ["Khaya ivorensis", None, "Khaya grandifoliola"],
        "CAP (cm)": ["abc", "30", "62.8"], "HT": ["95", "7.5", "14.2"], "Obs": ["", "", ""],
    })], ignore_index=True)
    df, _ = stored(raw)
    assert len(df) == len(raw)  # nothing dropped
    flagged = {c: int(df[c].sum()) for c in df if c.startswith("flag_")}
    assert flagged["flag_cap_cm_invalid"] == 1  # "abc"
    assert flagged["flag_ht_m_implausible"] == 1  # 95 m
    assert flagged["flag_ht_without_dap"] == 1  # the row whose CAP was "abc"
    assert flagged["flag_species_missing"] == 1
    assert flagged["flag_duplicate_row"] == 1  # the last row repeats the first
    assert flagged["flag_duplicate_key"] == 0  # a repeated row is counted once, as a repeated row


def test_the_hash_does_not_depend_on_the_order_of_rows_or_columns():
    raw = field_sheet()
    shuffled = raw.sample(frac=1, random_state=3)[list(reversed(raw.columns))]
    assert stored(raw)[1] == stored(shuffled)[1]
    a, b = stored(raw)[0], stored(shuffled)[0]
    pd.testing.assert_frame_equal(a[sorted(a.columns)], b[sorted(b.columns)])  # and they are stored the same way


def test_the_hash_changes_when_any_value_changes():
    base = stored(field_sheet())[1]
    for col, row, value in [("HT", 0, "14.3"), ("Espécie", 2, "Dalbergia  nigra"), ("Obs", 0, "x"),
                            ("HT", 1, "0.0")]:
        raw = field_sheet()
        raw.loc[row, col] = value
        assert stored(raw)[1] != base, (col, value)
    assert stored(field_sheet().iloc[:-1])[1] != base
    assert stored(field_sheet().rename(columns={"Obs": "Nota"}))[1] != base


def test_the_hash_of_a_known_table_never_changes():
    """If this fails, every dataset in every store would get a new hash: change CANON_VERSION on purpose instead."""
    assert stored(field_sheet())[1] == "6037e95fbb290ec4420d47a73087c223604b51a2581bcc1e2882346b8cf35c34"


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "store")


def imp(store, raw, **kw):
    return D.import_dataset(store, raw, family=kw.pop("family", "inventario"),
                            mapping=D.suggest_mapping(raw.columns), author="a@courageousland.com",
                            source_name="censo.csv", **kw)


def test_loading_checks_the_hash_and_catches_tampering(store):
    r = imp(store, field_sheet())
    df, m = D.load(store, "inventario@v1")
    assert m.sha256 == r.manifest.sha256 and len(df) == 5
    path = store.root / f"datasets/{m.sha256}/data.parquet"
    df.loc[0, "ht_m"] = 99.0
    path.write_bytes(D._to_parquet(df))  # someone with access to the bucket changes a value
    with pytest.raises(D.IntegrityError):
        D.load(store, "inventario@v1")


def test_importing_the_same_data_again_adds_no_version(store):
    first = imp(store, field_sheet())
    again = imp(store, field_sheet().iloc[::-1])
    assert first.created_version and not again.created_version
    assert again.manifest.dataset_id == "inventario@v1"
    assert [m.version for m in D.versions(store, "inventario")] == [1]


def test_a_new_version_points_to_the_one_before(store):
    v1 = imp(store, field_sheet()).manifest
    raw = field_sheet()
    raw.loc[1, "HT"] = "11.0"  # the missing height was measured
    v2 = imp(store, raw, note="alturas de 2026").manifest
    assert (v2.version, v2.parent_sha256, v2.note) == (2, v1.sha256, "alturas de 2026")
    assert D.families(store) == ["inventario"]
    assert D.load(store, "inventario@v1")[0]["ht_m"].notna().sum() == 4  # v1 still holds what it held


def test_going_back_to_older_values_is_a_new_version(store):
    imp(store, field_sheet())
    raw = field_sheet()
    raw.loc[0, "HT"] = "15.0"
    imp(store, raw)
    v3 = imp(store, field_sheet()).manifest
    assert v3.version == 3 and v3.sha256 == D.versions(store, "inventario")[0].sha256


def test_people_importing_at_once_get_consecutive_versions(store):
    sheets = []
    for i in range(6):
        raw = field_sheet()
        raw.loc[0, "HT"] = f"{14 + i}"
        sheets.append(raw)
    barrier = threading.Barrier(len(sheets))

    def go(raw):
        barrier.wait()
        imp(store, raw)

    threads = [threading.Thread(target=go, args=(s,)) for s in sheets]
    [t.start() for t in threads]
    [t.join() for t in threads]
    vs = D.versions(store, "inventario")
    assert [m.version for m in vs] == [1, 2, 3, 4, 5, 6] and len({m.sha256 for m in vs}) == 6
    assert all(b.parent_sha256 == a.sha256 for a, b in zip(vs, vs[1:]))


def test_a_family_name_is_safe_as_a_path():
    with pytest.raises(ValueError):
        D.import_dataset(LocalStore("unused"), field_sheet(), family="../x", mapping={}, author="a",
                         source_name="x")


def test_diff_counts_added_removed_and_changed():
    old = stored(field_sheet())[0]
    raw = field_sheet()
    raw.loc[0, "HT"] = "14.9"  # changed
    raw = raw.drop(index=4)  # removed
    raw = pd.concat([raw, pd.DataFrame({"Talhão": ["T3"], "Linha": ["1"], "Arv": ["1"],
                                        "Espécie": ["Khaya ivorensis"], "CAP (cm)": ["50"], "HT": ["12"],
                                        "Obs": [""]})], ignore_index=True)  # added
    d = D.diff(old, stored(raw)[0])
    assert (d.added, d.removed, d.changed, d.unchanged) == (1, 1, 1, 3)
    no_key = D.diff(old, stored(raw)[0], key=None)
    assert (no_key.added, no_key.removed, no_key.changed) == (2, 2, None)


@pytest.fixture(scope="module")
def census():
    if not CSV.is_file():
        pytest.skip("dados/GERAL_V2.csv not here")
    raw = D.read_upload(CSV.name, CSV.read_bytes())
    return raw, *D.prepare(raw, D.INVENTORY_MAPPING)


def test_geral_v2_is_read_as_the_census_it_is(census):
    raw, df, _ = census
    assert D.suggest_mapping(raw.columns) == D.INVENTORY_MAPPING
    assert len(df) == 20268 and "Unnamed: 0" not in df  # the saved pandas index is not data
    assert df["ht_m"].notna().sum() == 1122
    # 7,319 DAPs in the file, plus 10 jackfruits measured by CAP whose DAP column was left blank
    assert raw["DAP"].notna().sum() == 7319 and df["dap_cm"].notna().sum() == 7329
    assert int(df["flag_ht_without_dap"].sum()) == 1 and int(df["flag_duplicate_row"].sum()) == 9
    assert not df[[c for c in df if c.endswith(("_invalid", "_implausible"))]].any().any()
