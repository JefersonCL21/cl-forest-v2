"""A small register to test against: a census made with a known logistic height per species, imported as
inventario@v1 into a store in a temporary folder, and a register with a clock that ticks once per record."""

import itertools

import numpy as np
import pandas as pd

from cl_allometry import datasets as D
from cl_allometry.registry import Registry
from cl_allometry.store import LocalStore

SPECIES = {"Khaya grandifoliola": (21.4, 6.87, 0.157), "Khaya ivorensis": (24.0, 8.0, 0.14),
           "Dalbergia nigra": (15.0, 4.0, 0.2)}
WHO = "ana@courageousland.com"


def census(seed=0, n=120):
    """Heights made with a known logistic per species, as a field sheet of text."""
    rng = np.random.default_rng(seed)
    rows = []
    for sp, (b0, b1, b2) in SPECIES.items():
        dap = rng.uniform(4, 40, n)
        ht = b0 / (1 + b1 * np.exp(-b2 * dap)) * np.exp(rng.normal(0, 0.05, n))
        rows += [{"Talhao": "T1", "Linha": str(i // 10), "Arv": f"{sp[:3]}{i}", "Especie": sp, "DAP": f"{d:.2f}",
                  "HT": f"{h:.2f}"} for i, (d, h) in enumerate(zip(dap, ht))]
    return pd.DataFrame(rows)


def lab(tmp_path, raw=None):
    """(register, data, manifest) of inventario@v1 made from `raw` (default: census())."""
    store = LocalStore(tmp_path / "store")
    clock = (f"2026-10-16T12:{m:02d}:{s:02d}+00:00" for m, s in itertools.product(range(60), range(60)))
    reg = Registry(store, now=lambda: next(clock))
    raw = census() if raw is None else raw
    imported = D.import_dataset(store, raw, family="inventario", mapping=D.suggest_mapping(raw.columns),
                                author=WHO, source_name="censo.csv")
    data, m = D.load(store, imported.manifest.dataset_id)
    return reg, data, m
