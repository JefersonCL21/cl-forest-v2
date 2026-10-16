"""Puts version 1 of the height models in a store (cl_allometry.legacy.bootstrap_v1): dados/GERAL_V2.csv as
inventario@v1, every height form refitted per species on the measured heights, and the logistic promoted.

    python scripts/bootstrap_height_v1.py --store .allometry-local --author dev@local
    python scripts/bootstrap_height_v1.py --store gs://<bucket> --author jeferson@courageousland.com
    python scripts/bootstrap_height_v1.py --dry-run --author dev@local     # in a temporary folder

Running it again on the same store adds nothing.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cl_allometry.legacy import bootstrap_v1  # noqa: E402
from cl_allometry.store import open_store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", help="gs://bucket/prefix or a folder")
    ap.add_argument("--author", required=True, help="who runs it (an e-mail)")
    ap.add_argument("--csv", default=str(ROOT / "dados" / "GERAL_V2.csv"))
    ap.add_argument("--decided-by", default="Jeferson")
    ap.add_argument("--decided-on", default="16/10/2026")
    ap.add_argument("--dry-run", action="store_true", help="into a temporary folder, to see the report")
    ap.add_argument("--no-promote", action="store_true")
    args = ap.parse_args()
    if not args.store and not args.dry_run:
        ap.error("--store or --dry-run")

    with tempfile.TemporaryDirectory() as tmp:
        store = open_store(tmp if args.dry_run else args.store)
        report = bootstrap_v1(store, args.csv, author=args.author, decided_by=args.decided_by,
                              decided_on=args.decided_on, promote=not args.no_promote)
        print(f"{report.dataset_id}  sha256 {report.sha256}  ({'nova versão' if report.created_version else 'já existia'})")
        cols = ["species", "n", "logistic_vs_table", "curtis_log_vs_legacy", "converged", "fits", "rmse_pct",
                "cv_rmse_pct", "champion_seq"]
        table = report.table()
        print(table[[c for c in cols if c in table]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        if not args.no_promote and not report.promoted:
            print("\nNADA FOI PROMOVIDO: alguma logística reajustada se afastou da tabela; veja logistic_vs_table.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
