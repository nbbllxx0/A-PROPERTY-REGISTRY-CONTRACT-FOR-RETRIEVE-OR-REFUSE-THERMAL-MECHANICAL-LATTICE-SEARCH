"""Merge completed D* CSV into the upgrade catalogue copy. Run after dstar_catalogue.py finishes."""
from __future__ import annotations

import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CAT = os.path.join(HERE, "catalogue.csv")
DST = os.path.join(ROOT, "paper_aei", "data", "dstar_complete_pore.csv")


def main():
    if not os.path.exists(DST):
        raise SystemExit("missing " + DST)
    with io.open(DST, encoding="utf-8", newline="") as f:
        dstar = {str(int(float(r["uid"]))): r for r in csv.DictReader(f)}
    with io.open(CAT, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())
    for col in ("D11", "D22", "D33"):
        if col not in fields:
            fields.append(col)
    n_feas = 0
    missing = []
    for r in rows:
        if str(r.get("feasible", "")).strip().lower() not in ("true", "1", "yes"):
            continue
        n_feas += 1
        rec = dstar.get(str(int(float(r["uid"]))))
        if rec is None:
            missing.append(r["uid"])
            continue
        r["D11"], r["D22"], r["D33"] = rec["D11"], rec["D22"], rec["D33"]
    if missing:
        raise SystemExit("D* missing for %d feasible uids (e.g. %s)"
                         % (len(missing), missing[:8]))
    bak = CAT + ".pre_dstar"
    if not os.path.exists(bak):
        os.replace(CAT, bak)
        CAT_OUT = CAT
        # os.replace moved CAT; write new from rows
        with io.open(CAT_OUT, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    else:
        with io.open(CAT, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    print("merged D* into", CAT, "feasible", n_feas, "dstar rows", len(dstar))
    return 0


if __name__ == "__main__":
    sys.exit(main())
