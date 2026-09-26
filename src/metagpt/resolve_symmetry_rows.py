"""Re-solve catalogue rows whose stored tensor breaks the cell's symmetry.

A cubic cell has k11 = k22 = k33 and E11 = E22 = E33; a tetragonal cell with
frequency (a, a, b) has k11 = k22 and E11 = E22 (and the analogous pair for
(a, b, b)). The analytic cells satisfy these identities exactly. A stored row
can break them because the isovalue bisection lands on a set of
symmetry-equivalent voxels whose computed values differ only in their last bits,
and splits that set (see tpms.TIE_RULES).

This script finds every searchable row whose identity is broken by more than
THRESHOLD, rebuilds its mask at the stored level with the tie-safe rule, and
re-solves every stored quantity on that mask: k, C and derived moduli,
conn_frac, surface area, the void coefficients B and the pore diffusivity D*.
The rule used is written to a new `tie` column so the row still rebuilds
exactly; every other row keeps `tie = legacy` and is not touched.

    python resolve_symmetry_rows.py            # report only
    python resolve_symmetry_rows.py --write    # update catalogue.csv etc.
"""
import argparse
import csv
import io
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "metahomog"))
sys.path.insert(0, HERE)

from tpms import solid_mask  # noqa: E402
from homogenize import (homogenize_conductivity,  # noqa: E402
                        homogenize_conductivity_2phase)
import gen_dataset  # noqa: E402

CAT = os.path.join(HERE, "catalogue.csv")
VOID = os.path.join(HERE, "catalogue_void.csv")
DSTAR = os.path.join(ROOT, "paper_aei_v4", "data", "dstar_complete_pore.csv")
LOG = os.path.join(ROOT, "paper_aei_v4", "data", "symmetry_resolve.json")
THRESHOLD = 0.01
EPS_VOID = 1e-3          # same secant step as void_coefficients.py


def identity_pairs(sym, freq):
    """Axis pairs (0-based) that the analytic cell makes equal."""
    if sym == "cubic":
        return [(0, 1), (0, 2)]
    if sym == "tetragonal":
        a, b, c = (int(x) for x in freq)
        if a == b:
            return [(0, 1)]
        if b == c:
            return [(1, 2)]
        if a == c:
            return [(0, 2)]
    return []


def violation(row):
    pairs = identity_pairs(row["sym"], row["freq"])
    worst = 0.0
    for i, j in pairs:
        for q in ("k", "E"):
            a = float(row[f"{q}{i+1}{i+1}"])
            b = float(row[f"{q}{j+1}{j+1}"])
            worst = max(worst, abs(b / a - 1.0))
    return worst


def read_csv(path):
    with io.open(path, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        return rd.fieldnames, list(rd)


def write_csv(path, fields, rows):
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    fields, rows = read_csv(CAT)
    searchable = [r for r in rows if r["feasible"] == "1"]
    checkable = [r for r in searchable if identity_pairs(r["sym"], r["freq"])]
    todo = [r for r in checkable if violation(r) > THRESHOLD]
    print(f"searchable {len(searchable)}, with a symmetry identity "
          f"{len(checkable)}, broken by > {THRESHOLD:.0%}: {len(todo)}")

    log = {"threshold": THRESHOLD, "n_searchable": len(searchable),
           "n_checkable": len(checkable),
           "before_worst_all": max(violation(r) for r in checkable),
           "rows": []}
    new_vals = {}
    for r in todo:
        t0 = time.time()
        freq = tuple(int(c) for c in r["freq"])
        n = int(r["n"])
        level = float(r["level"])
        # Solve both symmetric choices; keep one that still carries load and
        # heat (a row must stay usable), then the one closer to the target.
        cands = []
        for tie in ("include", "exclude"):
            m = solid_mask(r["family"], level, n=n, freq=freq, mode=r["mode"], tie=tie)
            cand = {k: r[k] for k in fields}
            for k in gen_dataset.FIELDS:
                if k not in ("uid", "family", "mode", "freq", "sym", "n",
                             "rho_target", "t_total"):
                    cand[k] = np.nan
            gen_dataset.solve_fields(cand, m, level, r["family"], r["mode"], freq, n)
            err = abs(float(m.mean()) - float(r["rho_target"]))
            cands.append((-int(cand["feasible"]), err, tie, m, cand))
        cands.sort(key=lambda c: (c[0], c[1]))
        _, _, tie, mask, new = cands[0]
        k0 = np.array([float(new["k11"]), float(new["k22"]), float(new["k33"])])
        k2 = homogenize_conductivity_2phase(mask, k_solid=1.0, k_void=EPS_VOID)
        B = (np.diag(k2) - k0) / EPS_VOID
        D = homogenize_conductivity(~mask, k_solid=1.0)
        new["B11"], new["B22"], new["B33"] = ("%.6f" % x for x in B)
        new["D11"], new["D22"], new["D33"] = ("%.6f" % D[i, i] for i in range(3))
        new["tie"] = tie
        new["t_total"] = round(time.time() - t0, 2)
        ent = {"uid": r["uid"], "family": r["family"], "mode": r["mode"],
               "freq": r["freq"], "n": n, "tie": tie,
               "rho_before": float(r["rho"]), "rho_after": float(new["rho"]),
               "voxels_changed": int(round(abs(float(new["rho"]) - float(r["rho"]))
                                           * n ** 3)),
               "violation_before": violation(r),
               "violation_after": violation(new),
               "feasible_after": int(new["feasible"])}
        for q in ("k11", "k22", "k33", "E11", "E22", "E33", "conn_frac", "B11", "D11"):
            ent[q + "_before"] = float(r[q])
            ent[q + "_after"] = float(new[q])
        log["rows"].append(ent)
        new_vals[r["uid"]] = new
        print(f"  uid {r['uid']:>5} {r['family']:>14} {r['mode']:7} f{r['freq']} "
              f"tie={tie:7} rho {float(r['rho']):.4f}->{float(new['rho']):.4f} "
              f"violation {ent['violation_before']:.2%}->{ent['violation_after']:.2%} "
              f"({ent['k33_after']/ent['k11_after']:.4f} k33/k11)")

    after = [violation(new_vals.get(r["uid"], r)) for r in checkable]
    log["after_worst_all"] = max(after)
    log["after_n_over_threshold"] = sum(v > THRESHOLD for v in after)
    print(f"worst identity violation over {len(checkable)} checkable rows: "
          f"{log['before_worst_all']:.2%} -> {log['after_worst_all']:.2%}")

    if not a.write:
        print("report only; pass --write to update the catalogue")
        return 0

    stamp = os.path.join(HERE, "catalogue.pre_tie.csv")
    if not os.path.exists(stamp):
        shutil.copy2(CAT, stamp)
    out_fields = fields + ([] if "tie" in fields else ["tie"])
    out_rows = []
    for r in rows:
        rr = dict(new_vals.get(r["uid"], r))
        rr.setdefault("tie", "legacy")
        if not rr.get("tie"):
            rr["tie"] = "legacy"
        out_rows.append({k: rr.get(k, "") for k in out_fields})
    write_csv(CAT, out_fields, out_rows)

    for path, cols in ((VOID, ("B11", "B22", "B33")), (DSTAR, ("D11", "D22", "D33"))):
        if not os.path.exists(path):
            continue
        f2, r2 = read_csv(path)
        for rr in r2:
            if rr["uid"] in new_vals:
                for c in cols:
                    rr[c] = new_vals[rr["uid"]][c]
                if "rho_rebuilt" in rr:
                    rr["rho_rebuilt"] = "%.6f" % float(new_vals[rr["uid"]]["rho"])
                if "phi" in rr:
                    rr["phi"] = "%.6f" % (1 - float(new_vals[rr["uid"]]["rho"]))
        write_csv(path, f2, r2)

    with io.open(LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)
    print("wrote", CAT, "and", LOG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
