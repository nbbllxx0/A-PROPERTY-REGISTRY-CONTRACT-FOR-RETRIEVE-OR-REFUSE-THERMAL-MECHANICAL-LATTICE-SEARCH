"""Face versus node connectivity of the rows below conn_frac 0.99 before the
thin-wall re-solve (Sec. 4.1).

conn_frac measures face connectivity; the solver assembles on the element-node
graph, where voxels that share only an edge or a corner are connected. Each row
of metagpt/catalogue.pre_thin.csv with conn_frac < 0.99 is rebuilt with the
catalogue's own mask code (stored family, mode, f, level, grid, tie rule) and its
largest node-connected component is measured with the periodic labelling of
census_wrap.py. No solves.

    python node_conn_pre_thin.py      -> data/node_conn_pre_thin.json
"""
import csv
import json
import pathlib
import sys
import time

import numpy as np

PAPER = pathlib.Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT / "metahomog"))
sys.path.insert(0, str(PAPER / "scripts"))

from tpms import solid_mask  # noqa: E402
from census_wrap import _freq, components  # noqa: E402

OUT = PAPER / "data" / "node_conn_pre_thin.json"


def main():
    pre = [r for r in csv.DictReader(open(ROOT / "metagpt" / "catalogue.pre_thin.csv"))
           if r["feasible"] == "1"]
    low = [r for r in pre if float(r["conn_frac"]) < 0.99]
    node = np.ones((3, 3, 3), dtype=bool)
    rows, t0 = [], time.time()
    for r in low:
        m = solid_mask(r["family"], float(r["level"]), n=int(r["n"]), freq=_freq(r["freq"]),
                       mode=r["mode"], tie=r.get("tie") or "legacy")
        _, _, vox = components(m, node)
        rows.append({"uid": int(r["uid"]), "family": r["family"], "mode": r["mode"],
                     "freq": r["freq"], "n": int(r["n"]),
                     "conn_frac_face": float(r["conn_frac"]),
                     "node_largest_frac": max(vox) / sum(vox) if vox else 0.0,
                     "rho_rebuild_err": abs(float(m.mean()) - float(r["rho"]))})
    summary = {
        "rows_searchable_pre_thin": len(pre),
        "rows_face_below_0.99": len(rows),
        "rows_face_below_0.99_sheet": sum(o["mode"] == "sheet" for o in rows),
        "rows_face_below_0.01": sum(o["conn_frac_face"] < 0.01 for o in rows),
        "rows_node_at_least_0.99": sum(o["node_largest_frac"] >= 0.99 for o in rows),
        "max_rho_rebuild_err": max(o["rho_rebuild_err"] for o in rows),
        "seconds": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
