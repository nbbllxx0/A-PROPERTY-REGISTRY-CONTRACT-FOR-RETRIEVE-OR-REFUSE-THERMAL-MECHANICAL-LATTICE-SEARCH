"""Pool both halves of the grid study into one 60-cell result.

The first 30 cells were drawn with a stride that paired every rho = 0.35 with
f = (1,1,3) and nothing else, so "density separates the residual, not
frequency" was an association the design could not test. The complementary 30
fill exactly those gaps. Together they are the full grid: every rho x f pair
solved ten times.

    python grid_pool.py
"""
import json
import os
import sys
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(os.path.dirname(HERE), "logs")
PARTS = [os.path.join(LOGS, "grid_study.json"),
         os.path.join(LOGS, "grid_study_offset1.json")]
OUT = os.path.join(LOGS, "grid_study_pooled.json")
QUANT = ("k11", "k33", "E11", "E33", "k_ratio")


def main():
    cells = []
    for p in PARTS:
        if not os.path.exists(p):
            print("missing %s" % p)
            return 1
        cells += json.load(open(p, encoding="utf-8"))["cells"]

    seen, uniq = set(), []
    for c in cells:
        key = (c["family"], c["mode"], c["rho_target"], c["freq"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    print("pooled cells: %d (%d after dropping duplicates)"
          % (len(cells), len(uniq)))

    pairs = collections.Counter((c["rho_target"], c["freq"]) for c in uniq)
    print("\nevery density x frequency pair, and how many cells each has")
    for k in sorted(pairs):
        print("   rho %.2f  f=%s : %2d" % (k[0], k[1], pairs[k]))
    balanced = len(set(pairs.values())) == 1
    print("   balanced design: %s" % ("yes" if balanced else "NO"))

    print("\nresidual of n=32 against n=64, over %d cells" % len(uniq))
    summary = {}
    for q in QUANT:
        v = np.array([c["residual"][q] for c in uniq
                      if q in c.get("residual", {})
                      and np.isfinite(c["residual"][q])])
        summary[q] = {"n": int(v.size), "median": float(np.median(v)),
                      "p90": float(np.percentile(v, 90)),
                      "worst": float(v.max())}
        print("  %-8s n=%2d  median %5.2f%%  p90 %5.2f%%  worst %5.2f%%"
              % (q, v.size, 100 * np.median(v),
                 100 * np.percentile(v, 90), 100 * v.max()))

    # The question the first half could not answer.
    print("\nnow that rho and f are crossed, which one separates the residual?")
    for name, key in (("rho", "rho_target"), ("f", "freq")):
        print("  by %s" % name)
        groups = collections.defaultdict(list)
        for c in uniq:
            if "k11" in c.get("residual", {}):
                groups[c[key]].append(c["residual"]["k11"])
        for g in sorted(groups):
            a = np.array(groups[g])
            print("     %-6s n=%2d  median %5.2f%%  worst %5.2f%%"
                  % (g, a.size, 100 * np.median(a), 100 * a.max()))

    over = [c for c in uniq if c.get("residual", {}).get("k11", 0) > 0.03]
    print("\ncells above 3%% in k11: %d of %d" % (len(over), len(uniq)))
    print("   their densities: %s"
          % sorted({c["rho_target"] for c in over}))
    print("   their frequencies: %s" % sorted({c["freq"] for c in over}))

    json.dump({"cells": uniq, "summary": summary,
               "balanced": balanced}, open(OUT, "w", encoding="utf-8"),
              indent=1)
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
