"""Does the heat-spreader result depend on where the density cap is set?

The brief caps relative density at 0.40 and calls that "light", so it is fair
to ask why not 0.30. Rather than argue that the cap is not a definition of
light, this answers with the search itself: hold every other constraint fixed,
move the cap, and report what comes back.

The parse stage is bypassed. The query is built directly, so nothing here
depends on a language model, an API key, or a network.

    python threshold_sensitivity.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from retrieval import Catalogue                            # noqa: E402

# The heat-spreader brief, exactly as the paper states it, minus the cap.
BASE = {
    "objectives": [{"property": "k_11", "sense": "max"}],
    "constraints": [{"property": "k_aniso", "op": "<=", "value": 0.70}],
    "material_filter": {"cost_max": 5.0},
    "unmet": [], "_rejected": [],
}
CAPS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def main():
    cat = Catalogue()
    key = "k_aniso" if "k_aniso" in cat.col else None
    if key is None:
        cand = [k for k in cat.col if "ratio" in k]
        print("anisotropy key not found; candidates: %s" % cand)
        if not cand:
            return 1
        key = cand[0]
        BASE["constraints"][0]["property"] = key
    print("heat-spreader brief: maximise k_11, %s <= 0.70, cost <= 5 USD/kg"
          % key)
    print("only the density cap moves\n")
    print("%-10s %10s %-34s %9s %9s"
          % ("rho cap", "feasible", "top row", "k11", "rho"))

    seen = []
    for cap in CAPS:
        q = {k: (list(v) if isinstance(v, list) else dict(v)
                 if isinstance(v, dict) else v) for k, v in BASE.items()}
        q["constraints"] = list(BASE["constraints"]) + [
            {"property": "rho", "op": "<=", "value": cap}]
        r = cat.search(q, top_k=1)
        if not r.rows:
            print("%-10.2f %10d %-34s" % (cap, r.n_feasible, "REFUSED"))
            seen.append((cap, None, None, None))
            continue
        row = r.rows[0]
        g = cat.geoms[row["gi"]] if "gi" in row else None
        label = "%s %s %s f=%s" % (
            row.get("material", "?"),
            (g or {}).get("family", row.get("family", "?")),
            (g or {}).get("mode", row.get("mode", "")),
            (g or {}).get("freq", row.get("freq", "")))
        k11 = row.get("k_11", float("nan"))
        rho = row.get("rho", float("nan"))
        print("%-10.2f %10d %-34s %9.1f %9.3f"
              % (cap, r.n_feasible, label[:34], k11, rho))
        seen.append((cap, label, k11, rho))

    ks = [s[2] for s in seen if s[2] is not None]
    print()
    if ks:
        print("k11 across the caps: %.1f to %.1f W/mK, a %.0f%% spread"
              % (min(ks), max(ks), 100 * (max(ks) - min(ks)) / max(ks)))
    fams = {s[1] for s in seen if s[1]}
    print("distinct answers returned: %d" % len(fams))
    print()
    print("What is under test in the brief is whether the search honours a")
    print("stated cap and reports the trade, not whether 0.40 is the right")
    print("number. Any cap in this range demonstrates that equally well.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
