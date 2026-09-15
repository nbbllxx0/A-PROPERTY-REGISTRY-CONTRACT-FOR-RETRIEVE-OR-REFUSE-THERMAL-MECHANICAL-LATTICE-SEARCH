"""
Re-simulate a returned design and check it against the catalogue.

The pipeline advertises `candidates -> verify -> return`, but until now the
numbers handed back were the same precomputed values that did the ranking.
Nothing recomputed them, so the arrow asserted trust instead of establishing it.

Two decisions make this a real check rather than a formality.

**The row is the geometry.** family, mode, frequency vector and *isovalue*
reproduce the cell exactly. Rebuilding from the stored level rather than
re-bisecting the density is what keeps this deterministic -- bisection stops on
a tolerance, so re-deriving it would introduce a difference that has nothing to
do with whether the stored properties are right.

**Verify with the other solver.** Re-running the code that generated a row only
proves the code is deterministic. The GPU path in gpu_homog.py shares no code
with homogenize.py -- it is matrix-free where the other assembles a sparse
system, and it projects out the free constant where the other pins a node -- so
agreement between them is evidence about the physics, not about the program.
When no GPU is present this falls back to the CPU solver and says so, because
that is a reproducibility check and should not be reported as anything more.

    python verify.py            # sample the catalogue
    python verify.py --uid 704  # one row
"""

import argparse
import csv
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "metahomog"))

from tpms import solid_mask  # noqa: E402

TOL = 1e-6          # agreement required between the two independent solvers
# Density is compared against whatever precision the catalogue recorded, not
# against machine epsilon. A row written before repair_levels.py stores
# round(rho, 6), so demanding more than that reports a tolerance artefact as a
# physics failure -- which it did, on 6 rows whose every property agreed to
# 1e-14. Full-precision rows beat this bound by orders of magnitude anyway.
RHO_TOL = 1e-6

K_KEYS = ["k11", "k22", "k33"]
C_KEYS = ["C11", "C22", "C33", "C12", "C13", "C23", "C44", "C55", "C66"]
C_INDEX = {"C11": (0, 0), "C22": (1, 1), "C33": (2, 2), "C12": (0, 1),
           "C13": (0, 2), "C23": (1, 2), "C44": (3, 3), "C55": (4, 4),
           "C66": (5, 5)}


def _backend(prefer_gpu=True):
    """Return (name, conductivity_fn, elasticity_fn, independent?)."""
    if prefer_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                import gpu_homog as G

                def kfn(mask):
                    m = torch.from_numpy(np.ascontiguousarray(mask))[None]
                    return G.conductivity_batch(m).cpu().numpy()[0]

                def cfn(mask):
                    m = torch.from_numpy(np.ascontiguousarray(mask))[None]
                    return G.elasticity_batch(m).cpu().numpy()[0]

                return "gpu (matrix-free)", kfn, cfn, True
        except Exception:
            pass
    from homogenize import homogenize_conductivity, homogenize_elasticity
    return ("cpu (assembled sparse)", homogenize_conductivity,
            lambda m: homogenize_elasticity(m, tol=1e-10), False)


def rebuild(row):
    """The exact voxel mask the catalogue row describes."""
    freq = tuple(int(c) for c in str(row["freq"]))
    return solid_mask(row["family"], float(row["level"]), n=int(float(row["n"])),
                      freq=freq, mode=row["mode"])


def verify_row(row, prefer_gpu=True, tol=TOL, elastic=True):
    """Re-simulate one catalogue row. Returns a report dict."""
    name, kfn, cfn, independent = _backend(prefer_gpu)
    t0 = time.perf_counter()
    mask = rebuild(row)

    rep = {"uid": int(float(row["uid"])), "backend": name,
           "independent": independent, "checks": [], "worst": 0.0}

    rho_new = float(mask.mean())
    rho_old = float(row["rho"])
    rep["checks"].append(("rho", rho_old, rho_new,
                          abs(rho_new - rho_old), RHO_TOL))

    k = np.diag(kfn(mask))
    for j, key in enumerate(K_KEYS):
        old = float(row[key])
        rep["checks"].append((key, old, float(k[j]),
                              _rel(old, float(k[j])), tol))

    if elastic:
        C = cfn(mask)
        scale = max(abs(float(row[q])) for q in C_KEYS) or 1.0
        for key in C_KEYS:
            a, b = C_INDEX[key]
            old, new = float(row[key]), float(C[a, b])
            rep["checks"].append((key, old, new, abs(new - old) / scale, tol))

    rep["worst"] = max(c[3] for c in rep["checks"])
    rep["ok"] = all(c[3] <= c[4] for c in rep["checks"])
    rep["seconds"] = time.perf_counter() - t0
    return rep


def _rel(old, new):
    d = max(abs(old), abs(new), 1e-30)
    return abs(new - old) / d


def format_report(rep, verbose=False):
    head = (f"uid {rep['uid']}  {'VERIFIED' if rep['ok'] else 'MISMATCH'}  "
            f"worst {rep['worst']:.2e}  via {rep['backend']}"
            f"{'' if rep['independent'] else '  (same solver: reproducibility only)'}"
            f"  {rep['seconds']:.1f}s")
    if not verbose and rep["ok"]:
        return head
    lines = [head]
    for key, old, new, err, tol in rep["checks"]:
        flag = "" if err <= tol else "   <-- exceeds tolerance"
        lines.append(f"    {key:5s} stored {old: .6f}  recomputed {new: .6f}  "
                     f"err {err:.2e}{flag}")
    return "\n".join(lines)


def load_rows(path=None):
    path = pathlib.Path(path or HERE / "catalogue.csv")
    with open(path, newline="") as fh:
        return [r for r in csv.DictReader(fh) if r.get("feasible") == "1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", type=int, default=None)
    ap.add_argument("--sample", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="force the CPU solver")
    ap.add_argument("--no-elastic", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    rows = load_rows()
    if a.uid is not None:
        rows = [r for r in rows if int(float(r["uid"])) == a.uid]
        if not rows:
            sys.exit(f"no feasible row with uid {a.uid}")
    else:
        rng = np.random.default_rng(a.seed)
        pick = rng.choice(len(rows), size=min(a.sample, len(rows)), replace=False)
        rows = [rows[i] for i in sorted(pick)]

    print(f"verifying {len(rows)} row(s) by re-simulation\n")
    bad = 0
    for r in rows:
        rep = verify_row(r, prefer_gpu=not a.cpu, elastic=not a.no_elastic)
        print(format_report(rep, verbose=a.verbose))
        bad += not rep["ok"]
    print()
    print(f"{len(rows) - bad}/{len(rows)} verified")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
