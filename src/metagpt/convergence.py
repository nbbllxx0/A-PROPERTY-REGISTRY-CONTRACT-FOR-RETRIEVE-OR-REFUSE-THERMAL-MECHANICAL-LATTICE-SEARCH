"""
How much of the catalogue is mesh, and how much is physics?

The catalogue meshes at 32 or 48 voxels per side depending on frequency, while
the nearest large-scale peer work uses 128. That gap deserves a measured bound rather than an
assurance. This re-solves a spread of cells at 32, 48 and 64 and reports how far
each property still moves.

Cells are rebuilt from their catalogue rows, so this measures the same
geometries the paper reports -- not fresh ones chosen to look convergent.

    python convergence.py --sample 12
"""

import argparse
import csv
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "metahomog"))

from tpms import solid_at_density  # noqa: E402

GRIDS = (32, 48, 64)


def _backend():
    import torch
    import gpu_homog as G
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA; the CPU path is too slow at 64^3")

    def k_of(mask):
        m = torch.from_numpy(np.ascontiguousarray(mask))[None]
        return np.diag(G.conductivity_batch(m).cpu().numpy()[0])

    def c_of(mask):
        m = torch.from_numpy(np.ascontiguousarray(mask))[None]
        return G.elasticity_batch(m).cpu().numpy()[0]

    return k_of, c_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=12)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(HERE / "catalogue.csv"))
            if r["feasible"] == "1"]
    rng = np.random.default_rng(a.seed)
    pick = [rows[i] for i in sorted(rng.choice(len(rows), a.sample, replace=False))]

    k_of, c_of = _backend()
    print(f"resolution convergence on {len(pick)} catalogue cells, grids "
          f"{GRIDS}\n")
    print(f"{'cell':34s}{'k drift 32->48':>16s}{'k drift 48->64':>16s}"
          f"{'E drift 48->64':>16s}")
    print("-" * 82)

    kd1, kd2, ed2 = [], [], []
    for r in pick:
        freq = tuple(int(c) for c in r["freq"])
        vals = {}
        for n in GRIDS:
            mask, _, _ = solid_at_density(r["family"], float(r["rho_target"]),
                                          n=n, freq=freq, mode=r["mode"])
            vals[n] = (k_of(mask), c_of(mask))

        def drift(a_, b_):
            a_, b_ = np.asarray(a_), np.asarray(b_)
            s = max(np.abs(b_).max(), 1e-30)
            return float(np.abs(b_ - a_).max() / s)

        d1 = drift(vals[32][0], vals[48][0])
        d2 = drift(vals[48][0], vals[64][0])
        e2 = drift(vals[48][1], vals[64][1])
        kd1.append(d1); kd2.append(d2); ed2.append(e2)

        name = f"{r['family']}/{r['mode'][:3]} f{r['freq']} rho={float(r['rho']):.2f}"
        print(f"{name:34s}{d1*100:15.2f}%{d2*100:15.2f}%{e2*100:15.2f}%")

    print("-" * 82)
    print(f"{'median':34s}{np.median(kd1)*100:15.2f}%"
          f"{np.median(kd2)*100:15.2f}%{np.median(ed2)*100:15.2f}%")
    print(f"{'worst':34s}{max(kd1)*100:15.2f}%{max(kd2)*100:15.2f}%"
          f"{max(ed2)*100:15.2f}%")
    print()
    print("Read the 48->64 columns: that is the residual error of the grid the "
          "catalogue actually uses for its finest cells.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
