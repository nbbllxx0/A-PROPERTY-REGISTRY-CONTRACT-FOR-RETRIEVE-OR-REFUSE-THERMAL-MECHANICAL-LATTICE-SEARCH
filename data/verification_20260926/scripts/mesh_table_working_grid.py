"""Table 9 at the working grid, with the tails.

The 60-cell probe (logs/grid_study_pooled.json, copied to data/) solves every
cell at n = 32, 48 and 64 and stores a residual of n = 32 against n = 64. The
catalogue's working grid is n = 32 when max(f) <= 2 and n = 48 when max(f) = 3
(gen_dataset.resolution), so for 40 of the 60 cells the stored residual is not
the residual of the grid the catalogue used. This recomputes every slice from
the stored per-grid values at each cell's own working grid, and reports the
worst cases for k, E and k33/k11 as well as the medians.

    python mesh_table_working_grid.py
"""
import json
import pathlib
import statistics as st

PAPER = pathlib.Path(__file__).resolve().parents[1]
SRC = PAPER / "data" / "grid_study_pooled.json"
OUT = PAPER / "data" / "mesh_probe_working_grid.json"


def working_n(freq):
    return 32 if max(int(c) for c in freq) <= 2 else 48


def resid(c, q, n):
    g = c["grids"]
    return abs(g[str(n)][q] - g["64"][q]) / abs(g["64"][q])


def summary(vals):
    v = sorted(vals)
    return {"n": len(v), "median": st.median(v), "p90": v[int(0.9 * (len(v) - 1))],
            "worst": v[-1]}


def wall_voxels_base(c):
    """Voxels across the mean wall on the base grid, measured on the cell."""
    import sys
    sys.path.insert(0, str(PAPER.parent / "metahomog"))
    from tpms import solid_at_density
    from mesh import interface_mesh, surface_area
    freq = tuple(int(x) for x in c["freq"])
    n = working_n(c["freq"])
    mask, level, rho = solid_at_density(c["family"], c["rho_target"], n=64,
                                        freq=freq, mode=c["mode"])
    v, f, _ = interface_mesh(c["family"], level, n=64, freq=freq, mode=c["mode"])
    s_v = surface_area(v, f)
    k = 2.0 if c["mode"] == "sheet" else 4.0
    return k * rho / s_v * n


def main():
    cells = json.loads(SRC.read_text(encoding="utf-8"))["cells"]
    for c in cells:
        c["_w"] = wall_voxels_base(c)
    thick = [c for c in cells if c["_w"] >= 2.0]
    out = {"source": SRC.name, "reference_grid": 64, "n_cells": len(cells),
           "n_cells_working_48": sum(working_n(c["freq"]) == 48 for c in cells),
           "n_cells_wall_ge_2": len(thick),
           "wall_voxels": {f"{c['family']}/{c['mode']}/{c['freq']}/{c['rho_target']}": c["_w"]
                           for c in cells}}
    for label, pick, pool in (("working_grid", lambda c: working_n(c["freq"]), cells),
                              ("working_grid_wall_ge_2", lambda c: working_n(c["freq"]), thick),
                              ("n32_all", lambda c: 32, cells)):
        blk = {"n": len(pool)}
        for q in ("k11", "E11", "E33", "k_ratio"):
            vals = [resid(c, q, pick(c)) for c in pool]
            s = summary(vals)
            w = max(range(len(pool)), key=lambda i: vals[i])
            s["worst_cell"] = {k: pool[w][k] for k in ("family", "mode", "freq", "rho_target", "_w")}
            blk[q] = s
        for rho in (0.25, 0.35):
            v = [resid(c, "k11", pick(c)) for c in pool if c["rho_target"] == rho]
            blk[f"k11_rho_{rho}"] = summary(v)
        for f in sorted({c["freq"] for c in pool}):
            v = [resid(c, "k11", pick(c)) for c in pool if c["freq"] == f]
            blk[f"k11_freq_{f}"] = summary(v)
        big = [c for c in pool if resid(c, "k11", pick(c)) > 0.03]
        blk["k11_over_3pct"] = {"count": len(big),
                                "rho_targets": sorted({c["rho_target"] for c in big})}
        out[label] = blk
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("cells with wall >= 2 voxels on the base grid:", out["n_cells_wall_ge_2"])
    for lab in ("working_grid", "working_grid_wall_ge_2"):
        b = out[lab]
        print(lab, {q: f"med {b[q]['median']:.2%} worst {b[q]['worst']:.1%}" for q in ("k11", "E11", "E33", "k_ratio")},
              "k>3%:", b["k11_over_3pct"])
    wg = out["working_grid"]
    print(f"working grid ({out['n_cells_working_48']} of {out['n_cells']} cells at n=48):")
    for q in ("k11", "E11", "E33", "k_ratio"):
        s = wg[q]
        print(f"  {q:8s} median {s['median']:.2%}  p90 {s['p90']:.2%}  worst {s['worst']:.1%}"
              f"  ({s['worst_cell']})")
    for rho in (0.25, 0.35):
        s = wg[f"k11_rho_{rho}"]
        print(f"  k11 at rho {rho}: median {s['median']:.2%}, worst {s['worst']:.1%}")
    for k, s in wg.items():
        if k.startswith("k11_freq_"):
            print(f"  {k}: median {s['median']:.2%}")
    print("  cells with k11 residual > 3%:", wg["k11_over_3pct"])
    print("->", OUT)


if __name__ == "__main__":
    main()
