"""H1 (second round): homogenisation with components that keep null-space
modes, on both backends: the CPU assembled solver (pins one node per
component) and the GPU matrix-free solver (batched CG), which built the
thin-wall rows of the catalogue.

Cases, E = 1, nu = 0.3, k_s = 1, closed forms in brackets:
  rod      square rod wrapping along x only; keeps the rotation about x
           [C11 = phi*E; C22 = C33 = C12 = C13 = C23 = 0; k11 = phi; k22 = k33 = 0]
  slab     layer normal to z, wrapping along x and y; no rotational mode
           [plane stress: C11 = C22 = phi*E/(1-nu^2), C12 = nu*C11,
            C66 = phi*E/(2(1+nu)); C33 = C13 = C23 = C44 = C55 = 0;
            k11 = k22 = phi; k33 = 0]
  island   detached cube, wraps nowhere; keeps all three rotations
           [C = 0; k = 0]
  rod+isl  rod plus island: must equal the rod alone

Run: python scripts/test_partial_percolation.py [--gpu]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT / "metahomog"))
sys.path.insert(0, str(ROOT / "metagpt"))

E, NU = 1.0, 0.3


def masks(n):
    z = np.zeros((n, n, n), bool)
    rod = z.copy()
    rod[:, 5:9, 5:9] = True
    slab = z.copy()
    slab[:, :, 6:10] = True
    isl = z.copy()
    isl[11:14, 11:14, 11:14] = True
    return {"rod": rod, "slab": slab, "island": isl, "rod+isl": rod | isl}


def expected(name, m):
    phi = float(m.mean())
    C = np.zeros((6, 6))
    k = np.zeros((3, 3))
    if name in ("rod", "rod+isl"):
        phi = float((m & ~_island(m.shape[0])).mean()) if name == "rod+isl" else phi
        C[0, 0] = phi * E
        k[0, 0] = phi
    elif name == "slab":
        c11 = phi * E / (1 - NU ** 2)
        C[0, 0] = C[1, 1] = c11
        C[0, 1] = C[1, 0] = NU * c11
        C[5, 5] = phi * E / (2 * (1 + NU))
        k[0, 0] = k[1, 1] = phi
    return C, k


def _island(n):
    z = np.zeros((n, n, n), bool)
    z[11:14, 11:14, 11:14] = True
    return z


def cpu(m):
    from homogenize import homogenize_conductivity, homogenize_elasticity
    k = homogenize_conductivity(m, k_solid=1.0)
    k = np.asarray(k[0] if isinstance(k, tuple) else k)
    return np.asarray(homogenize_elasticity(m, E=E, nu=NU)), k


def gpu(m):
    import torch
    import gpu_homog as G
    t = torch.from_numpy(np.ascontiguousarray(m))[None]
    return (G.elasticity_batch(t, E=E, nu=NU).cpu().numpy()[0],
            G.conductivity_batch(t).cpu().numpy()[0])


def main(use_gpu=False, n=16):
    backends = {"cpu": cpu}
    if use_gpu:
        backends["gpu"] = gpu
    res = {"n": n, "E": E, "nu": NU, "cases": {}}
    ok = True
    for name, m in masks(n).items():
        Cx, kx = expected(name, m)
        for bname, fn in backends.items():
            C, k = fn(m)
            errC = float(np.max(np.abs(C - Cx)))
            errk = float(np.max(np.abs(k - kx)))
            res["cases"][f"{name}/{bname}"] = {"phi": float(m.mean()), "max_abs_err_C": errC,
                                               "max_abs_err_k": errk,
                                               "C11": float(C[0, 0]), "k11": float(k[0, 0])}
            good = errC < 1e-8 and errk < 1e-8
            ok &= good
            print(f"{name:8s} {bname}: max|C-closed| {errC:.2e}  max|k-closed| {errk:.2e}  "
                  f"C11 {C[0, 0]:.6f}  {'ok' if good else 'FAIL'}")
    out = PAPER / "data" / (
        "partial_percolation_cpu_gpu.json" if use_gpu else "partial_percolation.json")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("->", out)
    assert ok, "a backend missed a closed form"
    print("partial percolation: ok")


if __name__ == "__main__":
    main(use_gpu="--gpu" in sys.argv)
