"""
Does the GPU homogeniser agree with the CPU one?

The two solve the same problem by very different routes -- the CPU assembles a
sparse matrix over solid elements only and pins a node per connected component;
the GPU keeps the whole grid, zeroes the void with a mask, and projects out the
free translation instead. If they agree to many digits on real geometries, both
are almost certainly right, because they share no code path.

Run with the CPU generation job still going: this uses one core plus the GPU.
"""

import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "metahomog"))

from tpms import solid_at_density  # noqa: E402
from homogenize import homogenize_conductivity, homogenize_elasticity  # noqa: E402
import gpu_homog as G  # noqa: E402

CASES = [
    ("gyroid", "network", (1, 1, 1), 0.30, 32),
    ("gyroid", "sheet", (1, 1, 1), 0.30, 32),
    ("diamond", "network", (1, 1, 1), 0.35, 32),
    ("schwarz_p", "sheet", (1, 1, 1), 0.25, 32),
    ("iwp", "network", (1, 1, 2), 0.40, 32),
    ("diamond", "network", (1, 1, 3), 0.35, 48),
    ("gyroid", "network", (1, 2, 3), 0.30, 48),
    ("fischer_koch_s", "sheet", (1, 1, 2), 0.35, 32),
]


def main():
    print(f"device: {G.DEV}  dtype: {G.DT}\n")
    worst_k = worst_c = 0.0
    t_cpu = t_gpu = 0.0

    for fam, mode, fq, rho, n in CASES:
        mask, level, r = solid_at_density(fam, rho, n=n, freq=fq, mode=mode)

        t = time.perf_counter()
        k_cpu = np.diag(homogenize_conductivity(mask))
        C_cpu = homogenize_elasticity(mask, tol=1e-9)
        dt_cpu = time.perf_counter() - t
        t_cpu += dt_cpu

        mt = torch.from_numpy(np.ascontiguousarray(mask))[None]
        torch.cuda.synchronize() if G.DEV == "cuda" else None
        t = time.perf_counter()
        k_gpu = G.conductivity_batch(mt).cpu().numpy()[0]
        C_gpu = G.elasticity_batch(mt).cpu().numpy()[0]
        torch.cuda.synchronize() if G.DEV == "cuda" else None
        dt_gpu = time.perf_counter() - t
        t_gpu += dt_gpu

        kd = np.diag(k_gpu)
        ek = np.abs(kd - k_cpu).max() / max(np.abs(k_cpu).max(), 1e-30)
        ec = np.abs(C_gpu - C_cpu).max() / max(np.abs(C_cpu).max(), 1e-30)
        worst_k, worst_c = max(worst_k, ek), max(worst_c, ec)

        print(f"{fam:15s} {mode:7s} f{''.join(map(str,fq))} n={n} rho={r:.3f}")
        print(f"   k  cpu {np.array2string(k_cpu, precision=6)}")
        print(f"      gpu {np.array2string(kd, precision=6)}   rel err {ek:.2e}")
        print(f"   C11 cpu {C_cpu[0,0]:.8f}  gpu {C_gpu[0,0]:.8f}"
              f"   rel err {ec:.2e}")
        print(f"   time  cpu {dt_cpu:6.1f}s   gpu {dt_gpu:6.2f}s"
              f"   ({dt_cpu/max(dt_gpu,1e-9):5.1f}x)\n")

    print("=" * 60)
    print(f"worst relative error   conductivity {worst_k:.2e}   "
          f"stiffness {worst_c:.2e}")
    print(f"total time   cpu {t_cpu:.0f}s   gpu {t_gpu:.1f}s   "
          f"speedup {t_cpu/max(t_gpu,1e-9):.1f}x")
    ok = worst_k < 1e-6 and worst_c < 1e-6
    print("VERDICT:", "agree" if ok else "DO NOT AGREE -- do not use the GPU path")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
