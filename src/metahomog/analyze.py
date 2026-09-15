"""Turn results.csv into the three figures and the numbers that decide the call."""

import csv
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_study import hs_upper_conductivity, hs_upper_bulk

CSV = sys.argv[1] if len(sys.argv) > 1 else "results.csv"

rows = []
with open(CSV) as fh:
    for r in csv.DictReader(fh):
        for k, v in list(r.items()):
            if k not in ("study", "family", "mode", "freq"):
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    r[k] = np.nan
        rows.append(r)

A = [r for r in rows if r["study"] == "A"]
B = [r for r in rows if r["study"] == "B"]
print(f"loaded {len(rows)} cells: study A = {len(A)}, study B = {len(B)}")


def col(rs, name):
    return np.array([r[name] for r in rs], dtype=float)


def r2(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1] ** 2)


def pearson(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


# ===================================================================== FIG 1
# Validation: effective conductivity against the rigorous bounds.
rho = col(A, "rho")
kiso = col(A, "k_iso")
fams = sorted({r["family"] for r in A})
cmap = plt.get_cmap("tab10")

fig, ax = plt.subplots(figsize=(6.2, 4.6))
rr = np.linspace(0.01, 1.0, 200)
ax.plot(rr, rr, "k--", lw=1.2, label="Wiener upper ($\\rho\\,k_s$)")
ax.plot(rr, hs_upper_conductivity(rr), "r-", lw=1.4,
        label="Hashin–Shtrikman upper")
for i, f in enumerate(fams):
    sel = [r for r in A if r["family"] == f]
    for mode, mk in (("network", "o"), ("sheet", "^")):
        s2 = [r for r in sel if r["mode"] == mode]
        if not s2:
            continue
        ax.scatter(col(s2, "rho"), col(s2, "k_iso"), s=34, marker=mk,
                   color=cmap(i % 10), edgecolor="k", linewidth=0.4,
                   label=f if mode == "network" else None)
ax.set_xlabel("relative density $\\rho$")
ax.set_ylabel("effective conductivity $k^*/k_s$")
ax.set_title("Every cell sits below the rigorous bounds\n"
             "(circles: network, triangles: sheet)", fontsize=10)
ax.legend(fontsize=7, ncol=2, loc="upper left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("fig1_bounds.png", dpi=160)
print("wrote fig1_bounds.png")

viol = (kiso > hs_upper_conductivity(rho) * (1 + 1e-6)).sum()
print(f"  HS violations: {viol}/{len(A)}   worst ratio "
      f"{np.nanmax(kiso/hs_upper_conductivity(rho)):.4f}")

# ===================================================================== FIG 2
# Degeneracy test. Raw k* vs E* is dominated by density; the honest test
# normalises both by their bound at that density first.
E11 = col(A, "E11")

# Non-percolating cells sit at exactly (0, 0) and would dominate any
# correlation. They are excluded from the statistics and counted separately --
# note that homogenization identifies them for free, no geometric heuristic.
feas = (kiso > 1e-6) & (E11 > 1e-6)
nfeas = int((~feas).sum())

print("\nSTUDY A -- is conductivity just density in disguise?")
print(f"  infeasible (non-percolating) cells excluded: {nfeas}/{len(A)}")
print(f"  R^2( k* ~ rho )   = {r2(rho[feas], kiso[feas]):.4f}")
print(f"  R^2( E* ~ rho )   = {r2(rho[feas], E11[feas]):.4f}")
print(f"  R^2( k* ~ E* ) raw= {r2(kiso[feas], E11[feas]):.4f}   (both driven by rho)")

# The honest test: fit Gibson-Ashby style power laws in density, then ask
# whether what is left over is shared. If the residuals are uncorrelated, the
# two properties carry independent information beyond density.
lr, lk, le = np.log(rho[feas]), np.log(kiso[feas]), np.log(E11[feas])
bk, ak = np.polyfit(lr, lk, 1)
be, ae = np.polyfit(lr, le, 1)
res_k = lk - (ak + bk * lr)
res_E = le - (ae + be * lr)
r2_res = r2(res_k, res_E)
print(f"\n  power-law fits (feasible cells only):")
print(f"    k* ~ rho^{bk:.3f}   (R^2 = {r2(lr, lk):.4f})")
print(f"    E* ~ rho^{be:.3f}   (R^2 = {r2(lr, le):.4f})")
print(f"  R^2( residual_k ~ residual_E ) = {r2_res:.4f}   <-- the real test")
print(f"    residual spread: k* {np.exp(res_k).min():.2f}-{np.exp(res_k).max():.2f}x, "
      f"E* {np.exp(res_E).min():.2f}-{np.exp(res_E).max():.2f}x")

# Concrete version of the same question: matched pairs.
print("\n  matched pairs (density within 2%, conductivity within 5%):")
Af = [r for r, ok in zip(A, feas) if ok]
best = None
for i in range(len(Af)):
    for j in range(i + 1, len(Af)):
        a, b = Af[i], Af[j]
        if abs(a["rho"] - b["rho"]) / a["rho"] > 0.02:
            continue
        if abs(a["k_iso"] - b["k_iso"]) / a["k_iso"] > 0.05:
            continue
        ratio = max(a["E11"], b["E11"]) / min(a["E11"], b["E11"])
        if best is None or ratio > best[0]:
            best = (ratio, a, b)
if best:
    ratio, a, b = best
    print(f"    largest stiffness gap at matched (rho, k*): {ratio:.2f}x")
    for r in (a, b):
        print(f"      {r['family']:15s} {r['mode']:7s} rho={r['rho']:.3f} "
              f"k*={r['k_iso']:.4f} E11={r['E11']:.4f}")

print("\n  spread at fixed density (feasible cells only):")
for rt in sorted({r["rho_target"] for r in A}):
    s = [r for r, ok in zip(A, feas) if ok and r["rho_target"] == rt]
    if len(s) < 2:
        continue
    kk, ee = col(s, "k_iso"), col(s, "E11")
    print(f"    rho={rt:.2f}:  k* {kk.min():.4f}-{kk.max():.4f} "
          f"({kk.max()/kk.min():.2f}x)   E* {ee.min():.4f}-{ee.max():.4f} "
          f"({ee.max()/ee.min():.2f}x)   k/E {(kk/ee).min():.2f}-{(kk/ee).max():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))
ax = axes[0]
for i, f in enumerate(fams):
    s = [r for r in A if r["family"] == f]
    kk, ee = col(s, "k_iso"), col(s, "E11")
    m = (kk > 1e-6) & (ee > 1e-6)
    ax.scatter(col(s, "rho")[m], kk[m], s=32, color=cmap(i % 10),
               edgecolor="k", linewidth=0.4, label=f)
    ax.scatter(col(s, "rho")[m], ee[m], s=32, color=cmap(i % 10),
               marker="x", linewidth=1.1)
rr = np.linspace(0.13, 0.5, 50)
ax.plot(rr, np.exp(ak) * rr**bk, "k-", lw=1.2,
        label=f"$k^*\\propto\\rho^{{{bk:.2f}}}$")
ax.plot(rr, np.exp(ae) * rr**be, "k--", lw=1.2,
        label=f"$E^*\\propto\\rho^{{{be:.2f}}}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("relative density $\\rho$")
ax.set_ylabel("property (o = $k^*$,  x = $E^*_{11}$)")
ax.set_title("both follow power laws in density", fontsize=10)
ax.grid(alpha=0.25, which="both")
ax.legend(fontsize=7, ncol=2)

ax = axes[1]
for i, f in enumerate(fams):
    idx = [j for j, r in enumerate(Af) if r["family"] == f]
    if not idx:
        continue
    ax.scatter(res_E[idx], res_k[idx], s=40, color=cmap(i % 10),
               edgecolor="k", linewidth=0.4, label=f)
ax.axhline(0, color="grey", lw=0.8, ls=":")
ax.axvline(0, color="grey", lw=0.8, ls=":")
ax.set_xlabel("stiffness residual  $\\log E^* - $ fit")
ax.set_ylabel("conductivity residual  $\\log k^* - $ fit")
ax.set_title(f"after removing density: $R^2$ = {r2_res:.2f}", fontsize=10)
ax.grid(alpha=0.25)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("fig2_degeneracy.png", dpi=160)
print("wrote fig2_degeneracy.png")

# ===================================================================== FIG 3
# Anisotropy: what symmetry breaking actually buys.
if B:
    print("\nSTUDY B -- symmetry breaking")
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    freqs = sorted({r["freq"] for r in B})
    marks = ["o", "s", "^", "D", "v", "P", "X"]
    for i, fq in enumerate(freqs):
        s = [r for r in B if r["freq"] == fq]
        cubic = len(set(fq)) == 1
        ax.scatter(col(s, "E_aniso"), col(s, "k_aniso"), s=70,
                   marker=marks[i % len(marks)],
                   facecolor="none" if cubic else cmap(i % 10),
                   edgecolor="k" if cubic else "k", linewidth=1.0,
                   label=f"({fq[0]},{fq[1]},{fq[2]})" + (" cubic" if cubic else ""))
    ax.axhline(1, color="grey", lw=0.8, ls=":")
    ax.axvline(1, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("elastic anisotropy $E^*_{33}/E^*_{11}$")
    ax.set_ylabel("thermal anisotropy $k^*_{33}/k^*_{11}$")
    ax.set_title("Cubic cells collapse to the single point (1,1).\n"
                 "Breaking cubic symmetry opens a 2-D property plane.",
                 fontsize=10)
    ax.legend(fontsize=8, title="frequency vector", title_fontsize=8,
              loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("fig3_anisotropy.png", dpi=160)
    print("wrote fig3_anisotropy.png")

    ka, Ea = col(B, "k_aniso"), col(B, "E_aniso")
    cub = np.array([len(set(r["freq"])) == 1 for r in B])
    if cub.any():
        print(f"  cubic controls: k33/k11 max deviation from 1 = "
              f"{np.nanmax(np.abs(ka[cub]-1)):.2e}  (n={cub.sum()})")
    if (~cub).any():
        print(f"  non-cubic:      k33/k11 spans {np.nanmin(ka[~cub]):.3f} .. "
              f"{np.nanmax(ka[~cub]):.3f}")
        print(f"                  E33/E11 spans {np.nanmin(Ea[~cub]):.3f} .. "
              f"{np.nanmax(Ea[~cub]):.3f}")
        print(f"  R^2(thermal aniso ~ elastic aniso), non-cubic = "
              f"{r2(ka[~cub], Ea[~cub]):.4f}")

    print("\n  per-cell detail (non-cubic):")
    for r in B:
        if len(set(r["freq"])) == 1:
            continue
        print(f"    {r['family']:14s} f={r['freq']} rho={r['rho']:.3f}  "
              f"k33/k11={r['k_aniso']:.3f}  E33/E11={r['E_aniso']:.3f}  "
              f"k_iso={r['k_iso']:.4f}  E11={r['E11']:.4f}")

# ===================================================================== cost
tt, te = col(rows, "t_therm"), col(rows, "t_elast")
print(f"\nCOST  thermal total {np.nansum(tt):7.1f}s | "
      f"elastic total {np.nansum(te):7.1f}s | "
      f"ratio {np.nansum(te)/max(np.nansum(tt),1e-9):.1f}x")
