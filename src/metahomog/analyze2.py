"""
What the richer study bought, from an ML-data point of view.

The headline metric is the *effective dimensionality* of the label vector: the
participation ratio of the eigenvalues of its correlation matrix,

    D = (sum lambda)^2 / sum(lambda^2)

which equals p for p uncorrelated labels and 1 when everything is a rescaling of
one underlying quantity. This is the honest way to ask "does the model actually
have more to learn?", because simply counting columns rewards redundancy — and
in this dataset almost everything is partly driven by relative density.
"""

import csv
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "study2.csv"

rows = []
with open(CSV) as fh:
    for r in csv.DictReader(fh):
        for k, v in list(r.items()):
            if k not in ("family", "mode", "freq", "sym"):
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    r[k] = np.nan
        rows.append(r)

ok = [r for r in rows if np.isfinite(r["k11"]) and r["k11"] > 1e-6
      and np.isfinite(r["E11"]) and r["E11"] > 1e-6]
print(f"{len(rows)} cells, {len(ok)} with a percolating solid in both physics\n")


def col(rs, n):
    return np.array([r[n] for r in rs], dtype=float)


def eff_dim(mat):
    """Participation ratio of the correlation-matrix eigenvalues."""
    m = mat[np.all(np.isfinite(mat), axis=1)]
    m = m[:, np.nanstd(m, axis=0) > 1e-12]
    z = (m - m.mean(0)) / m.std(0)
    lam = np.linalg.eigvalsh(np.corrcoef(z, rowvar=False))
    lam = np.clip(lam, 0, None)
    return float(lam.sum() ** 2 / (lam**2).sum()), m.shape[1]


# ------------------------------------------------------------------ 1
print("=" * 66)
print("HOW MUCH INFORMATION IS IN THE LABEL VECTOR")
print("=" * 66)

sets = {
    "CIE26 today  (rho, E_x)": ["rho", "E11"],
    "+ full stiffness        ": ["rho", "C11", "C22", "C33", "C12", "C13",
                                 "C23", "C44", "C55", "C66"],
    "+ conductivity only     ": ["rho", "E11", "k11"],
    "full 13-number vector   ": ["rho", "k11", "k22", "k33", "C11", "C22",
                                 "C33", "C12", "C13", "C23", "C44", "C55", "C66"],
}
for name, cols in sets.items():
    m = np.column_stack([col(ok, c) for c in cols])
    d, p = eff_dim(m)
    print(f"  {name}  {p:2d} labels -> effective dimensionality {d:.2f}")

print("\n  (p labels that were independent would give D = p; D near 1 means")
print("   everything is relative density wearing different hats.)")

# Raw D is dominated by density, which every label follows. The question a
# designer actually cares about is how much freedom remains *at a fixed
# density* — so regress each label on density and re-measure the leftovers.
print("\n  After regressing every label on density (what is left is the")
print("  design freedom at fixed weight):")
lr = np.log(col(ok, "rho"))
for name, cols in sets.items():
    resid = []
    for c in cols:
        if c == "rho":
            continue
        y = col(ok, c)
        m = np.isfinite(y) & (y > 0)
        if m.sum() < 10:
            continue
        ly = np.log(y)
        b, a = np.polyfit(lr[m], ly[m], 1)
        r = np.full(len(y), np.nan)
        r[m] = ly[m] - (a + b * lr[m])
        resid.append(r)
    if len(resid) < 2:
        print(f"  {name}  (only one label — nothing to compare)")
        continue
    d, p = eff_dim(np.column_stack(resid))
    print(f"  {name}  {p:2d} labels -> effective dimensionality {d:.2f}")

# ------------------------------------------------------------------ 2
print("\n" + "=" * 66)
print("DOES SHEET MODE RESPOND TO SYMMETRY BREAKING?")
print("=" * 66)
print("  the first sweep only ever stretched network cells\n")
print(f"  {'mode':8s} {'freq':6s} {'n':>3s} {'k33/k11':>9s} {'E33/E11':>9s} "
      f"{'k_iso vs cubic':>15s}")
for mode in ["network", "sheet"]:
    base = {}
    for r in ok:
        if r["mode"] == mode and r["freq"] == "111":
            base[(r["family"], r["rho_target"])] = r["k_iso"]
    for fq in ["111", "112", "113", "123"]:
        sel = [r for r in ok if r["mode"] == mode and r["freq"] == fq]
        if not sel:
            continue
        ka, ea = col(sel, "k_aniso"), col(sel, "E_aniso")
        rel = [r["k_iso"] / base[(r["family"], r["rho_target"])]
               for r in sel if (r["family"], r["rho_target"]) in base]
        print(f"  {mode:8s} {fq:6s} {len(sel):3d} "
              f"{np.nanmedian(ka):9.3f} {np.nanmedian(ea):9.3f} "
              f"{100*(np.nanmedian(rel)-1):+14.1f}%")

# ------------------------------------------------------------------ 3
print("\n" + "=" * 66)
print("ORTHORHOMBIC: THREE INDEPENDENT CONDUCTIVITIES")
print("=" * 66)
orth = [r for r in ok if r["sym"] == "orthorhombic"]
if orth:
    sp = col(orth, "k_spread")
    print(f"  {len(orth)} cells;  k spread (max/min) median {np.nanmedian(sp):.2f}, "
          f"best {np.nanmax(sp):.2f}")
    best = max(orth, key=lambda r: r["k_spread"] if np.isfinite(r["k_spread"]) else 0)
    print(f"  best: {best['family']} {best['mode']} rho={best['rho']:.3f}  "
          f"k = ({best['k11']:.3f}, {best['k22']:.3f}, {best['k33']:.3f})")
    print(f"        E = ({best['E11']:.3f}, {best['E22']:.3f}, {best['E33']:.3f})")

# ------------------------------------------------------------------ 4
print("\n" + "=" * 66)
print("DECOUPLING: THERMAL VS ELASTIC ANISOTROPY")
print("=" * 66)
print("  ratio of the two anisotropies — far from 1 means they can be steered")
print("  somewhat separately, which is what makes the pair worth storing\n")
noncubic = [r for r in ok if r["sym"] != "cubic"
            and np.isfinite(r["k_aniso"]) and np.isfinite(r["E_aniso"])
            and r["E_aniso"] > 1e-6]
by_fam = {}
for r in noncubic:
    by_fam.setdefault(r["family"], []).append(r["k_aniso"] / r["E_aniso"])
for fam, v in sorted(by_fam.items(), key=lambda kv: -np.nanmedian(kv[1])):
    print(f"  {fam:16s} median k_aniso/E_aniso = {np.nanmedian(v):5.2f}  "
          f"(n={len(v)})")

# ------------------------------------------------------------------ 5
print("\n" + "=" * 66)
print("COOLING TRADE-OFF: IN-PLANE CONDUCTION VS PERMEABILITY")
print("=" * 66)
kk, KK = col(ok, "k11"), col(ok, "K_perm")
m = np.isfinite(kk) & np.isfinite(KK) & (KK > 0)
if m.sum() > 3:
    r = np.corrcoef(np.log(kk[m]), np.log(KK[m]))[0, 1]
    print(f"  corr(log k11, log K) = {r:+.3f}  "
          f"({'competing' if r < -0.3 else 'not strongly competing'})")
    front = sorted([rr for rr in ok if np.isfinite(rr["K_perm"])],
                   key=lambda rr: -rr["k11"])[:5]
    print("  highest in-plane conduction:")
    for rr in front:
        print(f"    {rr['family']:15s} {rr['mode']:7s} f={rr['freq']} "
              f"rho={rr['rho']:.2f}  k11={rr['k11']:.3f}  K={rr['K_perm']:.2e}")

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
cmap = plt.get_cmap("tab10")
syms = ["cubic", "tetragonal", "orthorhombic"]
marks = {"cubic": "o", "tetragonal": "s", "orthorhombic": "^"}

ax = axes[0]
for i, s in enumerate(syms):
    sel = [r for r in ok if r["sym"] == s]
    if sel:
        ax.scatter(col(sel, "E_aniso"), col(sel, "k_aniso"), s=38,
                   marker=marks[s], color=cmap(i), edgecolor="k", linewidth=0.4,
                   label=s, alpha=0.85)
ax.axhline(1, color="grey", lw=0.8, ls=":")
ax.axvline(1, color="grey", lw=0.8, ls=":")
ax.set_xlabel("elastic anisotropy $E_{33}/E_{11}$")
ax.set_ylabel("thermal anisotropy $k_{33}/k_{11}$")
ax.set_title("cubic collapses to a point;\nbreaking symmetry opens a plane", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

ax = axes[1]
for j, mode in enumerate(["network", "sheet"]):
    sel = [r for r in ok if r["mode"] == mode]
    if sel:
        ax.scatter(col(sel, "rho"), col(sel, "k_aniso"), s=38,
                   marker="o" if mode == "network" else "^",
                   color=cmap(3 + j), edgecolor="k", linewidth=0.4,
                   label=mode, alpha=0.85)
ax.axhline(1, color="grey", lw=0.8, ls=":")
ax.set_xlabel("relative density")
ax.set_ylabel("thermal anisotropy $k_{33}/k_{11}$")
ax.set_title("can sheet cells steer heat too?", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

ax = axes[2]
sel = [r for r in ok if np.isfinite(r["K_perm"]) and r["K_perm"] > 0]
sc = ax.scatter(col(sel, "k11"), col(sel, "K_perm"), s=38,
                c=col(sel, "rho"), cmap="viridis", edgecolor="k", linewidth=0.4)
ax.set_yscale("log")
ax.set_xlabel("in-plane conductivity $k_{11}$")
ax.set_ylabel("permeability (Kozeny–Carman est.)")
ax.set_title("cooling trade-off, coloured by density", fontsize=10)
plt.colorbar(sc, ax=ax, label="relative density")
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig("fig4_richer.png", dpi=160, bbox_inches="tight")
print("\nwrote fig4_richer.png")
