"""Contact sheet of the validation cells, labelled with what to expect."""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tpms import solid_at_density
from viz import cell_surface, draw_surface
from fig_concept import equalize

COOL = (0.42, 0.58, 0.68)
HOT = (0.88, 0.48, 0.28)
GREEN = (0.44, 0.62, 0.48)


def main(fname="figF_validation_set.png"):
    rows = list(csv.DictReader(open("validation_set/reference_properties.csv")))
    fig, axes = plt.subplots(2, 4, figsize=(16.5, 11.6))
    axes = axes.ravel()

    for ax, r in zip(axes, rows):
        freq = tuple(int(c) for c in r["freq"])
        _, lvl, _ = solid_at_density(r["family"], float(r["rho_target"]),
                                     n=72, freq=freq, mode=r["mode"])
        v, f, n = cell_surface(r["family"], lvl, n=88, freq=freq, mode=r["mode"])
        sym = r["symmetry"]
        col = {"cubic": COOL, "tetragonal": HOT, "orthorhombic": GREEN}[sym]
        draw_surface(ax, v, f, n, base=col)

        safe = float(r["kapp_over_kstar_x"]) > 0.999
        ax.set_title(
            f"#{r['id']}   {r['family'].replace('_',' ')}, {r['mode']}\n"
            f"{sym}" + ("   ←  start here" if safe else ""),
            fontsize=12, weight="bold" if safe else "normal",
            color="#1a6b3a" if safe else "0.15")

        k = (float(r["k11"]), float(r["k22"]), float(r["k33"]))
        same = abs(k[0] - k[2]) / k[0] < 0.01
        kline = (f"conducts the same in every direction: {k[0]:.3f}" if same
                 else f"conducts {k[0]:.3f} / {k[1]:.3f} / {k[2]:.3f}  (x / y / z)")
        ax.text(0.5, -0.06,
                f"{float(r['rho_fe'])*100:.0f}% metal\n{kline}\n"
                f"expect your answer ≈ {float(r['kapp_over_kstar_x'])*100:.0f}% of "
                f"the x value",
                transform=ax.transAxes, ha="center", va="top", fontsize=10.5,
                color="0.25")

    equalize(axes)
    fig.suptitle(
        "The eight test cells.  All are the same material — only the shape changes.\n"
        "Numbers are conductivity relative to the solid metal (1.000 = solid metal).",
        fontsize=14)
    fig.subplots_adjust(top=0.90, bottom=0.05, left=0.02, right=0.98,
                        hspace=0.42, wspace=0.04)
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"wrote {fname}")


if __name__ == "__main__":
    main()
