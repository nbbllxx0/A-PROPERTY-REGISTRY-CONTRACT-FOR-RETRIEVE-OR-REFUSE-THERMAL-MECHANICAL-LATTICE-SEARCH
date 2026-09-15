"""Transparent-background cell renders."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tpms import solid_at_density
from viz import cell_surface, draw_surface


def hero(fname, family="gyroid", rho=0.35, freq=(1, 3, 3), mode="network",
         base=(0.90, 0.50, 0.28), size=(6.4, 6.4), az=32, el=20):
    _, lvl, _ = solid_at_density(family, rho, n=64, freq=freq, mode=mode)
    v, f, n = cell_surface(family, lvl, n=88, freq=freq, mode=mode)
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    draw_surface(ax, v, f, n, base=base, az=az, el=el)
    fig.savefig(fname, dpi=170, transparent=True, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {fname}")


if __name__ == "__main__":
    hero("figHero.png", freq=(1, 3, 3), base=(0.91, 0.52, 0.29))
    hero("figHeroCubic.png", freq=(1, 1, 1), base=(0.45, 0.62, 0.72))
