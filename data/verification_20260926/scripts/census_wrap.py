"""Components of every searchable catalogue mask, by how many axes they wrap.

Whether a rigid rotation survives the periodic condition depends on how a
component wraps the cell: wrapping along two or three independent axes leaves
no rotational kernel, wrapping along one keeps the rotation about that axis,
and an isolated island keeps all three. Every searchable row is rebuilt from
its stored (family, mode, f, level, n, tie rule) with the catalogue's own mask
code and labelled twice: under node connectivity (the finite-element graph)
and under face connectivity (what conn_frac measures). No solves.

    python census_wrap.py      -> data/census_wrap.json
"""
import csv
import json
import pathlib
import sys
import time

import numpy as np
from scipy.ndimage import label

PAPER = pathlib.Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT / "metahomog"))

from tpms import solid_mask  # noqa: E402

OUT = PAPER / "data" / "census_wrap.json"


def _freq(s):
    return tuple(int(ch) for ch in str(s))


def components(mask, structure):
    """Periodic components with winding. Label the open grid, join labels
    across the periodic faces while tracking the image offset, and take the
    rank of the closed-loop offsets as the number of wrapped axes.
    Returns (n_components, wrap rank per component, voxels per component)."""
    lab, n = label(mask, structure=structure)
    if n == 0:
        return 0, [], []
    shape = np.array(mask.shape)
    parent = list(range(n + 1))
    offset = [np.zeros(3, dtype=int) for _ in range(n + 1)]
    cycles = {}

    def find(a):
        path = []
        while parent[a] != a:
            path.append(a)
            a = parent[a]
        root = a
        acc = np.zeros(3, dtype=int)
        for node in reversed(path):
            acc = acc + offset[node]
            offset[node] = acc.copy()
            parent[node] = root
        return root

    def union(a, b, shift):
        ra, rb = find(a), find(b)
        oa = offset[a] if a != ra else np.zeros(3, int)
        ob = offset[b] if b != rb else np.zeros(3, int)
        if ra == rb:
            cyc = oa + shift - ob
            if np.any(cyc):
                cycles.setdefault(ra, []).append(cyc)
            return
        parent[rb] = ra
        offset[rb] = oa + shift - ob
        if rb in cycles:
            cycles.setdefault(ra, []).extend(cycles.pop(rb))

    offs = [np.array(o) - 1 for o in np.ndindex(3, 3, 3)
            if np.any(np.array(o) - 1) and structure[o]]
    idx = np.argwhere(mask)
    for o in offs:
        tgt = idx + o
        wrap = np.floor_divide(tgt, shape)
        crossing = np.any(wrap != 0, axis=1)
        if not crossing.any():
            continue
        src = idx[crossing]
        t = np.mod(tgt[crossing], shape)
        ok = mask[t[:, 0], t[:, 1], t[:, 2]]
        src, t, w = src[ok], t[ok], wrap[crossing][ok]
        if len(src) == 0:
            continue
        la = lab[src[:, 0], src[:, 1], src[:, 2]]
        lb = lab[t[:, 0], t[:, 1], t[:, 2]]
        for a, b, w0, w1, w2 in np.unique(np.column_stack([la, lb, w]), axis=0):
            union(int(a), int(b), np.array([w0, w1, w2], dtype=int))
    roots = {}
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    for a in range(1, n + 1):
        r = find(a)
        roots[r] = roots.get(r, 0) + int(sizes[a])
    wraps, vox = [], []
    for r, sz in roots.items():
        cyc = cycles.get(r, [])
        wraps.append(int(np.linalg.matrix_rank(np.array(cyc))) if cyc else 0)
        vox.append(sz)
    return len(roots), wraps, vox


def main():
    rows = [r for r in csv.DictReader(open(ROOT / "metagpt" / "catalogue.csv"))
            if float(r["feasible"] or 0) >= 1]
    node = np.ones((3, 3, 3), dtype=bool)
    face = np.zeros((3, 3, 3), dtype=bool)
    face[1, 1, :] = face[1, :, 1] = face[:, 1, 1] = True
    out, t0 = [], time.time()
    for i, r in enumerate(rows):
        m = solid_mask(r["family"], float(r["level"]), n=int(r["n"]), freq=_freq(r["freq"]),
                       mode=r["mode"], tie=r.get("tie") or "legacy")
        nc, wraps, vox = components(m, node)
        nf, _, _ = components(m, face)
        out.append({"uid": int(r["uid"]), "family": r["family"], "mode": r["mode"],
                    "freq": r["freq"], "n": int(r["n"]), "builder": r.get("builder"),
                    "conn_frac": float(r["conn_frac"]),
                    "rho_rebuild_err": abs(float(m.mean()) - float(r["rho"])),
                    "n_components": nc, "wrap_dims": sorted(wraps, reverse=True),
                    "solid_in_islands": int(sum(v for w, v in zip(wraps, vox) if w == 0)),
                    "solid_total": int(m.sum()), "n_face_components": nf})
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    summ = {
        "rows": len(out),
        "max_rho_rebuild_err": max(o["rho_rebuild_err"] for o in out),
        "rows_largest_component_wrap": {w: sum(1 for o in out if o["wrap_dims"][0] == w)
                                        for w in (0, 1, 2, 3)},
        "rows_with_wrap1_component": sum(1 for o in out if 1 in o["wrap_dims"]),
        "rows_with_wrap2_component": sum(1 for o in out if 2 in o["wrap_dims"]),
        "rows_with_island": sum(1 for o in out if 0 in o["wrap_dims"]),
        "max_island_solid_fraction": max(o["solid_in_islands"] / o["solid_total"] for o in out),
        "rows_several_wrapping_components": sum(
            1 for o in out if sum(w >= 1 for w in o["wrap_dims"]) > 1),
        "rows_node_links_not_face_links": sum(1 for o in out
                                              if o["n_face_components"] > o["n_components"]),
        "seconds": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps({"summary": summ, "rows": out}, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
