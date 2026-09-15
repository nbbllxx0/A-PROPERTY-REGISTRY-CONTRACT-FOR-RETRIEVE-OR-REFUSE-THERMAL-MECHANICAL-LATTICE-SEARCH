"""
One-time repair: restore full-precision isovalues to an existing catalogue.

gen_dataset.py used to store `round(level, 6)`. That looks harmless and is not:
a TPMS level set evaluated on a symmetric voxel grid is massively degenerate --
large blocks of voxels hold one identical value -- so shifting the threshold by
5e-7 can flip hundreds of voxels together. Rebuilding a cell from the rounded
isovalue reached 3.9% error in relative density, which means the stored
properties did not belong to the cell the row described.

The geometry is still recoverable without re-simulating anything, because
bisection is deterministic: (family, mode, freq, n, rho_target) reproduces the
same isovalue it did during generation. This recomputes that isovalue at full
precision and rewrites only the `level` and `rho` columns.

Every row is checked, not sampled: the rebuilt mask must reproduce the stored
density exactly, or the row is reported and left untouched.

    python repair_levels.py --dry-run
    python repair_levels.py
"""

import argparse
import csv
import pathlib
import shutil
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "metahomog"))

from tpms import solid_at_density, solid_mask  # noqa: E402

from gen_dataset import build_jobs  # noqa: E402

# The stored rho_target was rounded to 5 decimals, and bisection stops on a
# 1e-4 tolerance, so re-bisecting from the rounded value can halt an iteration
# early or late and land on a different isovalue. build_jobs() is seeded and
# deterministic, so it still holds the exact target the generator used.
TARGETS = {uid: rho_t for uid, _f, _m, _q, rho_t in build_jobs()}

CAT = HERE / "catalogue.csv"

# `rho` was itself written as round(rho, 6), so a rebuilt mask can only ever be
# checked against it to that precision. Comparing an exact mask mean against a
# rounded record at machine tolerance rejects every correct row.
RHO_EPS = 1e-6


def repair(path=CAT, dry_run=False):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)

    fixed = exact = failed = skipped = 0
    bad = []
    for r in rows:
        if r.get("feasible") != "1":
            skipped += 1
            continue
        n = int(float(r["n"]))
        freq = tuple(int(c) for c in r["freq"])
        stored_rho = float(r["rho"])

        # Every feasible row is rewritten, including those whose rounded
        # isovalue happened to still rebuild the right cell. Leaving those
        # alone kept them at 6-decimal precision, which then failed
        # verification on density alone while every physical property agreed to
        # 1e-14 -- a tolerance artefact that looks exactly like a data fault.
        # Uniform precision is worth more than a skipped bisection.
        m = solid_mask(r["family"], float(r["level"]), n=n, freq=freq,
                       mode=r["mode"])
        if abs(float(m.mean()) - stored_rho) <= RHO_EPS:
            exact += 1

        # recover the isovalue the generator actually used
        target = TARGETS.get(int(float(r["uid"])), float(r["rho_target"]))
        _, level, rho = solid_at_density(r["family"], target,
                                         n=n, freq=freq, mode=r["mode"])
        m2 = solid_mask(r["family"], level, n=n, freq=freq, mode=r["mode"])
        if abs(float(m2.mean()) - stored_rho) > RHO_EPS:
            failed += 1
            bad.append((r["uid"], r["family"], r["mode"], r["freq"],
                        stored_rho, float(m2.mean())))
            continue

        r["level"] = repr(float(level))
        r["rho"] = repr(float(rho))
        fixed += 1

    print(f"rows: {len(rows)}   infeasible skipped: {skipped}")
    print(f"  isovalue was already sufficient : {exact}")
    print(f"  rewritten at full precision     : {fixed}")
    print(f"  unrecoverable : {failed}")
    for b in bad[:10]:
        print(f"      uid {b[0]} {b[1]}/{b[2]} f{b[3]}  stored {b[4]:.6f} "
              f"vs rebuilt {b[5]:.6f}")

    if dry_run:
        print("\ndry run -- nothing written")
        return failed

    if fixed:
        backup = path.with_suffix(".prerepair.csv")
        if not backup.exists():
            shutil.copy2(path, backup)
            print(f"\nbacked up to {backup.name}")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"rewrote {path.name}")
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--catalogue", default=None)
    a = ap.parse_args()
    return 1 if repair(pathlib.Path(a.catalogue or CAT), a.dry_run) else 0


if __name__ == "__main__":
    sys.exit(main())
