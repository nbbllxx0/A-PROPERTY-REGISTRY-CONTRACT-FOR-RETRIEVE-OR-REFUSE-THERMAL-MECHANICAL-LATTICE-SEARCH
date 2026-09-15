"""Fold the void coefficients into the catalogue, keeping a backup.

B is not added to the property registry. It is not a design quantity anyone
searches on -- it is the coefficient that lets a stored row be re-evaluated for
a pore that is not empty. Putting it in the registry would change the search
vocabulary and the frozen benchmark for no gain.

Infeasible rows get empty cells rather than a number, because there is no cell
to solve.

    python merge_void_coefficients.py
"""
import csv
import io
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAT = os.path.join(HERE, "catalogue.csv")
VOID = os.path.join(HERE, "catalogue_void.csv")
BAK = os.path.join(os.path.dirname(HERE), "logs", "catalogue.pre-void.csv")
NEW = ["B11", "B22", "B33"]


def main():
    if not os.path.exists(VOID):
        print("missing %s -- run void_coefficients.py first" % VOID)
        return 1

    with io.open(CAT, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0])
    if all(n in fields for n in NEW):
        print("catalogue already carries %s" % ", ".join(NEW))
        return 0

    with io.open(VOID, encoding="utf-8", newline="") as f:
        B = {r["uid"]: r for r in csv.DictReader(f)}

    if not os.path.exists(BAK):
        shutil.copy2(CAT, BAK)
        print("backup written to %s" % os.path.relpath(BAK, os.path.dirname(HERE)))

    got = 0
    for r in rows:
        b = B.get(r["uid"])
        for n in NEW:
            r[n] = b[n] if b else ""
        got += bool(b)

    with io.open(CAT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + NEW)
        w.writeheader()
        w.writerows(rows)

    print("catalogue.csv: %d rows, %d carry a void coefficient, %d blank "
          "(infeasible)" % (len(rows), got, len(rows) - got))
    return 0


if __name__ == "__main__":
    sys.exit(main())
