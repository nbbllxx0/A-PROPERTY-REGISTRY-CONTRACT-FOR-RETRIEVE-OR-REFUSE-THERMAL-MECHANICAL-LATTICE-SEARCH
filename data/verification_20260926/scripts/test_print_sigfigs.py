"""Printer contract: printed bound cannot exclude the witness."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metagpt"))

from retrieval import (  # noqa: E402
    print_sigfigs_nearest, print_sigfigs_outward, printed_repair_query,
)


def admits(x, op, p):
    if op == "<=":
        return x <= p
    if op == "<":
        return x < p
    if op == ">=":
        return x >= p
    if op == ">":
        return x > p
    raise AssertionError(op)


def check(x, op, expected=None):
    p = print_sigfigs_outward(x, op)
    if not admits(x, op, p):
        raise AssertionError(f"{x} {op} printed {p} excludes the witness")
    if expected is not None and p != expected:
        raise AssertionError(f"{x} {op}: got {p}, expected {expected}")
    return p


def main():
    # Gold empty-query witness (positive, non-strict).
    check(0.32503, "<=", 0.326)
    check(0.32503, ">=", 0.325)
    check(36.5, ">=", 36.5)
    check(100.0, ">=", 100.0)

    # Strict operators must leave the endpoint.
    p = check(1.0, "<")
    if p <= 1.0:
        raise AssertionError(f"< 1.0 printed {p}")
    p = check(1.0, ">")
    if p >= 1.0:
        raise AssertionError(f"> 1.0 printed {p}")

    # Negative values: signed direction, not magnitude-then-restore.
    p = check(-1.2345, "<=")
    if not (-1.2345 <= p):
        raise AssertionError(p)
    p = check(-1.2345, ">=")
    if not (-1.2345 >= p):
        raise AssertionError(p)

    # Zero.
    check(0.0, "<=", 0.0)
    check(0.0, ">=", 0.0)
    if not (0.0 < print_sigfigs_outward(0.0, "<")):
        raise AssertionError("strict < 0")
    if not (0.0 > print_sigfigs_outward(0.0, ">")):
        raise AssertionError("strict > 0")

    # Retained original bounds are not re-rounded.
    orig = [
        {"property": "rho", "op": "<=", "value": 0.151234},
        {"property": "E_11", "op": ">=", "value": 100.0},
    ]
    repair = {
        "atoms": [{"property": "E_11", "op": ">=", "value": 100.0,
                   "reached": 36.5123}],
    }
    q = printed_repair_query(orig, repair, "outward")
    kept = [c for c in q["constraints"] if c["property"] == "rho"]
    if len(kept) != 1 or float(kept[0]["value"]) != 0.151234:
        raise AssertionError(f"retained bound was rewritten: {kept}")
    dropped = [c for c in q["constraints"] if c["property"] == "E_11"]
    if not dropped or not math.isclose(dropped[0]["value"],
                                       print_sigfigs_outward(36.5123, ">=")):
        raise AssertionError(dropped)

    # Nearest-even still exists as the pre-registered artefact.
    if print_sigfigs_nearest(0.32503) != 0.325:
        raise AssertionError("nearest .3g of gold rho")

    print("print_sigfigs: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
