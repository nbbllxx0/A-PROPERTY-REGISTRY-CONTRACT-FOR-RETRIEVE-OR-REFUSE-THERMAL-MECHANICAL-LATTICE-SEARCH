"""Every searchable row must rebuild its stored relative density.

Rebuilds each row's mask from family, mode, frequency, isovalue, grid and tie
rule (verify.rebuild) and compares the voxel fraction with the stored rho.

    python check_rebuild_rho.py      -> data/rebuild_rho_check.json
"""
import json
import pathlib
import sys

PAPER = pathlib.Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT / "metagpt"))

from verify import load_rows, rebuild  # noqa: E402


def main():
    rows = load_rows(ROOT / "metagpt" / "catalogue.csv")
    worst, bad = 0.0, []
    for r in rows:
        err = abs(float(rebuild(r).mean()) - float(r["rho"]))
        worst = max(worst, err)
        if err > 1e-6:
            bad.append({"uid": r["uid"], "err": err})
    out = {"n": len(rows), "worst_abs_err": worst, "n_over_1e-6": len(bad), "over": bad[:20]}
    (PAPER / "data" / "rebuild_rho_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "over"}))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
