"""Sampled GPU vs CPU catalogue check. Writes JSON; does not touch metagpt/.

Rows the GPU built (builder = 'gpu') are checked on the CPU solver, which is
slow at fine grids, so those checks run in a process pool (largest grid
first). The other rows are checked on the GPU in this process.
"""
import json
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metagpt"))

from verify import load_rows, verify_row  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "verify_sample60.json"
WORKERS = 6


def _record(r, elastic):
    rep = verify_row(r, prefer_gpu=True, elastic=elastic)
    return {
        "uid": rep["uid"],
        "backend": rep["backend"],
        "independent": rep["independent"],
        "ok": bool(rep["ok"]),
        "worst": float(rep["worst"]),
        "seconds": float(rep["seconds"]),
        "checks": [
            {"key": a, "stored": b, "recomputed": c, "err": d, "tol": e}
            for a, b, c, d, e in rep["checks"]
        ],
    }


def _write(recs, seed, elastic):
    done = [x for x in recs if x is not None]
    summary = {
        "n": len(done),
        "seed": seed,
        "elastic": elastic,
        "n_ok": sum(x["ok"] for x in done),
        "worst": max(x["worst"] for x in done),
        "median_worst": float(np.median([x["worst"] for x in done])),
        "mean_seconds": float(np.mean([x["seconds"] for x in done])),
        "backends": sorted({x["backend"] for x in done}),
        "independent": all(x["independent"] for x in done),
        "rows": done,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(n=60, seed=0, elastic=True):
    rows = load_rows(ROOT / "metagpt" / "catalogue.csv")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    chosen = [rows[i] for i in sorted(pick)]
    recs = [None] * len(chosen)
    cpu = [k for k, r in enumerate(chosen) if (r.get("builder") or "cpu") == "gpu"]
    cpu.sort(key=lambda k: -int(chosen[k]["n"]))
    t0 = time.perf_counter()

    def report(k):
        x = recs[k]
        print(f"{sum(v is not None for v in recs)}/{len(chosen)} uid {x['uid']} "
              f"worst {x['worst']:.2e} {x['backend']} {x['seconds']:.1f}s", flush=True)

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(_record, chosen[k], elastic): k for k in cpu}
        for k, r in enumerate(chosen):
            if k in futs.values():
                continue
            recs[k] = _record(r, elastic)
            report(k)
            _write(recs, seed, elastic)
        for f in as_completed(futs):
            k = futs[f]
            recs[k] = f.result()
            report(k)
            _write(recs, seed, elastic)
    summary = _write(recs, seed, elastic)
    print(f"done {summary['n_ok']}/{summary['n']} worst {summary['worst']:.2e} "
          f"in {time.perf_counter()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
