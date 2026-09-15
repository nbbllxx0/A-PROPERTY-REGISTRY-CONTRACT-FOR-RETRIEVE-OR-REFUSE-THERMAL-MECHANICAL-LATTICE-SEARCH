"""
End to end: a sentence in, a design out -- or an explained refusal.

    python app.py "light, stiff along the length, spreads heat sideways"
    python app.py --demo          # run the standing demo set, write figures

The pipeline is deliberately four separable stages, so a bad answer can always
be attributed to one of them rather than to "the AI":

    text --[LLM]--> query --[search]--> candidates --[physics]--> verified

Only the first stage is learned. The search is arithmetic over precomputed
homogenisation results and can be checked by hand; the last stage is not a
lookup at all -- every returned design is rebuilt from its catalogue row and
re-solved by a solver that shares no code with the one that filled the
catalogue. That runs by default, because a design whose numbers were only
retrieved is a claim rather than a result.

    python app.py "..."               # verifies the top candidate
    python app.py "..." --no-verify   # skip it
"""

import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent

from llm import parse, summarise  # noqa: E402
from retrieval import Catalogue  # noqa: E402
from schema import REGISTRY  # noqa: E402

DEMOS = [
    "light and stiff along the length, and it should spread heat sideways "
    "while insulating upward",
    "a printable heat spreader for a chip: conduct as well as possible, "
    "stay under 3 dollars a kilo",
    "as light as possible while staying above 1 GPa stiffness",
    "insulating in every direction but still load bearing",
    "relative density under 0.15 with stiffness above 100 GPa",
]


def show(cat, request, top_k=5, make_fig=False, tag="demo", verify_top=0):
    t0 = time.time()
    q = parse(request)
    t_parse = time.time() - t0

    t0 = time.time()
    r = cat.search(q, top_k=top_k)
    t_search = time.time() - t0

    print("\n" + "=" * 76)
    print(f'REQUEST   "{request}"')
    print("-" * 76)
    print(f"UNDERSTOOD AS  {summarise(q)}")
    if q.get("unmet"):
        print(f"CANNOT EXPRESS {'; '.join(q['unmet'])}")
    if q.get("_rejected"):
        print(f"DISCARDED      {'; '.join(q['_rejected'])}")
    if r.dropped:
        print(f"NOT APPLIED    {'; '.join(r.dropped)}")
    if r.caveats:
        for c in r.caveats:
            print(f"ESTIMATE ONLY  {c}")
    if r.unranked:
        print("UNRANKED       no objective could be searched, so the order "
              "below is arbitrary rather than best-first")
    print(f"               parse {t_parse:.2f}s | search {t_search*1000:.0f}ms | "
          f"{r.n_considered:,} combinations considered")
    print("-" * 76)

    if not r.rows:
        print("NO CATALOGUE ROW SATISFIES THIS")
        print(f"  {r.rejected_reason}")
        if r.mus:
            print("  MUS: " + "; ".join("{" + ", ".join(u) + "}" for u in r.mus))
        if r.min_mcs:
            print("  min MCS: " + "; ".join("{" + ", ".join(h) + "}" for h in r.min_mcs))
        extra = [h for h in (r.mcs or []) if h not in (r.min_mcs or [])]
        if extra:
            print("  other MCS: " + "; ".join("{" + ", ".join(h) + "}" for h in extra))
        if r.relaxation:
            print(f"  closest achievable: {r.relaxation}")
    else:
        print(f"{r.n_feasible:,} qualify; {r.pareto_size} are genuine "
              f"alternatives rather than also-rans\n")
        hdr = f"{'#':>2} {'material':19s}{'shape':22s}{'rho':>6s}"
        keys = [k for k in ("k_11", "k_33", "E_11", "mass_density",
                            "cost_per_kg") if k in cat.col]
        for k in keys:
            hdr += f"{k:>14s}"
        print(hdr)
        for i, row in enumerate(r.rows, 1):
            shape = f"{row['family']}/{row['mode'][:3]} f{row['freq']}"
            line = f"{i:>2} {row['material']:19s}{shape:22s}{row['rho']:6.2f}"
            for k in keys:
                line += f"{row[k]:14.2f}"
            print(line)
        u = "  units: " + ", ".join(
            f"{k} {REGISTRY[k].unit}" for k in keys if REGISTRY[k].unit != "-")
        print(u)

    if verify_top and r.rows:
        from verify import verify_row, format_report
        print("\nVERIFICATION  the top candidate re-simulated from its row, "
              "by an independent solver")
        for row in r.rows[:verify_top]:
            print("  " + format_report(verify_row(row["_geom"])))

    if make_fig:
        from viz import result_card
        p = result_card(cat, q, r, fname=f"fig_result_{tag}.png", request=request)
        print(f"\n  figure: {p.name}")
    return q, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("request", nargs="*")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--catalogue", default=None)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--fig", action="store_true")
    # Re-solving is on by default. A returned design whose properties were only
    # looked up is a claim; one that has been rebuilt from its row and re-solved
    # by an independent solver is a result, and the difference costs a few
    # seconds. Making the honest path opt-out rather than opt-in is the whole
    # point of having built it.
    ap.add_argument("--verify", type=int, default=1,
                    metavar="N", help="re-simulate the top N results "
                                      "(default 1; 0 disables)")
    ap.add_argument("--no-verify", action="store_const", const=0, dest="verify",
                    help="skip re-simulation")
    a = ap.parse_args()

    path = a.catalogue or (HERE / "catalogue.csv")
    if not pathlib.Path(path).exists():
        sys.exit(f"no catalogue at {path} -- run gen_dataset.py first")
    cat = Catalogue(csv_path=path)
    s = cat.stats()
    print(f"catalogue: {s['geometries']:,} geometries x {s['materials']} "
          f"materials = {s['combinations']:,} combinations, "
          f"{s['properties']} properties each")

    if a.demo:
        for i, d in enumerate(DEMOS, 1):
            show(cat, d, top_k=a.top, make_fig=a.fig, tag=f"{i:02d}", verify_top=a.verify)
    else:
        req = " ".join(a.request) or DEMOS[0]
        show(cat, req, top_k=a.top, make_fig=a.fig, tag="one", verify_top=a.verify)


if __name__ == "__main__":
    main()
