"""Show what a successful request returns, and what the 1% nudge does to it.

The parse stage is bypassed and the two gold briefs built directly, so this
needs no language model, no API key and no network. It prints the rows that
come back, then one nudged draw so the mechanism is visible, then the forty
draws the paper reports.

    python rank_stability_example.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval import Catalogue                                 # noqa: E402

CHEAP = {"objectives": [{"property": "k_11", "sense": "max", "weight": 1.0}],
         "constraints": [{"property": "cost_per_kg", "op": "<=", "value": 3.0}],
         "material_filter": {}, "unmet": [], "_rejected": []}
SPREADER = {"objectives": [{"property": "k_11", "sense": "max", "weight": 1.0}],
            "constraints": [{"property": "k_aniso", "op": "<=", "value": 0.70},
                            {"property": "cost_per_kg", "op": "<=", "value": 5.0},
                            {"property": "rho", "op": "<=", "value": 0.40}],
            "material_filter": {}, "unmet": [], "_rejected": []}
REFUSAL = {"objectives": [],
           "constraints": [{"property": "rho", "op": "<=", "value": 0.15},
                           {"property": "E_11", "op": ">=", "value": 100.0}],
           "material_filter": {}, "unmet": [], "_rejected": []}
KEYS = ("k_11", "E_11")
FIELDS = [k for k in ("material", "family", "mode", "freq", "symmetry", "rho",
                      "level", "n", "uid", "porosity", "k_aniso", "E_aniso",
                      "cost_per_kg", "cte", "tmax", "k_11", "k_22", "k_33",
                      "k_mean", "E_11", "E_22", "E_33", "E_mean", "mass_density",
                      "specific_stiffness", "specific_conductivity",
                      "cost_per_m3", "permeability", "min_feature")]


def ident(row):
    return (row["family"], tuple(row["freq"]), row["material"],
            round(row["rho"], 4))


def show(rows, mark=None):
    for i, row in enumerate(rows, 1):
        flag = '  <- the unnudged winner' if mark and ident(row) == mark else ''
        print('   %2d  %-15s %-10s %-8s f=%s  rho %.3f  k11 %6.2f  E11 %5.1f  '
              '$/kg %.1f%s' % (i, row["material"], row["family"], row["mode"],
                              row["freq"], row["rho"], row["k_11"],
                              row["E_11"], row["cost_per_kg"], flag))


def nudge(cat, saved, rng, rel):
    for k in KEYS:
        col = cat.col[k]
        cat.M[:, col] = saved[k] * (1 + rel * rng.normal(size=saved[k].shape))


def restore(cat, saved):
    for k in KEYS:
        cat.M[:, cat.col[k]] = saved[k]


def draws(cat, query, rel, n_draw=40, seed=0, top_k=10):
    """Same procedure as paper_aei/scripts/aei_redteam_followup.py."""
    rng = np.random.default_rng(seed)
    saved = {k: cat.M[:, cat.col[k]].copy() for k in KEYS}
    base = cat.search(query, top_k=top_k)
    top0 = ident(base.rows[0]) if base.rows else None
    changed = stayed = flipped = 0
    for _ in range(n_draw):
        nudge(cat, saved, rng, rel)
        r = cat.search(query, top_k=top_k)
        if bool(r.rows) != bool(base.rows):
            flipped += 1
            continue
        if r.rows:
            ids = [ident(row) for row in r.rows]
            changed += ids[0] != top0
            stayed += top0 in ids
    restore(cat, saved)
    return base, changed, stayed, flipped


def main():
    cat = Catalogue()

    print('brief: the cheapest good conductor -- maximise k11, cost <= $3/kg')
    r = cat.search(CHEAP, top_k=5)
    print('candidates considered : %d' % r.n_considered)
    print('feasible              : %d' % r.n_feasible)
    print('rows returned         : %d  (the answer is row 1; rows 2-5 are '
          'context)' % len(r.rows))
    print('fields on each row    : %d stored' % len(FIELDS))
    print()
    show(r.rows)
    top, fifth = r.rows[0]["k_11"], r.rows[-1]["k_11"]
    print()
    print('   spread from row 1 to row 5 : %.2f W/mK on %.2f, i.e. %.1f%%'
          % (top - fifth, top, 100 * (top - fifth) / top))
    print('   -- a 1% nudge is the size of that gap, which is why the first '
          'row moves')

    print()
    print('one nudged draw: every stored k11 and E11 x (1 + 0.01 x N(0,1)), '
          'each row its own factor')
    rng = np.random.default_rng(0)
    saved = {k: cat.M[:, cat.col[k]].copy() for k in KEYS}
    nudge(cat, saved, rng, 0.01)
    r1 = cat.search(CHEAP, top_k=10)
    show(r1.rows, mark=ident(r.rows[0]))
    restore(cat, saved)

    print()
    print('forty fresh draws, the paper\'s procedure (seed 0)')
    print('   %-32s %-18s %-24s %s' % ('brief', 'first row changed',
                                       'winner still in top 10', 'answer/refuse flipped'))
    for name, q in (('cheapest good conductor', CHEAP),
                    ('heat spreader', SPREADER)):
        base, changed, stayed, flipped = draws(cat, q, 0.01)
        print('   %-32s %-18s %-24s %s' % (name, '%d of 40' % changed,
                                           '%d of 40' % stayed,
                                           '%d of 40' % flipped))
    for rel in (0.01, 0.05):
        base, changed, stayed, flipped = draws(cat, REFUSAL, rel)
        print('   %-32s %-18s %-24s %s' % ('light-and-stiff (a refusal), %d%%'
                                           % round(100 * rel), '-', '-',
                                           '%d of 40' % flipped))

    print()
    print('The first row is a member of a band of near-equals. The band holds; '
          'the decision to answer or refuse holds. The first row alone does not.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
