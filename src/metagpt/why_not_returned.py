"""Why stainless 316L and Ti-6Al-4V appear in no result.

The void-cost table shows the two largest air corrections on these two
materials, and says no result returns them. This prints the reason rather than
asserting it: for each worked brief, which material every policy returns, and
for each of the two, whether it has a feasible row at all and how far its best
row falls behind the winner. No language model, no API key, no network.

    python why_not_returned.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'paper_aei', 'scripts'))

from retrieval import Catalogue, constraint_label                     # noqa: E402
from aei_upgrade_analyses import (nearest_neighbour, penalty_search,   # noqa: E402
                                  min_repair_search)

BRIEFS = [
    ('cheap conductor', 'k_11',
     [{"property": "k_11", "sense": "max", "weight": 1.0}],
     [{"property": "cost_per_kg", "op": "<=", "value": 3.0}]),
    ('heat spreader', 'k_11',
     [{"property": "k_11", "sense": "max", "weight": 1.0}],
     [{"property": "k_aniso", "op": "<=", "value": 0.70},
      {"property": "cost_per_kg", "op": "<=", "value": 5.0},
      {"property": "rho", "op": "<=", "value": 0.40}]),
    ('cost-capped light part', 'specific_stiffness',
     [{"property": "specific_stiffness", "sense": "max", "weight": 1.0}],
     [{"property": "rho", "op": "<=", "value": 0.25},
      {"property": "cost_per_kg", "op": "<=", "value": 10.0}]),
    ('light and stiff (refused)', 'E_11', [],
     [{"property": "rho", "op": "<=", "value": 0.15},
      {"property": "E_11", "op": ">=", "value": 100.0}]),
]
WATCH = ('stainless 316L', 'Ti-6Al-4V')


def query(objs, cons):
    return {"objectives": objs, "constraints": cons, "material_filter": {},
            "unmet": [], "_rejected": []}


def main():
    cat = Catalogue()
    names = np.array([cat.materials[i].name for i in cat.mi])
    props = {m.name: m for m in cat.materials}

    print('what every policy returns on the four worked briefs')
    print('   %-28s %-16s %-16s %-16s %s' % ('brief', 'retrieve-or-refuse',
                                             'min-repair', 'nearest', 'penalty'))
    for name, key, objs, cons in BRIEFS:
        q = query(objs, cons)
        r = cat.search(q, top_k=1)
        ours = r.rows[0]['material'] if r.rows else 'refuses'
        mr, _, _ = min_repair_search(cat, q)
        nn = nearest_neighbour(cat, cons, objs)
        pen = penalty_search(cat, cons, objs or (
            [{"property": "E_11", "sense": "max", "weight": 1.0}] if cons else []))
        print('   %-28s %-16s %-16s %-16s %s'
              % (name, ours, mr['material'] if mr else '-',
                 nn['material'], pen['material']))

    print()
    print('why the two poorest conductors never win')
    for w in WATCH:
        m = props[w]
        print('  %s: k = %g W/mK, E = %g GPa, cost = %g $/kg' % (w, m.k, m.E, m.cost))
        for name, key, objs, cons in BRIEFS:
            masks = cat._mask_constraints(cons)
            feas = np.ones(cat.M.shape[0], dtype=bool)
            for mk in masks.values():
                feas &= mk
            mine = names == w
            if not (feas & mine).any():
                killers = [constraint_label(c) for n, c in enumerate(cons)
                           if not (masks[n] & mine).any()]
                print('     %-28s no feasible row: every %s row fails %s'
                      % (name, w, ' and '.join(killers) or 'the combination'))
                continue
            col = cat.col[key]
            best_mine = np.nanmax(cat.M[feas & mine, col])
            best_all = np.nanmax(cat.M[feas, col])
            r = cat.search(query(objs, cons), top_k=1)
            winner = r.rows[0]['material'] if r.rows else '-'
            print('     %-28s best %s row: %s = %.3g; winner %s has %.3g  (%.1fx)'
                  % (name, w, key, best_mine, winner, best_all,
                     best_all / best_mine))

    print()
    print('The air correction scales with k_air / k_solid, so it is largest on')
    print('the poorest conductors. The briefs maximise conductivity or stiffness')
    print('per mass under a cost cap, so the poorest conductors are exactly the')
    print('rows that lose. The two facts are the same fact.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
