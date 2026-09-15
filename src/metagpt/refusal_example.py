"""Produce a real refusal, so the reply can show one instead of describing it.

The parse stage is bypassed and the query built directly, so this needs no
language model, no API key and no network. The brief is the impossible one from
the demo set: relative density under 0.15 with stiffness above 100 GPa.

    python refusal_example.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval import Catalogue, constraint_label            # noqa: E402

QUERY = {
    "objectives": [],
    "constraints": [{"property": "rho", "op": "<=", "value": 0.15},
                    {"property": "E_11", "op": ">=", "value": 100.0}],
    "material_filter": {},
    "unmet": [], "_rejected": [],
}


def main():
    cat = Catalogue()
    r = cat.search(QUERY, top_k=5)

    print('request: relative density under 0.15 with stiffness above 100 GPa')
    print('candidates considered : %d' % r.n_considered)
    print('feasible              : %d' % r.n_feasible)
    print('rows returned         : %d' % len(r.rows))
    print()

    if r.rows:
        print('NOT a refusal -- this query is satisfiable, pick a harder one')
        for row in r.rows[:3]:
            print('   %s' % {k: row[k] for k in list(row)[:5]})
        return 1

    print('what it says instead of a nearest row')
    print('  reason   : %s' % (r.rejected_reason or '-'))
    def fmt(x):
        if isinstance(x, dict):
            return constraint_label(x)
        if isinstance(x, (list, tuple)):
            return ' AND '.join(fmt(y) for y in x)
        return str(x)

    for i, m in enumerate(r.mus, 1):
        print('  MUS %d    : %s' % (i, fmt(m)))
    for i, m in enumerate(r.mcs, 1):
        print('  fix %d    : relax %s' % (i, fmt(m)))
    for i, m in enumerate(r.min_mcs, 1):
        print('  smallest : relax %s' % fmt(m))
    for rep in r.repairs:
        print('  slack    : %s' % rep)
    if r.relaxation:
        print('  suggested: %s' % r.relaxation)

    print()
    print('Nothing above is a catalogue row. The system did not return the')
    print('nearest thing and let the reader assume it was an answer.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
