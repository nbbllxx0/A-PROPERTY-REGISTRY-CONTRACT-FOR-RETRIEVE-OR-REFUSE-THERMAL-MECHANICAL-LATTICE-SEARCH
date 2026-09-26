"""End-to-end checks of the registry contract and the fail-closed search.

Runs without a language model: the parse step is exercised through
llm.validate on hand-written model outputs.

    python test_contract.py        (or: pytest test_contract.py)
"""
import sys
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "metahomog"))

from retrieval import Catalogue, CATEGORICAL  # noqa: E402
from llm import validate, QUERY_SCHEMA  # noqa: E402
from schema import REGISTRY  # noqa: E402

CAT = Catalogue()


def _q(**kw):
    q = {"objectives": [], "constraints": []}
    q.update(kw)
    return q


def test_every_registry_key_is_searchable():
    for k, p in REGISTRY.items():
        assert k in CAT.col or k in CATEGORICAL, k


def test_categorical_symmetry_constraint():
    r = CAT.search(_q(constraints=[{"property": "symmetry", "op": "==",
                                    "value": "cubic"}]), top_k=50)
    assert r.status == "answered"
    n_cubic = sum(g["sym"] == "cubic" for g in CAT.geoms)
    assert r.n_feasible == n_cubic * len(CAT.materials)
    assert all(row["symmetry"] == "cubic" for row in r.rows)


def test_categorical_printable_matches_material_filter():
    a = CAT.search(_q(constraints=[{"property": "printable", "op": "==",
                                    "value": "true"}]))
    b = CAT.search(_q(material_filter={"printable_only": True}))
    assert a.status == b.status == "answered"
    assert a.n_feasible == b.n_feasible


def test_cubic_cells_cannot_pass_a_directional_bound():
    r = CAT.search(_q(constraints=[
        {"property": "symmetry", "op": "==", "value": "cubic"},
        {"property": "k_aniso", "op": "<=", "value": 0.90}]))
    assert r.status == "refused_empty", r.status
    assert r.mus, "the refusal must name its conflict"


def test_categorical_conflict_diagnosis_and_printed_repair():
    from retrieval import printed_repair_query
    q = _q(constraints=[{"property": "symmetry", "op": "==", "value": "cubic"},
                        {"property": "k_aniso", "op": "<=", "value": 0.90}])
    r = CAT.search(q)
    assert r.status == "refused_empty"
    assert any("symmetry" in a for m in r.mus for a in m)
    assert "dropped" in r.relaxation
    cons = [dict(c) for c in q["constraints"]]
    for rep in r.repairs:
        pq = printed_repair_query(cons, rep)
        assert CAT.search(pq).status in ("answered",), pq


def test_no_content_is_refused():
    r = CAT.search(_q())
    assert r.status == "refused_no_content" and not r.rows
    r = CAT.search(_q(unmet=["forced-convection cooling"]))
    assert r.status == "refused_no_content" and not r.rows
    assert any("forced-convection" in x for x in r.lost)


def test_vacuous_only_is_refused():
    r = CAT.search(_q(constraints=[{"property": "cte", "op": ">=", "value": 0}]))
    assert r.status == "refused_no_content" and not r.rows


def test_lost_content_is_gated_unless_accepted():
    q = _q(objectives=[{"property": "k_11", "sense": "max"}],
           unmet=["yield strength above 200 MPa"])
    r = CAT.search(q)
    assert r.status == "gated_reduced" and not r.rows and r.n_feasible > 0
    r = CAT.search(q, accept_reduced=True)
    assert r.status == "answered_reduced" and r.rows


def test_undeclared_key_is_lost_not_ignored():
    r = CAT.search(_q(objectives=[{"property": "k_11", "sense": "max"}],
                      constraints=[{"property": "yield_strength", "op": ">=",
                                    "value": 200}]))
    assert r.status == "gated_reduced"


def test_constraints_without_objective_are_flagged_unranked():
    r = CAT.search(_q(constraints=[{"property": "rho", "op": "<=", "value": 0.3}]))
    assert r.status == "answered" and r.unranked


def test_validate_folds_categorical_field():
    q = validate({"objectives": [], "constraints": [],
                  "categorical_constraints": [
                      {"property": "symmetry", "value": "Cubic"},
                      {"property": "printable", "value": "maybe"}]})
    assert {"property": "symmetry", "op": "==", "value": "cubic"} in q["constraints"]
    assert len(q["_lost"]) == 1 and "printable" in q["_lost"][0]


def test_validate_rejects_numeric_value_on_categorical_key():
    q = validate({"objectives": [], "constraints": [
        {"property": "symmetry", "op": "<=", "value": 1.0}]})
    assert not q["constraints"] and q["_lost"]


def test_schema_offers_every_categorical_key():
    item = QUERY_SCHEMA["properties"]["categorical_constraints"]["items"]
    assert set(item["properties"]["property"]["enum"]) == set(CATEGORICAL)


def test_gold_refusal_unchanged():
    r = CAT.search(_q(constraints=[{"property": "rho", "op": "<=", "value": 0.15},
                                   {"property": "E_11", "op": ">=", "value": 100}]))
    assert r.status == "refused_empty"
    assert len(r.mus) == 1 and len(r.mus[0]) == 2


def test_numeric_equality_is_judged_at_three_significant_figures():
    from retrieval import equal_at_sigfigs, equality_tolerance, print_sigfigs_outward
    assert abs(equality_tolerance(0.60) - 0.0005) < 1e-15
    assert abs(equality_tolerance(100.0) - 0.5) < 1e-12
    assert equal_at_sigfigs(0.59991, 0.60) and not equal_at_sigfigs(0.5985, 0.60)
    assert print_sigfigs_outward(0.58057, "==") == 0.581
    # numpy's isclose default refused this although a row holds 0.5999
    r = CAT.search(_q(constraints=[{"property": "porosity", "op": "==", "value": 0.60}]),
                   top_k=1)
    assert r.status == "answered" and equal_at_sigfigs(r.rows[0]["porosity"], 0.60)


def test_equality_repair_is_two_sided():
    rho = [g["rho"] for g in CAT.geoms]
    for target, nearest in ((0.60, max(rho)), (0.05, min(rho))):
        r = CAT.search(_q(constraints=[{"property": "rho", "op": "==", "value": target}]))
        assert r.status == "refused_empty"
        assert abs(r.repairs[0]["atoms"][0]["reached"] - nearest) < 1e-12


def test_printable_implies_one_face_connected_solid():
    from retrieval import PRINTABLE_CONN
    for q in (_q(objectives=[{"property": "k_11", "sense": "max"}],
                 material_filter={"printable_only": True}),
              _q(objectives=[{"property": "k_11", "sense": "max"}],
                 constraints=[{"property": "printable", "op": "==", "value": "true"}])):
        r = CAT.search(q, top_k=20)
        assert r.status == "answered" and r.implied
        assert all(row["conn_frac"] >= PRINTABLE_CONN for row in r.rows)
    # a stated connectivity bound is the user's, not overridden
    r = CAT.search(_q(constraints=[{"property": "conn_frac", "op": ">=", "value": 0.5}],
                      material_filter={"printable_only": True}))
    assert not r.implied


def test_unknown_material_name_is_lost_not_applied():
    q = _q(objectives=[{"property": "k_11", "sense": "max"}],
           material_filter={"allowed": ["aluminum"]})
    r = CAT.search(q)
    assert r.status == "gated_reduced" and not r.rows
    assert any("aluminum" in x for x in r.lost)
    r = CAT.search(_q(material_filter={"allowed": ["unobtainium"]}))
    assert r.status == "refused_no_content"
    r = CAT.search(_q(objectives=[{"property": "k_11", "sense": "max"}],
                      material_filter={"allowed": ["Copper"]}), top_k=1)
    assert r.status == "answered" and r.rows[0]["material"] == "copper"


def test_refusal_carries_flip_margin():
    r = CAT.search(_q(constraints=[{"property": "rho", "op": "<=", "value": 0.15},
                                   {"property": "E_11", "op": ">=", "value": 100}]))
    assert r.status == "refused_empty" and 1.5 < r.flip_margin < 1.7, r.flip_margin
    # a cubic cell cannot reach k33/k11 <= 0.90 under any mesh error: exact by symmetry
    r = CAT.search(_q(constraints=[{"property": "symmetry", "op": "==", "value": "cubic"},
                                   {"property": "k_aniso", "op": "<=", "value": 0.90}]))
    assert r.status == "refused_empty" and r.flip_margin == float("inf")
    r = CAT.search(_q(objectives=[{"property": "k_11", "sense": "max"}]), top_k=1)
    assert r.flip_margin is None


def test_search_identity_rule_matches_solver_rule():
    from retrieval import identity_pairs
    from resolve_symmetry_rows import identity_pairs as solver_pairs
    for g in CAT.geoms:
        assert identity_pairs(g["sym"], g["freq"]) == solver_pairs(g["sym"], g["freq"]), g["uid"]


def test_symmetry_identities_hold_in_catalogue():
    worst = 0.0
    for g in CAT.geoms:
        f = g["freq"]
        pairs = ([(1, 2), (1, 3)] if g["sym"] == "cubic" else
                 [(1, 2)] if f[0] == f[1] else [(2, 3)] if f[1] == f[2] else
                 [(1, 3)] if f[0] == f[2] else [])
        for i, j in pairs:
            for q in ("k", "E"):
                worst = max(worst, abs(g[f"{q}{j}{j}"] / g[f"{q}{i}{i}"] - 1))
    assert worst <= 0.01, worst


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                fails += 1
                print("FAIL", name, e)
    print(f"{fails} failure(s)")
    sys.exit(1 if fails else 0)
