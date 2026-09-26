"""P1-B: detector behaviour on typed queries. No live API."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metagpt"))

from schema import REGISTRY, prompt_block  # noqa: E402
from llm import build_system  # noqa: E402
from retrieval import Catalogue  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "p1b_fault_eval.json"


def generated_prompt():
    return build_system()


def manual_equivalent_prompt():
    lines = ["Queryable properties (key, unit, kind):"]
    for k, p in REGISTRY.items():
        lines.append(f"- {k} [{p.unit}] ({p.kind})")
    return "\n".join(lines)


def keys_in_text(text):
    return {k for k in REGISTRY if k in text}


def registry_units():
    return {k: p.unit for k, p in REGISTRY.items()}


def compile_query(query, evaluator_keys, evaluator_units):
    """Typed-query evaluator under a (possibly faulty) contract."""
    detections = []
    compiled = []
    gold_units = registry_units()
    for c in query["constraints"]:
        key = c["property"]
        if key not in evaluator_keys:
            detections.append({"type": "undeclared_key", "key": key})
            continue
        q_unit = c.get("unit") or gold_units.get(key)
        e_unit = evaluator_units.get(key)
        if q_unit and e_unit and q_unit != e_unit:
            detections.append({"type": "unit_mismatch", "key": key,
                               "query_unit": q_unit, "evaluator_unit": e_unit})
        compiled.append({"property": key, "op": c["op"], "value": float(c["value"])})
    return detections, compiled


def run_search(cat, constraints):
    r = cat.search({"objectives": [], "constraints": constraints}, top_k=1)
    return {
        "n_feasible": int(r.n_feasible),
        "answered": bool(r.rows),
        "dropped": list(r.dropped or []),
    }


def metric_row(name, **fields):
    row = {"id": name}
    row.update(fields)
    return row


def run_tests():
    cat = Catalogue()
    gen = generated_prompt()
    man = manual_equivalent_prompt()
    all_keys = set(cat.keys)
    units = registry_units()
    tests = []

    # Smoke: prompt coverage (not detector behaviour).
    tests.append(metric_row(
        "smoke_generated_names_keys",
        kind="smoke",
        pass_=keys_in_text(gen) >= set(REGISTRY) and "E_mean" in prompt_block(),
        detail="generated prompt names every registry key",
    ))
    tests.append(metric_row(
        "smoke_manual_equivalent_keys",
        kind="smoke",
        pass_=keys_in_text(man) == set(REGISTRY),
        detail=f"manual {len(keys_in_text(man))} vs registry {len(REGISTRY)}",
    ))

    # Gold typed queries. E_mean >= 80 GPa is empty for many cells; pair with
    # a light bound so omitting E_mean can open a previously empty set.
    q_emean = {"constraints": [
        {"property": "E_mean", "op": ">=", "value": 250.0, "unit": units["E_mean"]},
        {"property": "rho", "op": "<=", "value": 0.20, "unit": units["rho"]},
    ]}
    q_rho_only = {"constraints": [
        {"property": "rho", "op": "<=", "value": 0.40, "unit": units["rho"]},
    ]}
    q_k = {"constraints": [
        {"property": "k_11", "op": ">=", "value": 80.0, "unit": units["k_11"]},
        {"property": "rho", "op": "<=", "value": 0.30, "unit": units["rho"]},
    ]}

    gold_emean = run_search(cat, q_emean["constraints"])
    gold_rho = run_search(cat, q_rho_only["constraints"])
    gold_k = run_search(cat, q_k["constraints"])

    # Omit E_mean from the evaluator. Query that uses it must be detected.
    keys_omit = all_keys - {"E_mean"}
    det, compiled = compile_query(q_emean, keys_omit, units)
    silent = run_search(cat, compiled)
    detected = any(d["type"] == "undeclared_key" and d["key"] == "E_mean" for d in det)
    false_accept = (not gold_emean["answered"]) and silent["answered"]
    feas_changed = gold_emean["n_feasible"] != silent["n_feasible"]
    tests.append(metric_row(
        "detect_omitted_key",
        kind="detector",
        pass_=detected,
        detail="query using E_mean is flagged when the evaluator omitted E_mean",
        detected=detected,
        gold_n_feasible=gold_emean["n_feasible"],
        silent_n_feasible=silent["n_feasible"],
        false_accept_if_silent_drop=false_accept,
        feasible_set_changed=feas_changed,
    ))
    tests.append(metric_row(
        "omission_changes_feasible_set",
        kind="detector",
        pass_=feas_changed,
        detail="dropping the omitted constraint changes n_feasible vs gold",
        gold_n_feasible=gold_emean["n_feasible"],
        silent_n_feasible=silent["n_feasible"],
    ))
    tests.append(metric_row(
        "omission_false_accept_if_silent",
        kind="detector",
        pass_=false_accept,
        detail="silent drop would return a row for a gold-empty query",
        gold_answered=gold_emean["answered"],
        silent_answered=silent["answered"],
    ))

    # False refusal: evaluator missing E_mean, query does not use E_mean.
    det_ok, compiled_ok = compile_query(q_rho_only, keys_omit, units)
    search_ok = run_search(cat, compiled_ok)
    false_refuse = bool(det_ok) or (gold_rho["n_feasible"] != search_ok["n_feasible"])
    tests.append(metric_row(
        "no_false_refusal_on_unused_omission",
        kind="detector",
        pass_=not false_refuse and gold_rho["answered"] and search_ok["answered"],
        detail="rho-only query is unchanged when the unused key E_mean is omitted",
        detections=det_ok,
        gold_n_feasible=gold_rho["n_feasible"],
        eval_n_feasible=search_ok["n_feasible"],
        false_refuse=false_refuse,
    ))

    # Unit-scale fault: evaluator believes k_11 is mW/(m K); query is W/(m K).
    bad_units = dict(units)
    bad_units["k_11"] = "mW/(m K)"
    det_u, compiled_u = compile_query(q_k, all_keys, bad_units)
    unit_detected = any(d["type"] == "unit_mismatch" and d["key"] == "k_11"
                        for d in det_u)
    # If the mismatch is ignored and the numeric bound is treated as mW,
    # the evaluator would search k_11 >= 0.080 instead of 80.
    scaled = [{"property": "k_11", "op": ">=", "value": 80.0 / 1000.0},
              {"property": "rho", "op": "<=", "value": 0.30}]
    scaled_search = run_search(cat, scaled)
    feas_unit = gold_k["n_feasible"] != scaled_search["n_feasible"]
    tests.append(metric_row(
        "detect_unit_scale_mismatch",
        kind="detector",
        pass_=unit_detected,
        detail="W/(m K) vs mW/(m K) on k_11 is a unit mismatch, not a spelling variant",
        detections=det_u,
    ))
    tests.append(metric_row(
        "unit_scale_changes_feasible_set",
        kind="detector",
        pass_=feas_unit,
        detail="interpreting 80 W/(m K) as 80 mW/(m K) changes n_feasible",
        gold_n_feasible=gold_k["n_feasible"],
        scaled_n_feasible=scaled_search["n_feasible"],
    ))

    # E_mean remains registered (documentation, not a detector rate).
    tests.append(metric_row(
        "historical_E_mean_still_registered",
        kind="smoke",
        pass_="E_mean" in REGISTRY,
        detail="E_mean remains registered, so an axis-free stiffness limit has a key",
    ))

    # v3: a constraint on an undeclared key is not dropped silently; the query
    # is held back as a reduced query and returns no row.
    gated = cat.search({"objectives": [{"property": "k_11", "sense": "max"}],
                        "constraints": [{"property": "yield_strength", "op": ">=",
                                         "value": 150}]})
    tests.append(metric_row(
        "undeclared_key_is_gated",
        kind="detector",
        pass_=gated.status == "gated_reduced" and not gated.rows,
        detail=f"status {gated.status}; rows returned {len(gated.rows)}",
    ))

    for t in tests:
        t["pass"] = bool(t.pop("pass_"))
    return tests


def main():
    tests = run_tests()
    detector = [t for t in tests if t.get("kind") == "detector"]
    failed = [t for t in tests if not t["pass"]]
    out = {
        "n": len(tests),
        "n_detector": len(detector),
        "n_detector_pass": sum(1 for t in detector if t["pass"]),
        "n_smoke": sum(1 for t in tests if t.get("kind") == "smoke"),
        "failed": failed,
        "tests": tests,
        "note": ("Detector rows measure detection, false acceptance under silent "
                 "drop, false refusal, and feasible-set change. Smoke rows are "
                 "prompt/registry construction checks."),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for t in tests:
        print(f"[{'PASS' if t['pass'] else 'FAIL'}] {t['kind']:8} {t['id']}  {t['detail']}")
    print(f"wrote {OUT}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
