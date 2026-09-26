"""Baseline: a tool-using language model over the same table.

The practitioner alternative to a registry-compiled search is to hand a
capable model the design table and a query tool. This script does that, as a
fair baseline:
  * the same model family the paper deploys (Gemini 3.5 Flash, temperature 0);
  * the same 26,543-row product, loaded into a read-only SQLite table whose
    columns carry the registry's labels and units;
  * requirements given with their column names, so parsing is not tested --
    only whether the model returns a row that meets every requirement, refuses
    when none does, and names a conflict that is actually infeasible;
  * an explicit instruction to return a row only if it meets every
    requirement, and to say so when a requirement cannot be checked.

Three sets:
  suite      the frozen 64-query typed suite (48 feasible, 16 empty)
  empty216   the 216 frozen empty queries of the printed-repair study
  undeclared 12 requests that add a requirement the table cannot check

    python agent_baseline.py suite            # then empty216, undeclared
    python agent_baseline.py score            # writes data/agent_baseline.json
"""
import itertools
import json
import pathlib
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request

import numpy as np

PAPER = pathlib.Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT / "metagpt"))
sys.path.insert(0, str(PAPER / "scripts"))

from retrieval import Catalogue, CATEGORICAL  # noqa: E402
from schema import REGISTRY  # noqa: E402
import llm  # noqa: E402

MODEL = "gemini-3.5-flash"
OUTDIR = PAPER / "data" / "agent_baseline"
OUTDIR.mkdir(parents=True, exist_ok=True)
MAX_CALLS = 8  # tool turns; the turn after them must answer
BUDGET_MSG = ("The query budget is used up. Give your final JSON object now, "
              "based on the results you have seen.")
# The last turn is decoded against this schema, so it always yields a decision.
ANSWER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "decision": {"type": "STRING", "enum": ["row", "none"]},
        "id": {"type": "INTEGER", "nullable": True},
        "conflicting_requirements": {"type": "ARRAY", "items": {"type": "INTEGER"}},
        "unverifiable_requirements": {"type": "ARRAY", "items": {"type": "INTEGER"}},
        "suggested_change": {"type": "STRING"},
    },
    "required": ["decision", "conflicting_requirements", "unverifiable_requirements",
                 "suggested_change"],
}
OPW = {"<=": "at most", "<": "below", ">=": "at least", ">": "above", "==": "equal to"}

SYSTEM = """You help an engineer choose a porous metal lattice design from a table.
The table `designs` has one row per (cell geometry, metal) combination. Use the
run_sql tool (SQLite, read-only) to inspect it. Columns:
{columns}

Rules:
- Return a row only if it satisfies EVERY numbered requirement. Check this with
  a query before you answer.
- If no row satisfies all requirements, do not return a row. Say which
  requirements conflict (their numbers) and how far one would have to move.
- If a requirement refers to a quantity the table does not contain, you cannot
  verify it: list its number under unverifiable_requirements and do not return a
  row that claims to satisfy it.
- Finish with one JSON object and nothing after it:
  {{"decision": "row" or "none", "id": <row id or null>,
    "conflicting_requirements": [<numbers>],
    "unverifiable_requirements": [<numbers>],
    "suggested_change": "<text>"}}"""


def build_db(cat):
    keys = list(cat.keys)
    con = sqlite3.connect(":memory:")
    cols = ["id INTEGER", "material TEXT", "family TEXT", "mode TEXT", "freq TEXT",
            "symmetry TEXT", "printable TEXT"] + [f"{k} REAL" for k in keys]
    con.execute(f"CREATE TABLE designs ({', '.join(cols)})")
    rows = []
    for i in range(cat.M.shape[0]):
        g, m = cat.geoms[cat.gi[i]], cat.materials[cat.mi[i]]
        vals = [float(x) if np.isfinite(x) else None for x in cat.M[i]]
        rows.append([i, m.name, g["family"], g["mode"], g["freq"], g["sym"],
                     "true" if m.am else "false"] + vals)
    con.executemany(f"INSERT INTO designs VALUES ({','.join('?' * len(cols))})", rows)
    con.commit()
    con.execute("PRAGMA query_only = ON")
    desc = ["  id: row id", "  material, family, mode, freq: identify the design",
            "  symmetry: cubic | tetragonal | orthorhombic",
            "  printable: 'true' if the metal is routinely 3D printed"]
    for k in keys:
        p = REGISTRY[k]
        u = "" if p.unit == "-" else f" [{p.unit}]"
        desc.append(f"  {k}{u}: {p.label}")
    return con, "\n".join(desc)


def run_sql(con, sql):
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(21)
        names = [d[0] for d in cur.description] if cur.description else []
        more = len(rows) > 20
        return {"columns": names, "rows": [list(r) for r in rows[:20]],
                "truncated": more}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def render(q, extra=None):
    lines, n = [], 0
    for c in q.get("constraints", []):
        n += 1
        p = REGISTRY.get(c["property"])
        unit = "" if (p is None or p.unit == "-") else f" {p.unit}"
        v = c["value"]
        vs = f"{v:g}" if isinstance(v, (int, float)) else str(v)
        lines.append(f"{n}. {c['property']} {OPW[c['op']]} {vs}{unit}")
    for e in extra or []:
        n += 1
        lines.append(f"{n}. {e}")
    obj = [f"{'maximise' if o['sense'] == 'max' else 'minimise'} {o['property']}"
           for o in q.get("objectives", [])]
    head = "Requirements:\n" + "\n".join(lines)
    return head + ("\nAmong designs that meet them, " + "; ".join(obj) + "." if obj else "")


def call(body):
    url = llm.ENDPOINT.format(m=MODEL, k=llm._key())
    data = json.dumps(body).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"API failed: {last}")


def agent(con, system, text):
    tools = [{"functionDeclarations": [{
        "name": "run_sql", "description": "Run one read-only SQLite query on table designs; "
                                          "returns at most 20 rows.",
        "parameters": {"type": "OBJECT", "properties": {"sql": {"type": "STRING"}},
                       "required": ["sql"]}}]}]
    contents = [{"role": "user", "parts": [{"text": text}]}]
    calls, t0 = [], time.time()
    for turn in range(MAX_CALLS + 1):
        last = turn == MAX_CALLS
        body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents,
                "generationConfig": {"temperature": 0}}
        if last:
            # Budget used up. The model ignored functionCallingConfig NONE and,
            # asked in text, sometimes kept planning; the last turn offers no
            # tool and is decoded against the answer schema.
            contents[-1]["parts"].append({"text": BUDGET_MSG})
            body["generationConfig"].update({"responseMimeType": "application/json",
                                             "responseSchema": ANSWER_SCHEMA})
        else:
            body["tools"] = tools
        out = call(body)
        content = out["candidates"][0]["content"]
        contents.append(content)
        fcs = [p["functionCall"] for p in content.get("parts", []) if "functionCall" in p]
        if not fcs or last:
            text_out = "".join(p.get("text", "") for p in content.get("parts", []))
            return {"final": text_out, "calls": calls, "rounds": turn, "forced": last,
                    "seconds": round(time.time() - t0, 2)}
        resp_parts = []
        for fc in fcs:
            res = run_sql(con, fc.get("args", {}).get("sql", ""))
            calls.append({"sql": fc.get("args", {}).get("sql", ""),
                          "n_rows": len(res.get("rows", [])), "error": res.get("error")})
            resp_parts.append({"functionResponse": {"name": "run_sql", "response": res}})
        contents.append({"role": "user", "parts": resp_parts})
    return {"final": "", "calls": calls, "seconds": round(time.time() - t0, 2)}


def parse_final(text):
    m = None
    for m in re.finditer(r"\{.*\}", text, re.S):
        pass
    if not m:
        return None
    blob = m.group(0)
    # take the last balanced JSON object
    for start in [i for i, ch in enumerate(blob) if ch == "{"]:
        try:
            return json.loads(blob[start:])
        except json.JSONDecodeError:
            continue
    return None


def load_sets(cat):
    sys.path.insert(0, str(PAPER / "scripts"))
    from aei_upgrade_analyses import frozen_suite
    suite = [{"id": s["id"], "query": {"objectives": s.get("objectives", []),
                                       "constraints": s["constraints"]}}
             for s in frozen_suite()]
    p2 = json.loads((PAPER / "data" / "p2_queries.json").read_text(encoding="utf-8"))
    empty = [{"id": x["id"], "query": {"objectives": [], "constraints": x["constraints"]}}
             for x in p2["items"]]
    extras = ["yield strength at least 150 MPa", "fatigue life at least 1e6 cycles",
              "corrosion resistance rated good in sea water", "largest pore at most 2 mm",
              "Charpy impact energy at least 20 J", "pressure drop below 500 Pa at 1 m/s",
              "service temperature cycling survives 1000 cycles", "surface roughness Ra below 5 um",
              "electrical conductivity at least 10 MS/m", "weldable by TIG",
              "magnetic permeability below 1.05", "biocompatible for implants"]
    base = [s for s in suite if s["id"].startswith("r_")] + \
           [s for s in suite if s["id"].startswith("p_")][:8]
    undeclared = []
    for i, e in enumerate(extras):
        b = base[i % len(base)]
        undeclared.append({"id": f"u_{i:02d}", "query": b["query"], "extra": [e]})
    return {"suite": suite, "empty216": empty, "undeclared": undeclared}


def run(set_name, shard=None):
    """Run one set. With shard "k/n", run items k, k+n, ... into their own file
    (merge them with `merge <set>`); items are independent, so shards can run
    in parallel."""
    cat = Catalogue(csv_path=ROOT / "metagpt" / "catalogue.csv")
    con, cols = build_db(cat)
    system = SYSTEM.format(columns=cols)
    items = load_sets(cat)[set_name]
    main_path = OUTDIR / f"{set_name}.json"
    finished = json.loads(main_path.read_text(encoding="utf-8")) if main_path.exists() else {}
    path = main_path
    if shard:
        j, n = (int(x) for x in shard.split("/"))
        items = [it for i, it in enumerate(items) if i % n == j]
        path = OUTDIR / f"{set_name}.shard{j}.json"
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for k, it in enumerate(items):
        if it["id"] in done or it["id"] in finished:
            continue
        text = render(it["query"], it.get("extra"))
        try:
            res = agent(con, system, text)
            res["answer"] = parse_final(res["final"])
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}"}
        res["prompt"] = text
        done[it["id"]] = res
        path.write_text(json.dumps(done, indent=1, default=str), encoding="utf-8")
        print(f"{set_name} {k+1}/{len(items)} {it['id']} "
              f"{(res.get('answer') or {}).get('decision')} calls={len(res.get('calls', []))}",
              flush=True)


def infeasible(cat, cons):
    if not cons:
        return False
    masks = cat._mask_constraints(cons)
    keep = np.ones(cat.M.shape[0], bool)
    for m in masks.values():
        keep &= m
    return not keep.any()


def score():
    cat = Catalogue(csv_path=ROOT / "metagpt" / "catalogue.csv")
    sets = load_sets(cat)
    report = {"model": MODEL}
    for name in ("suite", "empty216", "undeclared"):
        path = OUTDIR / f"{name}.json"
        if not path.exists():
            continue
        res = json.loads(path.read_text(encoding="utf-8"))
        tally = {"n": 0, "api_error": 0, "no_final_answer": 0, "feasible": 0, "empty": 0,
                 "feasible_row_ok": 0, "feasible_false_refusal": 0,
                 "feasible_row_violates": 0, "empty_refused": 0,
                 "empty_row_returned": 0, "conflict_named": 0,
                 "conflict_sound": 0, "conflict_is_mus": 0,
                 "undeclared_flagged": 0, "undeclared_row_without_flag": 0,
                 "tool_calls_median": None, "seconds_median": None}
        ncalls, secs = [], []
        for it in sets[name]:
            r = res.get(it["id"])
            if r is None:
                continue
            tally["n"] += 1
            cons = [dict(c) for c in it["query"]["constraints"]]
            from retrieval import stamp_constraints
            cons = stamp_constraints(cons)
            empty = infeasible(cat, cons)
            tally["empty" if empty else "feasible"] += 1
            ans = r.get("answer")
            if r.get("error"):
                tally["api_error"] += 1
                continue
            if not isinstance(ans, dict):
                # no JSON decision, even after the answer-now turn
                tally["no_final_answer"] += 1
                tally["no_final_answer_" + ("empty" if empty else "feasible")] = \
                    tally.get("no_final_answer_" + ("empty" if empty else "feasible"), 0) + 1
                continue
            ncalls.append(len(r.get("calls", [])))
            secs.append(r.get("seconds", 0))
            # answered in the schema-decoded turn after all eight rounds
            tally["used_all_rounds"] = tally.get("used_all_rounds", 0) + bool(r.get("forced"))
            dec = ans.get("decision")
            rid = ans.get("id")
            row_ok = None
            if dec == "row" and isinstance(rid, int) and 0 <= rid < cat.M.shape[0]:
                masks = cat._mask_constraints(cons)
                row_ok = all(bool(m[rid]) for m in masks.values())
            if name == "undeclared":
                n_req = len(cons) + 1
                if n_req in (ans.get("unverifiable_requirements") or []) or dec == "none":
                    tally["undeclared_flagged"] += 1
                elif dec == "row":
                    tally["undeclared_row_without_flag"] += 1
                continue
            if not empty:
                if dec == "row" and row_ok:
                    tally["feasible_row_ok"] += 1
                elif dec == "row":
                    tally["feasible_row_violates"] += 1
                else:
                    tally["feasible_false_refusal"] += 1
            else:
                if dec == "row":
                    tally["empty_row_returned"] += 1
                else:
                    tally["empty_refused"] += 1
                    named = [int(x) for x in (ans.get("conflicting_requirements") or [])
                             if str(x).isdigit() and 1 <= int(x) <= len(cons)]
                    if named:
                        tally["conflict_named"] += 1
                        sub = [cons[i - 1] for i in named]
                        if infeasible(cat, sub):
                            tally["conflict_sound"] += 1
                            if all(not infeasible(cat, [c for c in sub if c is not d])
                                   for d in sub):
                                tally["conflict_is_mus"] += 1
        if ncalls:
            tally["tool_calls_median"] = float(np.median(ncalls))
            tally["seconds_median"] = float(np.median(secs))
        report[name] = tally
    (PAPER / "data" / "agent_baseline.json").write_text(json.dumps(report, indent=1),
                                                         encoding="utf-8")
    print(json.dumps(report, indent=1))


def merge(set_name):
    """Fold shard files into the set's file and delete them."""
    main_path = OUTDIR / f"{set_name}.json"
    done = json.loads(main_path.read_text(encoding="utf-8")) if main_path.exists() else {}
    shards = sorted(OUTDIR.glob(f"{set_name}.shard*.json"))
    for sp in shards:
        for key, val in json.loads(sp.read_text(encoding="utf-8")).items():
            if key in done:
                raise SystemExit(f"{key} is in two files")
            done[key] = val
    main_path.write_text(json.dumps(done, indent=1, default=str), encoding="utf-8")
    for sp in shards:
        sp.unlink()
    print(f"{set_name}: {len(done)} items after merging {len(shards)} shards")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "suite"
    if cmd == "score":
        score()
    elif cmd == "merge":
        merge(sys.argv[2])
    else:
        run(cmd, sys.argv[2] if len(sys.argv) > 2 else None)
