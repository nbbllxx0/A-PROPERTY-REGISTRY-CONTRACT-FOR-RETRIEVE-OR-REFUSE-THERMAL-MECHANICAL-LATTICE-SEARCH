"""
Does the language model earn its place?

This is the part that decides whether the interface is a contribution or a
wrapper. The honest comparison is against the parser the conference paper
already has: a keyword table. If a table matches the model, the model is
decoration -- and over a one-dimensional property space a table genuinely is
optimal, which is worth saying out loud rather than hiding.

So the harness measures both on the same requests, and separates the request
types where a table is sufficient from the ones where it cannot work at all:

  simple        one property, plain wording            -- a table should win
  compositional several properties at once             -- combinatorial
  directional   needs an axis resolved from words      -- table has no mechanism
  infeasible    correct answer is "no such design"     -- needs the search too
  contradictory mutually exclusive requirements
  vocabulary    mentions something the system cannot represent at all
  paraphrase    same intent, unusual wording

    python evaluate.py --quick        # a few cases, no API cost to speak of
    python evaluate.py                # the full set
"""

import argparse
import json
import pathlib
import re
import statistics as stats
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent

from schema import REGISTRY  # noqa: E402

# --------------------------------------------------------------- the baseline
# A keyword table, in the spirit of the conference paper's semantic grounding.
# Deliberately built as well as a table can reasonably be built.
KEYWORDS = [
    (r"\blight(er|weight)?\b|\blow mass\b|\bnot heavy\b", "rho", "min"),
    (r"\bheav(y|ier)\b|\bdense\b", "rho", "max"),
    (r"\bstiff(er|ness)?\b|\brigid\b|\bstrong\b", "E_11", "max"),
    (r"\bcomplian(t|ce)\b|\bflexible\b|\bsoft\b", "E_11", "min"),
    (r"\bconduct(ive|ivity|s heat)\b|\bdissipate heat\b|\bcool\b", "k_11", "max"),
    (r"\binsulat(e|ing|ion)\b|\bthermal barrier\b", "k_11", "min"),
    (r"\bcheap(er)?\b|\baffordable\b|\blow cost\b", "cost_per_kg", "min"),
    (r"\bexpansion\b|\bdimensionally stable\b|\bwarp\b", "cte", "min"),
    (r"\bhot\b|\bhigh temperature\b", "tmax", "max"),
]

NUM_CONSTRAINT = re.compile(
    r"(?:at least|minimum|min|no less than|over|above)\s+([\d.]+)\s*"
    r"(gpa|w/mk|w/\(m k\)|c|celsius)?", re.I)


def rule_based_parse(text: str) -> dict:
    """The if/elif baseline. No axis handling, no infeasibility awareness,
    no way to notice that a request mentions something it cannot express."""
    t = text.lower()
    objs, seen = [], set()
    for pat, prop, sense in KEYWORDS:
        if re.search(pat, t) and prop not in seen:
            objs.append({"property": prop, "sense": sense})
            seen.add(prop)
    cons = []
    for m in NUM_CONSTRAINT.finditer(t):
        val, unit = float(m.group(1)), (m.group(2) or "").lower()
        if "gpa" in unit:
            cons.append({"property": "E_11", "op": ">=", "value": val})
        elif "w/" in unit:
            cons.append({"property": "k_11", "op": ">=", "value": val})
        elif unit.startswith("c"):
            cons.append({"property": "tmax", "op": ">=", "value": val})
    mf = {}
    if re.search(r"\bprint(able|ed)\b|\badditive\b|\b3d print", t):
        mf["printable_only"] = True
    return {"objectives": objs, "constraints": cons, "material_filter": mf,
            "unmet": [], "_rejected": []}


# ------------------------------------------------------------------ test set
# props: properties that must appear (any sense).  axes: property keys whose
# axis index matters.  oov: content the system cannot represent.
TESTS = [
    # --- simple ----------------------------------------------------------
    dict(id="s1", cat="simple", text="I want the lightest possible part",
         props={"rho"}),
    dict(id="s2", cat="simple", text="as stiff as you can make it",
         props={"E_11"}),
    dict(id="s3", cat="simple", text="something that conducts heat well",
         props={"k_11"}),
    dict(id="s4", cat="simple", text="keep the material cost down",
         props={"cost_per_kg"}),
    # --- compositional ---------------------------------------------------
    dict(id="c1", cat="compositional",
         text="light, stiff, and a good conductor of heat",
         props={"rho", "E_11", "k_11"}),
    dict(id="c2", cat="compositional",
         text="cheap, printable, stiff, and it has to survive 400 C",
         props={"cost_per_kg", "E_11", "tmax"}),
    dict(id="c3", cat="compositional",
         text="stiff for its weight and conductive for its weight",
         props={"specific_stiffness", "specific_conductivity"}),
    dict(id="c4", cat="compositional",
         text="minimise mass, keep stiffness above 2 GPa, and don't spend more "
              "than 10 dollars a kilo",
         props={"E_11", "cost_per_kg"}),
    # --- directional -----------------------------------------------------
    # Directional cases are graded on what the search RETURNS, not on which
    # properties the parse names. "spreads sideways, insulates upward" can be
    # written as (k_11 max, k_33 min) or as (k_aniso min); both are right, and
    # a parse-matching test would fail the second. Only the returned design
    # settles it.
    dict(id="d1", cat="directional",
         text="spreads heat sideways but insulates upward",
         props={"k_11", "k_33"},
         expect=lambda r: r["k_11"] > 1.5 * r["k_33"],
         expect_desc="returns a cell conducting >1.5x better sideways than up"),
    dict(id="d2", cat="directional",
         text="stiff along the length, doesn't matter through the thickness",
         props={"E_11"},
         expect=lambda r: r["E_11"] >= r["E_33"],
         expect_desc="returns a cell at least as stiff along 1 as along 3"),
    dict(id="d3", cat="directional",
         text="I need the heat to travel through the thickness, not sideways",
         props={"k_33", "k_11"},
         expect=lambda r: r["k_33"] > 1.5 * r["k_11"],
         expect_desc="returns a cell conducting >1.5x better up than sideways"),
    dict(id="d4", cat="directional",
         text="conducts equally in every direction",
         props={"k_aniso"},
         expect=lambda r: 0.9 <= r["k_aniso"] <= 1.1,
         expect_desc="returns a near-isotropic cell"),
    # --- infeasible (the search must refuse) -----------------------------
    dict(id="i1", cat="infeasible",
         text="relative density under 0.15 but stiffness above 100 GPa",
         props={"rho", "E_11"}, must_refuse=True),
    dict(id="i2", cat="infeasible",
         text="conductivity above 500 W/mK at 20 percent density",
         props={"k_11", "rho"}, must_refuse=True),
    dict(id="i3", cat="infeasible",
         text="cheaper than 1 dollar a kilo and conducts better than 300 W/mK",
         props={"cost_per_kg", "k_11"}, must_refuse=True),
    dict(id="i4", cat="infeasible",
         text="cubic symmetry but conducts three times better sideways than up",
         props={"k_aniso"}, must_refuse=True),
    # --- contradictory ----------------------------------------------------
    dict(id="x1", cat="contradictory",
         text="as light as possible and as dense as possible",
         props={"rho"}),
    dict(id="x2", cat="contradictory",
         text="insulating but highly conductive", props={"k_11"}),
    # --- out of vocabulary -------------------------------------------------
    dict(id="v1", cat="vocabulary",
         text="stiff, light, and resistant to salt water corrosion",
         props={"rho", "E_11"}, oov=True),
    dict(id="v2", cat="vocabulary",
         text="conductive and with good fatigue life at 10^7 cycles",
         props={"k_11"}, oov=True),
    dict(id="v3", cat="vocabulary",
         text="cheap to machine, stiff, and paintable",
         props={"E_11"}, oov=True),
    # --- paraphrase / colloquial -------------------------------------------
    dict(id="p1", cat="paraphrase",
         text="needs to shift a lot of heat without weighing a ton",
         props={"k_11", "rho"}),
    dict(id="p2", cat="paraphrase",
         text="basically a heat spreader for a chip, has to be printable",
         props={"k_11"}),
    dict(id="p3", cat="paraphrase",
         text="don't want it warping when it heats up, and it should be rigid",
         props={"cte", "E_11"}),
    dict(id="p4", cat="paraphrase",
         text="think aluminium heat sink but lighter",
         props={"rho", "k_11"}),
]


def props_of(q):
    return {o["property"] for o in q.get("objectives", [])} | \
           {c["property"] for c in q.get("constraints", [])}


# Score the physical concerns identified, not the key names chosen.
#
# The first version of this harness compared property keys exactly, and reported
# the language model losing to a keyword table. That was wrong. Asked for "the
# lightest possible part" it returned `mass_density` where the answer key said
# `rho` -- and mass_density is the better answer, because relative density alone
# ignores that copper at 20% is heavier than magnesium at 40%. Asked for "stiff
# and light" it returned `specific_stiffness`, which is how an engineer would
# actually express it.
#
# Exact-key matching was measuring agreement with one arbitrary encoding rather
# than understanding of the request. Concepts are what the request is about.
CONCEPT = {
    "rho": {"MASS"}, "mass_density": {"MASS"},
    "E_11": {"STIFF"}, "E_22": {"STIFF"}, "E_33": {"STIFF"}, "E_mean": {"STIFF"},
    "specific_stiffness": {"STIFF", "MASS"},
    "k_11": {"HEAT"}, "k_22": {"HEAT"}, "k_33": {"HEAT"}, "k_mean": {"HEAT"},
    "specific_conductivity": {"HEAT", "MASS"},
    "cost_per_kg": {"COST"}, "cost_per_m3": {"COST"},
    "tmax": {"TEMP"}, "cte": {"EXPANSION"},
    "k_aniso": {"HEAT", "DIRECTION"}, "E_aniso": {"STIFF", "DIRECTION"},
    "porosity": {"MASS"}, "permeability": {"FLOW"},
}

# CONCEPT is a hand-written parallel to the registry, and it has silently
# drifted from it once already: adding E_mean to schema.py without adding it
# here scored a correct answer as zero and cost 0.12 of overall f1, which read
# as the model getting worse. A property the search understands but the scorer
# does not is not a model failure, so make the drift loud instead of silent.
_UNSCORED = {  # deliberately outside the concept vocabulary
    "k_off_rel", "C_coupling_rel", "conn_frac", "level", "n", "rho_target",
}


def _check_concept_coverage():
    from schema import REGISTRY
    missing = sorted(set(REGISTRY) - set(CONCEPT) - _UNSCORED)
    if missing:
        print(f"WARNING: {len(missing)} registry properties have no CONCEPT "
              f"entry and will score 0 whenever the model names them: "
              f"{', '.join(missing)}")
    return missing


def concepts_of(props):
    out = set()
    for p in props:
        out |= CONCEPT.get(p, set())
    return out


def _f1(got, want):
    if not want:
        return 1.0
    tp = len(got & want)
    prec = tp / len(got) if got else 0.0
    rec = tp / len(want)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def score(q, t):
    """Per-case scoring. Returns a dict of independent checks.

    ``f1`` is concept-set overlap (HEAT, STIFF, …). It does not score
    objective vs constraint role, operator, threshold, unit, or material
    filter. ``key_f1`` is property-key overlap. ``constraint_atom_f1`` is
    exact (property, operator, value) overlap when the gold item lists
    constraints; it is omitted when the gold has none.
    """
    got = props_of(q)
    want = t["props"]
    f1 = _f1(concepts_of(got), concepts_of(want))
    out = {"f1": f1,
           "key_f1": _f1(got, want),      # kept for transparency
           "recall": len(got & want) / len(want) if want else 1.0,
           "precision": len(got & want) / len(got) if got else 0.0}

    if t.get("oov"):
        out["oov_flagged"] = 1.0 if q.get("unmet") else 0.0
    out["hallucinated"] = float(len(q.get("_rejected", [])) > 0)
    gold_atoms, got_atoms = set(), set()
    for c in t.get("constraints") or []:
        try:
            gold_atoms.add(f"{c['property']}{c['op']}{float(c['value']):.6g}")
        except (KeyError, TypeError, ValueError):
            pass
    for c in q.get("constraints") or []:
        try:
            got_atoms.add(f"{c['property']}{c['op']}{float(c['value']):.6g}")
        except (KeyError, TypeError, ValueError):
            pass
    if gold_atoms:
        out["constraint_atom_f1"] = _f1(got_atoms, gold_atoms)
    return out


def main():
    _check_concept_coverage()
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--catalogue", default=None)
    ap.add_argument("--no-search", action="store_true",
                    help="skip the refusal check (no catalogue needed)")
    a = ap.parse_args()

    tests = TESTS[:8] if a.quick else TESTS
    cat = None
    if not a.no_search:
        try:
            from retrieval import Catalogue
            cat = Catalogue(csv_path=a.catalogue or HERE / "catalogue.csv")
            print(f"catalogue: {cat.stats()}")
        except Exception as e:
            print(f"(no catalogue -- refusal check skipped: {e})")

    from llm import parse

    rows, lat = [], []
    for t in tests:
        rec = {"id": t["id"], "cat": t["cat"], "text": t["text"]}
        q_llm = parse(t["text"])
        lat.append(q_llm.get("_latency_s", 0))
        q_rul = rule_based_parse(t["text"])
        rec["llm"] = score(q_llm, t)
        rec["rule"] = score(q_rul, t)

        if cat is not None:
            for tag, q in (("llm", q_llm), ("rule", q_rul)):
                r = cat.search(q, top_k=1)
                if t.get("must_refuse"):
                    rec[tag]["refused"] = 1.0 if not r.rows else 0.0
                if t.get("expect") is not None:
                    # end-to-end: did the system return what was asked for,
                    # regardless of how the request was encoded?
                    try:
                        rec[tag]["delivered"] = (
                            1.0 if r.rows and t["expect"](r.rows[0]) else 0.0)
                    except Exception:
                        rec[tag]["delivered"] = 0.0
        rows.append(rec)
        print(f"  {t['id']:4s} {t['cat']:14s} "
              f"llm f1={rec['llm']['f1']:.2f}  rule f1={rec['rule']['f1']:.2f}")

    summarise(rows, lat)
    (HERE / "eval_results.json").write_text(json.dumps(rows, indent=1))
    print(f"\nwrote eval_results.json   mean latency {stats.mean(lat):.2f}s")


def summarise(rows, lat):
    cats = sorted({r["cat"] for r in rows})
    print("\n" + "=" * 78)
    print("f1 = concepts identified (see CONCEPT map); key = exact property "
          "key agreement")
    print(f"{'category':16s}{'n':>3s}{'LLM f1':>9s}{'rule f1':>9s}"
          f"{'LLMkey':>8s}{'deliv':>8s}{'refuse':>8s}{'oov':>7s}")
    print("-" * 78)
    for c in cats:
        sel = [r for r in rows if r["cat"] == c]
        f_l = stats.mean(r["llm"]["f1"] for r in sel)
        f_r = stats.mean(r["rule"]["f1"] for r in sel)
        k_l = stats.mean(r["llm"]["key_f1"] for r in sel)
        ax = [r["llm"].get("delivered") for r in sel if "delivered" in r["llm"]]
        rf = [r["llm"].get("refused") for r in sel if "refused" in r["llm"]]
        ov = [r["llm"].get("oov_flagged") for r in sel if "oov_flagged" in r["llm"]]
        print(f"{c:16s}{len(sel):3d}{f_l:9.2f}{f_r:9.2f}{k_l:8.2f}"
              f"{(stats.mean(ax) if ax else float('nan')):8.2f}"
              f"{(stats.mean(rf) if rf else float('nan')):8.2f}"
              f"{(stats.mean(ov) if ov else float('nan')):7.2f}")
    print("-" * 78)
    print(f"{'ALL':16s}{len(rows):3d}"
          f"{stats.mean(r['llm']['f1'] for r in rows):9.2f}"
          f"{stats.mean(r['rule']['f1'] for r in rows):9.2f}"
          f"{stats.mean(r['llm']['key_f1'] for r in rows):8.2f}")
    hall = stats.mean(r["llm"]["hallucinated"] for r in rows)
    print(f"\nhallucinated property names, rejected by validation: {hall:.0%}")


if __name__ == "__main__":
    main()
