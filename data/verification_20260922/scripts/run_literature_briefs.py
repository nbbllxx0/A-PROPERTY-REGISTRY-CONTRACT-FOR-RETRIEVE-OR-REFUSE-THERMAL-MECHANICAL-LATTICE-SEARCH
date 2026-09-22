"""Independently authored briefs: design-aim sentences from published papers.

Selection is fixed before any sentence is read, so the authors cannot pick
briefs the system handles well:

  1. Europe PMC, open-access papers, query QUERY below, relevance order.
  2. Rule v2 (full text). From each paper's Europe PMC full text (body
     paragraphs, then abstract if no full text), the FIRST sentence of 8-60
     words that (a) states an aim or requirement (INTENT regex) and (b) names
     at least two property families (FAMILIES), one of them weight, stiffness
     or thermal. Papers with no such sentence are skipped.
  3. The first N_TAKE papers that yield a sentence are used, verbatim.

Rule v1 used abstracts only. On 2026-09-22 it returned 4 sentences from the
22 hits, all outcome statements rather than requirements, so v2 replaced it
before the parse was run on any sentence. v1 output is kept in the JSON.

Each sentence goes through the deployed parse (llm.parse, 28-key registry
prompt, temperature 0) and the unchanged search. Nothing is edited by hand.

  python run_literature_briefs.py harvest   # writes out/literature_briefs.json
  python run_literature_briefs.py run       # adds parse + search results
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "metagpt"))
OUT = HERE / "out" / "literature_briefs.json"

# Abstract-scoped so every hit is about a lattice/TPMS thermal part. A first
# form with TITLE: fields returned 4 hits and one with PUB_YEAR:[..] was
# rejected by the API; both were replaced before any abstract was read.
QUERY = ('ABSTRACT:("TPMS" OR "triply periodic minimal surface" OR "lattice structure" '
         'OR "lattice structures") AND ABSTRACT:("heat sink" OR "heat exchanger" OR '
         '"thermal management" OR "heat dissipation") AND OPEN_ACCESS:Y AND '
         'FIRST_PDATE:[2019-01-01 TO 2026-12-31]')
N_SCAN, N_TAKE = 100, 15

INTENT = re.compile(r"\b(aims?|objectives?|goals?|requires?|required|requirements?|"
                    r"must|needs?|demands?|desired|designed to|to achieve|to design)\b", re.I)
FAMILIES = {
    "weight": r"\b(light|lightweight|weight|mass|density|porosity)\b",
    "stiffness": r"\b(stiff\w*|strength|modulus|mechanical|load[- ]bearing)\b",
    "thermal": r"\b(thermal conductivity|heat transfer|heat dissipation|cooling|"
               r"thermal performance|conductiv\w+|thermal)\b",
    "cost": r"\b(cost|price|cheap\w*|economic\w*)\b",
    "manufacturing": r"\b(additive\w*|print\w*|manufactur\w*)\b",
    "fluid": r"\b(pressure drop|flow|permeab\w*|convect\w*)\b",
}


def families(sent):
    return [k for k, pat in FAMILIES.items() if re.search(pat, sent, re.I)]


def sentences(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = " ".join(text.split())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()]


def fulltext_body(pmcid):
    """Body text of an open-access PMC article, or '' if unavailable."""
    if not pmcid:
        return ""
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            xml = r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - absent full text is a normal outcome
        return ""
    m = re.search(r"<body>(.*)</body>", xml, re.S)
    if not m:
        return ""
    body = re.sub(r"<(table-wrap|fig|disp-formula|ref-list)\b.*?</\1>", " ", m.group(1), flags=re.S)
    return re.sub(r"<[^>]+>", " ", body)


def harvest():
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
           + urllib.parse.urlencode({"query": QUERY, "format": "json",
                                     "resultType": "core", "pageSize": N_SCAN}))
    for attempt in range(4):  # the search endpoint intermittently returns only {"version"}
        with urllib.request.urlopen(url, timeout=60) as r:
            resp = json.load(r)
        if "resultList" in resp:
            break
        time.sleep(3 * (attempt + 1))
    else:
        raise SystemExit("Europe PMC search returned no resultList after 4 tries")
    hits = resp["resultList"]["result"]
    prior = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else None
    taken, scanned = [], 0
    for h in hits:
        scanned += 1
        body, where = fulltext_body(h.get("pmcid")), "full text"
        if not body:
            body, where = h.get("abstractText", ""), "abstract"
        for s in sentences(body):
            fam = families(s)
            n = len(s.split())
            if (INTENT.search(s) and len(fam) >= 2 and 8 <= n <= 60
                    and set(fam) & {"weight", "stiffness", "thermal"}):
                taken.append({
                    "id": f"lit{len(taken):02d}", "text": s, "families": fam,
                    "found_in": where,
                    "source": {"title": h.get("title"), "doi": h.get("doi"),
                               "journal": (h.get("journalInfo") or {}).get("journal", {}).get("title"),
                               "year": h.get("pubYear"), "pmcid": h.get("pmcid"),
                               "authors": h.get("authorString")},
                })
                break
        if len(taken) >= N_TAKE:
            break
    out = {"query": QUERY, "retrieved": time.strftime("%Y-%m-%d"),
           "rule": ("v2: first full-text sentence of 8-60 words with an aim/requirement "
                    "cue and >=2 property families, one of weight/stiffness/thermal"),
           "rule_v1_result": (prior or {}).get("briefs") if prior and "rule_v1_result" not in prior
           else (prior or {}).get("rule_v1_result"),
           "n_scanned": scanned, "n_taken": len(taken), "briefs": taken}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"scanned {scanned}, taken {len(taken)} -> {OUT}")


def run(model=None):
    from llm import parse, MODEL
    from retrieval import Catalogue
    model = model or MODEL
    cat = Catalogue(csv_path=ROOT / "metagpt" / "catalogue.csv")
    d = json.loads(OUT.read_text(encoding="utf-8"))
    for b in d["briefs"]:
        rec = {"model": model}
        try:
            q = parse(b["text"], model=model)
        except RuntimeError as e:
            rec.update(parse_ok=False, error=str(e))
            b.setdefault("runs", {})[model] = rec
            continue
        dropped = q.get("_dropped") or q.get("dropped") or []
        rec.update(parse_ok=True,
                   objectives=q.get("objectives", []), constraints=q.get("constraints", []),
                   material_filter=q.get("material_filter"),
                   unmet=q.get("unmet", []), dropped=dropped)
        r = cat.search(q, top_k=1)
        rec.update(n_feasible=r.n_feasible,
                   refused=(r.n_feasible == 0),
                   reason=getattr(r, "reason", ""),
                   mus=getattr(r, "mus", []),
                   top=({k: r.rows[0].get(k) for k in ("material", "family", "mode", "freq", "rho",
                                                          "k_11", "E_11", "cost_per_kg")}
                        if r.rows else None))
        b.setdefault("runs", {})[model] = rec
        time.sleep(0.5)
    OUT.write_text(json.dumps(d, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("ran", model, "on", len(d["briefs"]), "briefs")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "harvest"
    if cmd == "harvest":
        harvest()
    else:
        run(sys.argv[2] if len(sys.argv) > 2 else None)
