"""Summarise out/literature_briefs.json: re-search the saved parses (no model calls)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "metagpt"))
from retrieval import Catalogue  # noqa: E402

d = json.loads((HERE / "out" / "literature_briefs.json").read_text(encoding="utf-8"))
cat = Catalogue(csv_path=ROOT / "metagpt" / "catalogue.csv")
rows = []
for b in d["briefs"]:
    run = next(iter(b["runs"].values()))
    q = {"objectives": run.get("objectives", []), "constraints": run.get("constraints", []),
         "material_filter": run.get("material_filter")}
    has_content = bool(q["objectives"] or q["constraints"] or
                       any((q["material_filter"] or {}).values()))
    r = cat.search(q, top_k=1)
    rows.append({
        "id": b["id"], "parse_ok": run["parse_ok"], "registry_content": has_content,
        "n_objectives": len(q["objectives"]), "n_constraints": len(q["constraints"]),
        "material_filter": q["material_filter"], "unmet": run.get("unmet", []),
        "undeclared_keys": run.get("dropped", []), "unranked": bool(r.unranked),
        "estimate_caveats": list(r.caveats or []), "n_feasible": r.n_feasible,
        "refused": r.n_feasible == 0, "mus": r.mus, "relaxation": r.relaxation,
    })
summ = {
    "n": len(rows),
    "parse_ok": sum(x["parse_ok"] for x in rows),
    "undeclared_keys_total": sum(len(x["undeclared_keys"]) for x in rows),
    "with_registry_content": sum(x["registry_content"] for x in rows),
    "with_unmet": sum(bool(x["unmet"]) for x in rows),
    "nothing_expressible": sum(not x["registry_content"] for x in rows),
    "unranked": sum(x["unranked"] for x in rows),
    "refused": sum(x["refused"] for x in rows),
    "estimate_flagged": sum(bool(x["estimate_caveats"]) for x in rows),
}
d["summary"] = summ
d["per_brief"] = rows
(HERE / "out" / "literature_briefs.json").write_text(
    json.dumps(d, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps(summ, indent=1))
for x in rows:
    if x["refused"] or x["estimate_caveats"]:
        print(x["id"], "refused" if x["refused"] else "", x["mus"], x["relaxation"], x["estimate_caveats"])
