"""
Natural language -> structured query, via Gemini.

The language model does exactly one job: turn a sentence into the query object
defined in schema.py. Everything after that is deterministic physics and search.
Keeping the boundary this sharp is what makes the system testable -- a wrong
answer is attributable to the parse or to the search, never ambiguously to both.

It also lets us ask the honest question later: is a learned parser actually
needed, or would a lookup table do? That depends entirely on how rich the
property space is, and we can measure it either way.

Needs api.txt in the project root. The key is read at call time and never
written to disk, logged, or included in any output.
"""

import json
import pathlib
import time
import urllib.error
import urllib.request

import numpy as np

from schema import prompt_block, AXIS_CONVENTION, REGISTRY

WEIGHT_MAX = 100.0

MODEL = "gemini-3.5-flash-lite"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={k}"
KEYFILE = pathlib.Path(__file__).resolve().parent.parent / "api.txt"

_OPS = ["<=", ">=", "==", "<", ">"]

QUERY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "objectives": {
            "type": "ARRAY",
            "description": "what to optimise. Relative importance is carried by "
                           "`weight`, not by position -- the search reads weight "
                           "and ignores order.",
            "items": {"type": "OBJECT", "properties": {
                "property": {"type": "STRING"},
                "sense": {"type": "STRING", "enum": ["max", "min"]},
                "weight": {"type": "NUMBER"}},
                "required": ["property", "sense"]}},
        "constraints": {
            "type": "ARRAY",
            "description": "hard limits that must hold",
            "items": {"type": "OBJECT", "properties": {
                "property": {"type": "STRING"},
                "op": {"type": "STRING", "enum": _OPS},
                "value": {"type": "NUMBER"}},
                "required": ["property", "op", "value"]}},
        "material_filter": {
            "type": "OBJECT",
            "properties": {
                "printable_only": {"type": "BOOLEAN"},
                "cost_max": {"type": "NUMBER"},
                "allowed": {"type": "ARRAY", "items": {"type": "STRING"}},
                "excluded": {"type": "ARRAY", "items": {"type": "STRING"}}}},
        "notes": {"type": "STRING",
                  "description": "anything in the request that could not be "
                                 "expressed as an objective or constraint"},
        "unmet": {"type": "ARRAY", "items": {"type": "STRING"},
                  "description": "parts of the request this vocabulary cannot "
                                 "represent at all"},
    },
    "required": ["objectives", "constraints"],
}

SYSTEM = f"""You translate an engineer's request for a porous metal part into a
structured query. You do not answer the request or suggest a material yourself
-- a physics search does that afterwards.

{AXIS_CONVENTION}

Available properties:
{prompt_block()}

Rules:
- Use only property keys from the list above. Never invent one.
- A wish ("as light as possible", "cheap") is an objective. A requirement
  ("at least 5 GPa", "under 200 C") is a constraint with a number.
- If the request implies a direction, pick the axis-specific property rather
  than the mean.
- Convert units to those listed. Conductivity in W/(m K), stiffness in GPa.
- If the request mentions something this vocabulary cannot express -- fatigue,
  corrosion, cost of manufacture, anything absent from the list -- put it in
  `unmet` rather than forcing it into a property that does not mean the same
  thing.
- Requests can be contradictory or impossible. Translate them faithfully
  anyway; deciding feasibility is the search's job, not yours.
- Give every objective a positive `weight`. Use 1.0 when the request treats its
  wishes as equally important, and a larger number for whatever it emphasises
  ("mainly cheap, and if possible also stiff" -> cost 2.0, stiffness 1.0).
  Ordering carries no meaning; only the weights are read."""


def _key() -> str:
    if not KEYFILE.exists():
        raise FileNotFoundError(f"no API key at {KEYFILE}")
    return KEYFILE.read_text().strip()


def build_system(properties: str = None) -> str:
    """The system prompt, optionally over a restricted property vocabulary.

    Exposed so the registry-size ablation can hand the model a smaller
    vocabulary without editing the module: the whole point of that experiment is
    to vary how rich the property space is and watch what it does to the gap
    between a learned parser and a keyword table."""
    return SYSTEM if properties is None else SYSTEM.replace(prompt_block(),
                                                            properties)


def parse(request: str, model: str = MODEL, retries: int = 3,
          temperature: float = 0.0, system: str = None,
          allowed: set = None) -> dict:
    """Turn one request into a query dict. Raises on unrecoverable failure."""
    body = {
        "systemInstruction": {"parts": [{"text": system or SYSTEM}]},
        "contents": [{"parts": [{"text": request}]}],
        "generationConfig": {"responseMimeType": "application/json",
                             "responseSchema": QUERY_SCHEMA,
                             "temperature": temperature},
    }
    url = ENDPOINT.format(m=model, k=_key())
    data = json.dumps(body).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.load(r)
            q = json.loads(out["candidates"][0]["content"]["parts"][0]["text"])
            q["_latency_s"] = round(time.time() - t0, 3)
            q["_model"] = model
            return validate(q, allowed=allowed)
        except (urllib.error.HTTPError, urllib.error.URLError,
                json.JSONDecodeError, KeyError) as e:
            last = e
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"parse failed after {retries} attempts: {last}")


def validate(q: dict, allowed: set = None) -> dict:
    """Drop anything the search cannot honour, and say so rather than silently.

    A hallucinated property name is the failure mode that matters here: left in
    place it would be silently ignored during scoring and the user would get a
    confident answer to a question nobody asked.
    """
    bad = []
    q.setdefault("objectives", [])
    q.setdefault("constraints", [])
    q.setdefault("unmet", [])

    keep = []
    for o in q["objectives"]:
        if o.get("property") not in (allowed or REGISTRY):
            bad.append(f"objective on unknown property '{o.get('property')}'")
            continue
        # An unbounded weight is as damaging as a bad property name and less
        # visible: zero silently deletes an objective, and a negative one
        # silently inverts it, so a request to minimise would be maximised.
        if "weight" in o:
            try:
                w = float(o["weight"])
            except (TypeError, ValueError):
                w = float("nan")
            if not np.isfinite(w) or w <= 0:
                bad.append(f"weight {o['weight']!r} on '{o['property']}' "
                           f"replaced with 1.0 (must be positive)")
                o["weight"] = 1.0
            elif w > WEIGHT_MAX:
                bad.append(f"weight {w:g} on '{o['property']}' clamped to "
                           f"{WEIGHT_MAX:g}")
                o["weight"] = WEIGHT_MAX
        keep.append(o)
    q["objectives"] = keep

    keep = []
    for c in q["constraints"]:
        if c.get("property") not in (allowed or REGISTRY):
            bad.append(f"constraint on unknown property '{c.get('property')}'")
        elif c.get("op") not in _OPS:
            bad.append(f"unknown operator '{c.get('op')}'")
        else:
            keep.append(c)
    q["constraints"] = keep
    q["_rejected"] = bad
    return q


def explain(query: dict, result) -> str:
    """A short plain-language account of what the search returned.

    Written from the actual numbers, not from the request, so it cannot claim
    something the search did not find.
    """
    lines = [f"Request understood as: {summarise(query)}", ""]
    if result.rejected_reason:
        lines.append(f"No design satisfies this. {result.rejected_reason}")
        if result.mus:
            lines.append("MUS: " + "; ".join("{" + ", ".join(u) + "}" for u in result.mus))
        if result.min_mcs:
            lines.append("min MCS: " + "; ".join("{" + ", ".join(h) + "}" for h in result.min_mcs))
        if result.relaxation:
            lines.append(f"Closest possible: {result.relaxation}")
        return "\n".join(lines)
    lines.append(f"{len(result.rows)} candidates, best shown first.")
    return "\n".join(lines)


def summarise(q: dict) -> str:
    bits = []
    for o in q.get("objectives", []):
        bits.append(f"{o['sense']} {o['property']}")
    for c in q.get("constraints", []):
        bits.append(f"{c['property']} {c['op']} {c['value']:g}")
    mf = q.get("material_filter") or {}
    if mf.get("printable_only"):
        bits.append("printable only")
    if mf.get("cost_max"):
        bits.append(f"cost <= {mf['cost_max']:g} USD/kg")
    return "; ".join(bits) if bits else "(nothing)"


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or \
        "light and stiff along the length, spreads heat sideways but insulates upward"
    q = parse(text)
    print(json.dumps({k: v for k, v in q.items() if not k.startswith("_")},
                     indent=2))
    print(f"\nlatency {q['_latency_s']}s   rejected: {q['_rejected']}")
