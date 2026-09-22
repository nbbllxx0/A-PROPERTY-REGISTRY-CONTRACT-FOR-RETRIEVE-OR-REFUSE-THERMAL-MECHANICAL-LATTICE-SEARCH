# Verification records, 22 September 2026

| File | What it holds |
|---|---|
| `literature_briefs.json` | 15 design-aim sentences taken verbatim from published open-access papers by a fixed rule, with source DOI and journal, the deployed parse of each, and the search outcome. The first (abstract-only) form of the rule and its 4 sentences are kept under `rule_v1_result`. |
| `poisson_sweep_extremes.json` | E11*/Es on 16 cells (8 families × 2 modes, rho ≈ 0.35, n = 32, f = 111) at nu = 0.17, 0.22, 0.24, 0.30, 0.37, 0.44. Worst deviation from the nu = 0.30 value is 5.3% (gold, nu = 0.44). |
| `scripts/` | The drivers that wrote them. `run_literature_briefs.py harvest` selects the sentences; `run` parses and searches them (needs a model key, not included); `summarise_literature_briefs.py` re-runs the search on the saved parses without any model call. `poisson_sweep_extremes.py` re-solves the sweep. |

## The literature sentences

Selection: Europe PMC, open-access, abstract mentions a TPMS or lattice
structure and a heat sink, heat exchanger or thermal-management use,
2019–2026, relevance order. From each full text, the first sentence of 8–60
words with an aim or requirement cue and at least two property families (one
of weight, stiffness or thermal). The first 15 papers that yield a sentence are
used. Nothing is edited.

Outcome: 15/15 parse, 0 undeclared keys, 8 compile to at least one objective,
constraint or material filter, 11 report content under CANNOT EXPRESS, 1 is
refused (porosity = 0.60; closest achievable 0.58), 7 compile to nothing
searchable. The sentences are aims written for papers, not requests written
for this tool; this is not a user study.

The scripts are copied as run; their import paths point at the working tree
(`metagpt/`), which is `src/metagpt/` in this repository.
