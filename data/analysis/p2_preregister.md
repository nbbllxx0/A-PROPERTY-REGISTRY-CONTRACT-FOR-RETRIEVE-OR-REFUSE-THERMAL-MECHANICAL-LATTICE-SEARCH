# P2 pre-registration (frozen before results)

Committed: 2026-09-05, before any extra-query draw or metric tally.
Tree: `upgrade_20260905/` only. Do not change this file after `p2_repair_witness.py` writes `repair_info.json`.

## Question

Does acting on the reported repair restore feasibility after numbers are printed
at the tool's displayed precision (`.3g`)? Mesh replay is repair-witness
validation of the selected cell, not global infeasibility.

## Sampling

- Include all 16 empty queries from `frozen_suite()` in `aei_upgrade_analyses.py`.
- Draw extra queries with numpy Generator `seed=20260905`.
- Keep drawing until **N = 200** extra *empty* queries are stored, each with 2–5
  constraints.
- Property pool (frozen): `rho`, `E_11`, `k_11`, `cost_per_kg`, `k_aniso`,
  `mass_density`. Ops: `<=` on `rho`, `cost_per_kg`, `k_aniso`, `mass_density`;
  `>=` on `E_11`, `k_11`.
- Values: independent uniform draws on the closed interval between the 10th and
  90th percentiles of the searchable catalogue column (inclusive).
- Reject a draw if the full-precision search is feasible, if it exceeds the
  eight-constraint diagnosis cap, if it has a duplicate property, or if its
  exact constraint-set signature matches an already stored query.
- Signature: sorted `(property, op, format(value, '.3g'))`. Near-duplicate
  threshold: **none** beyond that exact signature.
- Do not shrink N after seeing rates.

## Methods (same printed precision)

1. **Min-repair.** First minimum-cardinality MCS in reasoner order; jointly
   attainable row (`min_repair_search`).
2. **Best-objective min-cardinality repair.** Among repairs with
   `minimum_cardinality`, pick the jointly attainable row that maximises the
   pre-registered scalar below. Not second-MCS, not lex-drop.
3. **Full diagnosis.** MUS family, every inclusion-minimal MCS, joint repair
   for each.

Scalar for method 2: if the query has objectives, `Catalogue._score` on that
row (same rank-normalised weighted sum as the paper). If not, the mean of
rank-normalised column values oriented toward each original constraint
(`>=` uses the column; `<=` uses `1 -` the column). Ties: smaller `uid`.

Printed repair query: keep non-MCS constraints; replace each MCS atom with the
same op and the repair-vector value formatted then parsed as `.3g`. Executability
is a non-empty search on that printed query.

## Denominators

- Executability of a method: queries where that method returns a repair.
- Disagreement of methods 1 vs 2: queries where both return a repair.
- Cardinality: min-cardinality counts and inclusion-minimal counts are separate.
- Witness column: only if the selected cell `uid` is in `verify_sample60.json`.
  Pass = catalogue values of that cell satisfy the printed repaired query.
  Residual check (same cell only): scale `k_11` and `E_11` by the stored
  relative residual when present; still not a catalogue-wide model.
  Witness fail is not labelled global infeasibility.

## Outputs

`paper_aei/data/repair_info.json` and a printed four-fraction table. No
post-hoc change to sampling, scalar, or denominators.
