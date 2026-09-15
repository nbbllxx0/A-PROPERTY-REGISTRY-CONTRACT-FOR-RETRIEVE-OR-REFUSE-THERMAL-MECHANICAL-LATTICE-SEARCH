# P2-R requirement-revision comparison (frozen before results)

Committed: 2026-09-05, on the existing frozen 216 queries in `p2_queries.json`.
No new draw. This is one explicit revision task, not a new empty-query campaign.

## Question

Given protected requirements and allowable relaxations, does full diagnosis
(all inclusion-minimal MCS) preserve the protected bounds more often, or with
less repair loss, than a transparent min-cardinality alternative-repair method
that receives the same protection mask and may rerun search?

If the two methods match, that is the result: full diagnosis does not add a
measured revision benefit beyond the min-cardinality repair list.

## Protection policy (same information for every method)

- Protected (must keep the original bound): `rho`, `cost_per_kg` when present.
  Mass and budget limits are the requirements an engineer typically will not
  drop first.
- Allowable (may drop or loosen): `E_11`, `k_11`, `k_aniso`, `mass_density`.
- Eligible query: at least one protected constraint and at least one allowable
  constraint. Other queries are reported as out of pool, not as failures.

## Action protocol (identical for both scored methods)

Each method sees the original constraints and the protection mask. It returns
one repaired typed query, or reports that it cannot revise without dropping a
protected constraint. Search is then re-run on that printed query.

Printed query: retained (including all protected) bounds kept exactly; repaired
atoms printed with the deployed outward 3 s.f. rule.

## Methods

1. **Transparent min-cardinality alternative repair (informed).** Among
   minimum-cardinality MCS that contain no protected property, pick the jointly
   attainable row with the highest P2 fallback scalar (mean rank-normalised
   orientation toward the original constraints; ties: smaller `uid`). If no
   such MCS exists, cannot-preserve. This method sees every min-cardinality
   alternative; it does not see larger inclusion-minimal MCS.

2. **Full diagnosis (informed).** Among all inclusion-minimal MCS that contain
   no protected property, pick smallest cardinality, then the same scalar and
   uid tie-break. If none exists, cannot-preserve.

A first-in-order min-repair that ignores the protection mask is reported as a
descriptive third column only. It is not the baseline.

## Outcomes

- `preserves`: printed query is executable, and every protected original
  constraint is still present at its original value.
- `repair_loss` (among queries both methods preserve): number of allowable
  constraints dropped. Lower is better.
- Primary contrast: queries where full diagnosis preserves and min-cardinality
  alternative-repair does not.

No post-hoc change to the protection mask, scalar, or denominators.

A later post-result control (`p2_revision_control.md`) enforces protection
before minimising allowable relaxations, without MUS/MCS. It is not part of
this frozen comparison.
