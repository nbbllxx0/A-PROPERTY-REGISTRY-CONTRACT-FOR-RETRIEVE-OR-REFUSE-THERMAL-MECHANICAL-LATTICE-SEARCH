# P2 follow-up: outward rounding (not pre-registered)

Date: 2026-09-05, after `p2_preregister.md` and after the frozen draw
`p2_queries.json`. Same 216 queries; no redraw.

Protocol correction: retained original bounds are kept exactly. Only repaired
atoms are printed. Mesh-witness residual scaling is withdrawn.

The pre-registered metric remains nearest-even `.3g` (**107/216** min-repair,
**94/216** best min-cardinality) after that correction.

Follow-up rule, now the deployed printer: repaired bounds shown at three
significant figures are rounded outward in signed value (`<=` / `<` toward
+inf, `>=` / `>` toward -inf; strict operators take one extra quantum) so
the printed number cannot exclude the witness. On the same frozen queries
this is 216/216 for both single-answer methods. Gold example: witness
ρ=0.32503 prints as 0.325 under nearest-even `.3g` and as 0.326 outward.
