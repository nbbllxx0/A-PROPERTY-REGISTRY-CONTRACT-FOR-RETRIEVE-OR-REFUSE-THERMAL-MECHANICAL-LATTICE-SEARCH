# P2-R protection-first control (post-result, same frozen queries)

Added: 2026-09-05, after independent review of the 209/211 vs 211/211 gap.
Not a new query draw. Does not rewrite `p2_revision_preregister.md`.

## Why

The informed min-cardinality baseline first keeps globally smallest correction
sets of the original unconstrained problem, then rejects those that touch
protected requirements. On `t_01` and `x_102` the only globally smallest
repair drops density, so that list reports failure. Density is not an
admissible relaxation once it is protected.

## Control (no MUS/MCS)

1. Keep catalogue rows that satisfy every protected original bound.
2. Count violated allowable constraints on those rows.
3. Take a row with the smallest count (ties: lowest catalogue index).
4. Replace only those violated allowable bounds with outward-printed witness
   values; retained protected bounds stay exact.

This is ordinary constrained minimum repair. Equal-score row choice is not
matched to full diagnosis, so cell identity is not compared.

## Frozen inputs

Same `p2_queries.json` (seed 20260905), same protection mask, same printer.

## Measured (same run as `p2_revision.json`)

- Eligible: 211/216
- Informed min-cardinality list, then protection: 209/211
- Protection-first control: 211/211
- Full diagnosis: 211/211
- Repair loss matches full diagnosis on 211/211 eligible queries
- Min-cardinality list fails, others succeed: 2/211 (`t_01`, `x_102`)
