# P2 protocol correction (not a new draw)

Date: 2026-09-05, after independent replay of the frozen 216 queries in
`p2_queries.json`. Sampling in `p2_preregister.md` is unchanged.

## What was wrong

`printed_query` rounded every retained original bound as well as the repaired
bounds. The preregistered protocol keeps non-MCS constraints. The production
slack printer also rewrites only repair values.

A separate mesh column matched nearby catalogue densities and applied an
unsigned residual to conductivity and stiffness. That is not a selected-cell
n=64 replay and is withdrawn.

## What is now measured

- Retained original bounds are kept exactly.
- Repaired atoms are printed at display precision (nearest-even `.3g` as the
  pre-registered metric; outward 3 s.f. as the deployed follow-up).
- Executability is a non-empty catalogue search on that printed query.
- No mesh-witness column.

Queries are not redrawn.
