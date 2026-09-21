# Verification records, 20 September 2026

Records produced while answering a co-author review. Each file holds the raw
output of a check reported in the manuscript, so the reported number can be
recomputed rather than taken on trust.

| File | What it holds |
|---|---|
| `catalogue_checks.json` | the registry key list; the dual-axis heat-spreader query before and after adding the `k22/k11` constraint, with its density-cap sweep; the connectivity-floor comparison; the empty-material-filter case; the enumeration-benefit queries; and the density/target/tolerance figures |
| `homogenize_checks.json` | the nullity measurement on the lowest-connectivity searchable cell, the mesh-refinement spot check, and the Poisson re-solves |
| `frame_scores_llm.json` | per-request semantic-frame scoring of the language model against author-assigned gold frames |
| `frame_scores_rule.json` | the same scoring for the registry-derived keyword table |

## What the frame scores mean

`concept_f1` is vocabulary overlap: it can be perfect while an operator, a
threshold, a unit, or an objective-versus-constraint role is wrong. The frame
scores measure those directly. On 88 gold constraint atoms:

| | keyword table | language model |
|---|---|---|
| constraint-atom F1 | 0.227 | 0.977 |
| objective-key F1 | 0.185 | 0.734 |
| concept F1 | 0.606 | 0.909 |

The language-model figures are **one run on a revised 28-key prompt**, not the
frozen five-run measurement reported in the manuscript's parse table, which is
unchanged. The gold frames are author-assigned from the template category.

## The dual-axis heat-spreader check

A ratio constraint on `k33/k11` alone is a two-axis condition. It admits the
tetragonal family `f=(1,3,3)`, on which `k22 = k33` by symmetry — so a cell can
satisfy the bound while conducting no better sideways along axis 2 than through
the thickness. `catalogue_checks.json` holds both the original query and the
one that adds `k22/k11 >= 0.90`, with the returned row for each.
