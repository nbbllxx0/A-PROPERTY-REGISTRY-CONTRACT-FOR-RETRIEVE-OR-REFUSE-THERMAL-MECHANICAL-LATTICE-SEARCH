# A property-registry contract for retrieve-or-refuse thermal–mechanical lattice search

Data and code for the paper.

> Shaoliang Yang, Henry Chu, Zu Yashengjiang, Jun Wang.
> *A property-registry contract for retrieve-or-refuse thermal–mechanical lattice search.*
> arXiv:2609.14741 [cs.CE], 2026. <https://arxiv.org/abs/2609.14741>

An engineer's sentence is compiled against a declared property registry into a
typed query over a catalogue of solved metal–cell combinations. If the feasible
set is non-empty the system returns a catalogue row that a second solver can
rebuild. If it is empty the system returns **why** — the minimal unsatisfiable
subsets, the correction sets, and the slack of a repair — rather than the
nearest row.

---

## What is here

```
data/      every artefact the paper's Data availability statement lists
src/       the registry, the reasoner, the homogenisation solver, the checks
```

### `data/`

| File | What it is |
|---|---|
| `catalogue.csv` | The catalogue. 1,536 enumerated rows, 45 columns; the `feasible` flag marks the **1,397** searchable rows. Stores ρ, k\* (3×3), C\* (6×6), derived directional moduli, the void coefficients `B11/B22/B33`, and the pore diffusivities `D11/D22/D33`. |
| `dstar_complete_pore.csv` | The 1,397 complete-periodic-pore D\* solves behind Sec. 5.4. |
| `suite_64.csv` | The frozen typed suite: 64 queries, 48 feasible and 16 empty. |
| `boundary_bench_80.csv` | The constructed boundary bench: 56 single-constraint items and 24 jointly infeasible pairs. |
| `poisson_sweep.csv` | The Poisson-ratio sweep behind the factorisation limits. |
| `benchmark.json` | The 308-request parse benchmark, seven categories, paired literal and paraphrased. |
| `benchmark_boundary.json` | The source items for the 80-item boundary bench. |

### `data/analysis/`

The result record behind each reported number. `aei_upgrade_analyses.json`
holds the two worked gold queries, the four typed briefs, the frozen-suite
policy scores, the material/geometry leverage, the stability draws and the
latencies. `binding_set_eval.json` holds the boundary-bench scores;
`mus_oracle.json` and `threeway_mus.json` the MUS/MCS oracle tests;
`verify_sample60.json` the sampled GPU-versus-CPU re-solve reports;
`repair_info.json`, `p2_queries.json` and `p2_revision.json` the frozen
empty-query repair and revision checks.

The `*_preregister.md` files fix each metric **before** its result and are
included unedited, as are `p2_protocol_correction.md` and
`p2_outward_followup.md`, which record where a protocol was corrected after
the fact.

### `data/model_outputs/`

The raw scoring records behind the 308-request parse benchmark and the
vocabulary ablation, for both model SKUs, plus the two registry-derived
baselines. These make the one learned stage checkable without re-running
the model. See the README in that directory.

### `data/verification_20260920/`

Records from a co-author review round: semantic-frame scoring of the parse
against author-assigned gold frames, the nullity measurement on the
lowest-connectivity cell, the dual-axis heat-spreader query and its density
sweep, and the connectivity-floor comparison. See the README in that
directory.

### `data/verification_20260922/`

Fifteen design-aim sentences from published papers, selected by a fixed rule
and run unchanged through parse and search, and the Poisson re-solves at the
ceramic and noble-metal ratios. See the README in that directory.

### `data/openfem/`

The second-library check: `scikit-fem` and TetGen run against the catalogue
solver on the same voxel occupancy, in both physics. Drivers, raw logs, and
`openfem_checks.json`. The logs are the original run output and state their own
limitation — the driver is ours, not an independent third-party run.

### `src/`

| Package | Contents |
|---|---|
| `metahomog/` | Geometry and physics. `tpms.py` builds the implicit cells, `homogenize.py` solves the periodic cell problem on trilinear hexes, `pore_phase.py` points the same solver at the void to get D\*, and **`validate.py` is the sixteen-check closed-form ladder** referenced in Sec. 4.2. |
| `metagpt/` | The contract and the search. **`schema.py` is the property registry** — the single declaration from which both the language-model prompt and the deterministic evaluator are generated. `retrieval.py` is the retrieve-or-refuse reasoner with MUS and MCS enumeration, `refusal.py` the refusal objects, `materials.py` the 19-metal handbook layer, `evaluate.py` the parse benchmark harness, `verify.py` the GPU-versus-CPU re-solve check. |

---

## Reproducing the checks

Everything except the parse stage is deterministic arithmetic over the frozen
catalogue and needs no network and no API key.

```bash
pip install numpy scipy pandas matplotlib
cd src/metahomog && python validate.py        # the sixteen-check ladder
cd ../metagpt    && python verify.py          # re-solve a catalogue row
```

`scikit-fem` and TetGen are needed only to re-run `data/openfem/`; the logs of
the runs reported in the paper are included so the check can be read without
installing them.

**The parse stage is the only learned component.** `llm.py` reads a Gemini API
key from `api.txt` in the project root at call time. That file is **not** in
this repository and is git-ignored. Nothing else in the pipeline touches a
model: retrieval is a boolean mask intersection followed by stable sorts over a
frozen table, so the same parse returns the same row bit for bit.

## Scope

Every effective property here is computed, not measured. Material constants are
nominal wrought values; additively manufactured parts sit below them. The
elastic geometry factor is computed once at Poisson's ratio 0.3. Agreement
between two of our code paths is evidence about the discrete problem, not about
a physical cell. The paper's Limitations section states the rest.

## Licence

Code in `src/` is MIT (see `LICENSE`). The data in `data/` and the paper itself
are CC BY 4.0, matching the arXiv posting. If you use either, please cite the
paper above.
