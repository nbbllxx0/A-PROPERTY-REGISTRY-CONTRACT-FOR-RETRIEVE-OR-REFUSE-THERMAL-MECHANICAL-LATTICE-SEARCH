# A property-registry contract for retrieve-or-refuse thermal–mechanical lattice search

Data and code for the paper.

> Shaoliang Yang, Henry Chu, Zupukaer Yashengjiang, Jun Wang.
> *A property-registry contract for retrieve-or-refuse thermal–mechanical lattice search.*
> arXiv:2609.14741 [cs.CE], 2026. <https://arxiv.org/abs/2609.14741>

An engineer's sentence is compiled against a declared property registry into a
typed query over a catalogue of solved metal–cell combinations. If the feasible
set is non-empty the system returns a catalogue row that a second solver can
rebuild. If it is empty the system returns **why** — the minimal unsatisfiable
subsets, the correction sets, and the slack of a repair — rather than the
nearest row.

This repository holds the data and code of the revised manuscript (v4,
September 2026). The release made with arXiv:2609.14741v1 is commit `b19eb84`.

---

## What is here

```
data/      every artefact the paper's Data availability statement lists
src/       the registry, the reasoner, the homogenisation solver, the checks
```

### `data/`

| File | What it is |
|---|---|
| `catalogue.csv` | The catalogue. 1,536 enumerated rows, 47 columns; the `feasible` flag marks the **1,397** searchable rows. Stores ρ, k\* (3×3), C\* (6×6), derived directional moduli, the void coefficients `B11/B22/B33`, and the pore diffusivities `D11/D22/D33`. `n` is the grid a row was solved on, `tie` the rule that classifies voxels tied with the isovalue (`legacy`, `include` or `exclude`; see `src/metahomog/tpms.py`), and `builder` the solver that built the row (`gpu` for the 430 thin-walled rows re-solved on finer grids, `cpu` otherwise). |
| `dstar_complete_pore.csv` | The 1,397 complete-periodic-pore D\* solves behind Sec. 5.4. |
| `suite_64.csv` | The frozen typed suite: 64 queries, 49 feasible and 15 empty, with each search policy's result. |
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

Records added with the revised manuscript: `v3_analyses.json` (hidden
priorities, flip margins, cap timing, density-cap sweep) and
`v4_analyses.json` (per-property decision classes, objective-scored repairs,
parse repeat agreement and complete frames, end-to-end re-runs);
`census_wrap.json`, the axes that every solid component of every catalogue
mask wraps; `partial_percolation_cpu_gpu.json`, the closed-form tests on
partly percolating components for both solvers; `refusal_boundary_check.json`,
the three stiffest cells with ρ ≤ 0.15 re-solved at n = 192;
`symmetry_resolve.json`, `thin_resolve*.json`, `thin_wall_convergence*.json`,
`mesh_probe_working_grid.json`, `grid_study_pooled.json` and
`rebuild_rho_check.json`, the row re-solves and grid checks;
`agent_baseline.json`, the tool-using-model comparison; `literature_v3.json`,
the published design-aim sentences through the current prompt.

The `*_preregister.md` files fix each metric **before** its result and are
included unedited, as are `p2_protocol_correction.md` and
`p2_outward_followup.md`, which record where a protocol was corrected after
the fact.

### `data/model_outputs/`

The raw scoring records behind the 308-request parse benchmark and the
vocabulary ablation, for both model SKUs, plus the two registry-derived
baselines. `frames_v3/` holds every parse on the current prompt (three
Flash-Lite runs, two Flash runs) with its frame scores, and
`agent_baseline/` every transcript of the tool-using-model comparison. These
make the one learned stage checkable without re-running the model. See the
README in that directory.

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

### `data/verification_20260926/`

Records from the revision: the run of the sixteen-test solver suite, the
Poisson-ratio re-solves behind Table E.6 of the revised manuscript, and the drivers that wrote the
revised result records. See the README in that directory.

### `data/openfem/`

The second-library check: `scikit-fem` and TetGen run against the catalogue
solver on the same voxel occupancy, in both physics. Drivers, raw logs, and
`openfem_checks.json`. The logs are the original run output and state their own
limitation — the driver is ours, not an independent third-party run.

### `src/`

| Package | Contents |
|---|---|
| `metahomog/` | Geometry and physics. `tpms.py` builds the implicit cells and holds the tie rule, `homogenize.py` solves the periodic cell problem on trilinear hexes, `pore_phase.py` points the same solver at the void to get D\*, and **`validate.py` is the sixteen-test suite** of Sec. 4.2 and Appendix E.3 of the revised manuscript. |
| `metagpt/` | The contract and the search. **`schema.py` is the property registry** — the single declaration from which both the language-model prompt and the deterministic evaluator are generated. `retrieval.py` is the retrieve-or-refuse reasoner with MUS and MCS enumeration, `refusal.py` the refusal objects, `materials.py` the 19-metal handbook layer, `evaluate.py` the parse benchmark harness, `verify.py` the GPU-versus-CPU re-solve check, **`test_contract.py` the tests of the registry contract**, and `resolve_thin_rows.py` and `resolve_symmetry_rows.py` the row re-solves. |

---

## Reproducing the checks

Everything except the parse stage is deterministic arithmetic over the frozen
catalogue and needs no network and no API key.

```bash
pip install numpy scipy pandas matplotlib
cp data/catalogue.csv src/metagpt/            # the code reads the catalogue from its own folder
cd src/metahomog && python validate.py        # the sixteen-test solver suite
cd ../metagpt    && python test_contract.py   # the registry-contract tests
python verify.py --uid 0                      # re-solve one catalogue row (seconds)
```

Without `--uid`, `verify.py` re-solves eight random rows; a row on the
n = 128 grid can take many minutes.

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
