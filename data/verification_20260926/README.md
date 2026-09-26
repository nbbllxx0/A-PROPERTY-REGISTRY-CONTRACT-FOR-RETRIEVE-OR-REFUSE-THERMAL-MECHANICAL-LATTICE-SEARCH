# Verification records, 26 September 2026

Records added with the revised manuscript (v4). The result records
themselves are in `data/analysis/` and `data/model_outputs/`; this directory
holds the run logs and the drivers that wrote them.

| File | What it holds |
|---|---|
| `validate_ladder.log` | A run of `src/metahomog/validate.py`, the sixteen-test solver suite: 16 of 16 pass in about 34 s. |
| `poisson_full_range_16cells.log` | E11\*/Es of sixteen cubic and tetragonal cells at ν = 0.15, 0.30 and 0.45 (grid 24³), with the rank shift and the regret on the returned cell (Table E.6 of the revised manuscript). |
| `poisson_metal_band_16cells.log` | The same sixteen cells at ν = 0.29, 0.30 and 0.34, the band of the table's metals. |
| `poisson_forty_cells.log` | Forty cells at ν = 0.15, 0.30 and 0.45 (grid 32³), including the E33/E11 drift. |
| `scripts/` | The drivers that wrote the revised result records. `run_all_v4.sh` gives the order; its last two steps build the manuscript's appendices and figures and are not included. |

Runs that call a language model (`agent_baseline.py`, `run_frames.py --llm`,
`run_literature_v3.py` without `--research`) need a model key, which is not
included; their stored outputs are in `data/model_outputs/` and are re-scored
without a model call.

The scripts are copied as run. Their import paths point at the working tree:
`metagpt/` and `metahomog/` are `src/metagpt/` and `src/metahomog/` in this
repository, and the paper's `data/` directory is `data/analysis/`.
