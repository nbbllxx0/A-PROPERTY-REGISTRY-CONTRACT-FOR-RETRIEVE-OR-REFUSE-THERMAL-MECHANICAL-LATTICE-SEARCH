#!/usr/bin/env bash
# Recompute every catalogue-dependent result of the v4 manuscript, in order.
# Run after metagpt/catalogue.csv is final. API runs (scripts/agent_baseline.py,
# scripts/run_frames.py --llm, scripts/run_literature_v3.py without --research)
# are run separately; their stored outputs are re-scored here.
# SKIP_SOLVES=1 skips the two steps that rebuild or re-solve geometry and do
# not depend on the search code (check_rebuild_rho, run_verify_sample: about
# three hours of CPU).
set -euo pipefail
PAPER="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(dirname "$PAPER")"
PY="${PY:-python}"
LOG="$PAPER/data/run_all_v4.log"
: > "$LOG"
step() { echo "== $*" | tee -a "$LOG"; "$@" >> "$LOG" 2>&1; }
cd "$PAPER"
step "$PY" "$ROOT/metagpt/test_contract.py"
step "$PY" scripts/aei_upgrade_analyses.py
step "$PY" scripts/aei_redteam_followup.py
step "$PY" scripts/binding_set_eval.py
step "$PY" scripts/test_mus_oracle.py
step "$PY" scripts/find_threeway_mus.py
step "$PY" scripts/p1a_d_suite.py
step "$PY" scripts/p1b_fault_eval.py
step "$PY" scripts/p2_repair_witness.py
step "$PY" scripts/p2_revision.py
step "$PY" scripts/test_print_sigfigs.py
if [ "${SKIP_SOLVES:-0}" != "1" ]; then
  step "$PY" scripts/check_rebuild_rho.py
fi
step "$PY" scripts/v3_analyses.py
step "$PY" scripts/v4_analyses.py
step "$PY" scripts/census_wrap.py
step "$PY" scripts/test_partial_percolation.py --gpu
step "$PY" scripts/run_literature_v3.py --research
if [ "${SKIP_SOLVES:-0}" != "1" ]; then
  step "$PY" scripts/run_verify_sample.py
fi
step "$PY" scripts/make_appendices.py
step "$PY" make_figs.py
step "$PY" scripts/fill_numbers.py --write
step "$PY" scripts/fill_numbers.py --check
echo "done" | tee -a "$LOG"
