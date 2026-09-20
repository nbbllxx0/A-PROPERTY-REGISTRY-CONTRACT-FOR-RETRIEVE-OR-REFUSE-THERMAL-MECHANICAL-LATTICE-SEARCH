# Language-model output records

The raw scoring records behind the parse benchmark and the vocabulary
ablation, so both can be checked without re-running the model.

The parse benchmark is the "Concept F1 on 308 template-generated requests"
table — **Table 15** in the arXiv preprint (2609.14741v1).

| File | What it holds |
|---|---|
| `bench_eval_flash.json` | Gemini 3.5 Flash on the 308-request benchmark, two runs |
| `bench_eval_flashlite.json` | Gemini 3.5 Flash-Lite on the same benchmark, five runs |
| `eval_results.json` | per-request scoring records for the benchmark |
| `baseline_results.json` | the registry-derived keyword table and the TF-IDF 1-NN baseline |
| `ablation_literal.json` | vocabulary-size ablation, literal phrasings |
| `ablation_para.json` | vocabulary-size ablation, paraphrased phrasings |
| `refusal_flash.json` | refusal-path records, Flash |
| `refusal_flashlite.json` | refusal-path records, Flash-Lite |

The parse stage is the only learned component, and it is the only stage
these records cover. Retrieval is a boolean mask intersection followed by
stable sorts over a frozen table, so it is deterministic and reproducible
from `data/` and `src/` alone, with no model and no network.

Run-to-run variation sits entirely in the parse: the two Flash runs returned
identical scores, and the five Flash-Lite runs spread 1.95 percentage points.

Prompts are generated from the property registry in `src/metagpt/schema.py`;
they are not hand-written, and no API key is required to read these files.
