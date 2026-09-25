# CASES — Constructing and Auditing Scientific Exploration Substrates

CASES is a research framework for running **scientific exploration campaigns**
against benchmark oracles under an explicit audit discipline.  Instead of
letting an agent explore a benchmark directly, CASES first **constructs** an
explicit solution-space substrate from the scarce observations gathered so far,
then **audits** that substrate (fidelity certificates, leakage discipline,
budget accounting) before it is allowed to steer exploration, and only then
runs **scoped exploration** and **recovery** of the solution set with honest,
well-formed metrics.

The core runs fully **without an LLM** (deterministic, no network); LLM-backed
hypothesis generation plugs in through any OpenAI-compatible endpoint.

Included benchmark lanes:

- **Buchwald–Hartwig (BH)** — reaction-yield optimization over the official
  5,568-experiment table; graph-propagation recovery vs. feature-GP baselines.
- **GB1** — combinatorial protein fitness over the measured 149,361-variant
  table; Hamming-1 propagation recovery.
- **HypoSpace** — set-valued hypothesis generation (boolean / causal / 3d
  domains) through the *official* evaluator repository, pinned by commit.
- **ResearchBench** (`cases_exp` package) — research-hypothesis composition
  episodes (E1 methods: direct refine / fixed schema / editable structured /
  CASES substrate lanes with the one-edit audit loop) over the official
  ResearchBench tasks; scored by the official `matched_score` (0–5) judge
  with a frozen cross-family evaluation model.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[recovery,dev]"        # core + GP baselines + pytest
pip install -r requirements-benchmarks.txt   # benchmark/LLM extras
python scripts/setup_external.py        # fetch the pinned official HypoSpace repo
```

## Smoke test (no LLM, no downloads)

```bash
./smoke_test.sh
```

constructs a model on the toy Boolean domain, ingests a batch of hypotheses,
recovers the posterior, and emits a budgeted readout — the whole loop with
zero network access (it fetches the pinned HypoSpace evaluator once if
missing).

Run the full test suite (~340 tests; benchmark-data-dependent tests skip
themselves when the data has not been downloaded):

```bash
pytest -q
```

## BH / GB1 scored campaigns

```bash
# one-time: official benchmark data (both MIT-licensed upstreams)
./scripts/download_benchmarks.sh

export CASES_LLM_BASE_URL="https://YOUR-ENDPOINT.example.com/v1"
export CASES_LLM_API_KEY="..."            # never committed
export CASES_LLM_MODEL="zai-org/GLM-5.3-Flash"

python scripts/run_bh_gb1.py --project bh  --phase pilot   # 8-seed pilot
python scripts/run_bh_gb1.py --project bh  --phase scored --jobs 24
python scripts/run_bh_gb1.py --project gb1 --phase scored --jobs 24
python scripts/run_bh_gb1.py --project all --phase aggregate
python scripts/run_bh_gb1.py --project all --phase figs
```

Arms, seeds, checkpoints and batch schedules are frozen in
`configs/bh/default.yaml` and `configs/gb1/default.yaml`.

## ResearchBench lane (`cases_exp`)

The E1 hypothesis-composition methods ship in `cases_exp/methods/e1_methods.py`
(arm registry `E1_METHODS`: `direct_refine`, `fixed_schema`, `editable_structured`,
`cases_static`, `cases_full`, plus the `h5_nc` / `h5_nce` factorial ablation arms):

```python
from cases_exp.models.openai_compat import LLMClient, Budget
from cases_exp.methods.e1_methods import E1_METHODS

client = LLMClient(base_url=..., model=..., api_key=...)   # any OpenAI-compatible endpoint
method = E1_METHODS["cases_full"](client, Budget(...))     # construct -> audit -> compose
result = method.run(task)   # task: research question + background survey (+ empty inspirations)
print(result.final_answer, result.representation)
```

Tasks come from the official ResearchBench release (github.com/ankitala/ResearchBench,
MIT — see THIRD_PARTY.md); scoring uses the official `matched_score` (0–5) judge prompt,
run frozen at temperature 0 with a cross-family evaluation model. The generation side
never sees gold hypotheses.

Scoring is a thin wrapper around the official `score-generate` CLI (install the
upstream package once: `pip install git+https://github.com/ankitala/ResearchBench`):

```bash
export CASES_JUDGE_BASE_URL="https://YOUR-JUDGE-ENDPOINT.example.com/v1"
export CASES_JUDGE_API_KEY="..."
export CASES_JUDGE_MODEL="qwen38-27b"      # the paper's frozen judge setting

python scripts/run_rb_judge.py --pred generations.jsonl --data tasks.jsonl --out scores.json
```

`--pred` rows carry `{"sample_id", "final_hypothesis"}`; `--data` rows carry
`{"sample_id", "gold_hypothesis", "gold_key_points"}`. The official CLI fixes
temperature at 0 and skips already-scored rows (resumable); the judge endpoint
is read from the environment and never hardcoded.

The package also ships the native discovery-episode runners
(`cases_exp/benchmarks/run_e2.py` / `run_e2b.py`). `--model` selects a named *layer*;
each layer is configured purely through environment variables (no model id is
hardcoded anywhere):

```bash
export CASES_LAYER_MYLLM_BASE_URL="https://..."
export CASES_LAYER_MYLLM_API_KEY="..."
export CASES_LAYER_MYLLM_MODEL="..."
python -m cases_exp.benchmarks.run_e2 --model myllm ...
```

## Configuration & environment

| Variable | Purpose |
| --- | --- |
| `CASES_LLM_BASE_URL` / `CASES_LLM_API_KEY` / `CASES_LLM_MODEL` | OpenAI-compatible endpoint for the legacy `glm` track (see `configs/llm_endpoints.example.json`) |
| `CASES_LAYER_<NAME>_*` | endpoint/key/model/rate caps for roster layers (`cases.llm.layers`) |
| `CASES_HYPOSPACE_REPO` | override the official HypoSpace checkout location |
| `CASES_BH_DATA_TABLE` | override the BH `data_table.csv` location |
| `CASES_GB1_XLSX` | override the GB1 measured-table xlsx location |

Config files: `configs/bh/default.yaml`, `configs/gb1/default.yaml` (frozen
arms/criteria), `configs/hypospace/`, `configs/paradigm/`, `configs/recovery/`.

## Repository layout

```
src/cases            core library
  core/              evidence log, representation induction, model facade
                     (CASESModel: update → recover → readout)
  protocol/          admission/revision gates, controller, campaign schema,
                     fairness metering, prereg-mapped metrics
  certification/     completeness certificates, intervention & masking checks
  state_strategies/  persistent-substrate state strategies
  campaigns/         BH / GB1 campaign engines (oracles, GP baselines,
                     recovery evaluation)
  baselines/         reference arms behind the shared campaign tool schema
  recovery/          posterior backends (laplacian / mean / rbf-gp / …)
  stats/             paired-seed significance tests, TOST equivalence
  llm/               OpenAI-compatible client, layer registry, metering
  metrics/ api/ runner/ runout/ prereg/ data/ logging/ infra/
  adapters/          hypospace · bh · gb1 · gate · _template + registry
src/cases_exp        ResearchBench composition methods (E1) + discovery-episode
                     runners (benchmarks, methods, substrate, evaluators, dispatcher)
scripts/             run_bh_gb1.py, setup_external.py, external_lock.py,
                     download_benchmarks.sh, jsonl_writer.py
configs/ tests/ docs/
external/repos.lock.yaml   pinned third-party checkouts (see THIRD_PARTY.md)
```

## Design invariants (short version)

1. **Audit before explore** — a substrate may only steer acquisition after its
   fidelity certificate and leakage checks pass (protocol admission gates).
2. **No leakage by construction** — evaluator-only truths (top-set thresholds,
   label cardinalities, gold structures) never reach prompts, state, or
   readouts; ground truth is loaded only through GT-stripping loaders.
3. **Everything metered** — LLM calls, token budgets, queries and invalid
   proposals are ledgered; budgets freeze at the terminal step.
4. **Pinned externals** — official benchmark/evaluator code is fetched at
   pinned commits (`scripts/external_lock.py --verify`), never vendored.
5. **Outcome-blind freezes** — arms/seeds/criteria are frozen in prereg-style
   configs before results are seen.

See `docs/PROTOCOL.md` for the protocol-level details.

## License & third-party code

MIT (see `LICENSE`).  The official HypoSpace evaluator and the BH/GB1 upstream
repositories are fetched at pinned revisions and remain under their own
licenses — see `THIRD_PARTY.md`.
