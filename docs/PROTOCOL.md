# The CASES protocol — audit gates in brief

This note summarizes the protocol-level invariants enforced by the code in
`src/cases`.  Config keys reference the frozen YAML files under `configs/`.

## 1. Construct → Audit → Explore → Recover

A campaign proceeds in rounds around a persistent **substrate state**
(`src/cases/core/state.py`, `model.py`):

1. **Construct** — a representation of the current solution space is induced
   from the evidence so far (`core/representation.py`, `core/reground.py`).
   Paradigms, from least to most expressive: `fixed_embedding`,
   `fixed_symbolic`, `induced_no_logic`, `full`.
2. **Audit** — the proposal is admitted only if it passes the admission gates
   (`protocol/admission.py`): a symbolic validity check over the declared
   logic (`LogicValid`), plus an actor-visible fidelity vector meeting its
   registered thresholds.  Completeness is stamped `UNVERIFIED` during the
   campaign — the actor never receives evaluator feedback.
3. **Scoped exploration** — a frozen controller score
   (`protocol/controller.py`) ranks candidates by saturation, utility and
   open-coverage gaps under resource caps (`protocol/fairness.py`).
4. **Recover** — posterior backends (`recovery/`) expose a calibrated
   `solution_probability`; the solution set is always a *high-probability*
   cut (`S_t(eta)`), never silently the whole belief space.
5. **Readout** — the host policy sees only the frozen readout schema
   (`protocol/readout.py`) under an explicit token budget; revision
   (`core/revision.py`) re-enters at the admission gate.

## 2. Certificates

`certification/` provides the pre-admission certification machinery:

- `completeness.py` — Level-2 completeness reports via congruence closure over
  the declared relation set (`certify_level2`).
- `interventions.py`, `masking.py` — intervention-consistency and masking
  checks backing the leakage discipline.

## 3. Leakage discipline

- Evaluator-only truths (top-set thresholds, gold cardinalities, gold
  structures, hidden labels) never appear in prompts, evidence, state, or
  readouts; oracles are loaded only through GT-stripping loaders
  (`adapters/*/data.py`), and gold is handed directly to metrics.
- The finite oracles (`campaigns/bh_oracle.py`, `campaigns/gb1_oracle.py`)
  hide gold membership from acquisition arms; tests assert oracle purity.

## 4. Budgets and fairness

- Every LLM call, token spend, query and invalid proposal is ledgered
  (`core/model.py` metered backend, `llm/`).
- Equal-resource comparisons run under seven frozen caps
  (`protocol/fairness.py`); an incomplete run **is** the primary
  equal-resource result for the full chain.
- Significance uses paired seeds with SD floors and TOST equivalence bounds
  (`stats/`); seed counts are frozen per benchmark (BH 10 paired, GB1 30).

## 5. Preregistration & artifacts

- `prereg/registry.py` builds a versioned, machine-readable
  `preregistration_registry.json`, hashed (SHA-256) and frozen **before** any
  final outcome is accessible; unset mandatory fields block the affected
  scored cell.
- `runout/` archives each run into the standard artifact layout (manifest with
  token budgets + prompt hashes, ledger, figure slots) — see
  `tests/core/test_v7_runout.py` for the exact contract.

## 6. Pinned official evaluators

Official evaluator code is never vendored: `external/repos.lock.yaml` pins the
HypoSpace evaluator repository by commit, `scripts/setup_external.py` fetches
it, and `scripts/external_lock.py --verify` re-checks the checkout against the
lock at CI time.
