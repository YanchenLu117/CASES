# Third-party code and data

CASES itself is MIT-licensed. It intentionally ships **no** third-party code in
its own tree; the following official repositories are fetched at pinned
revisions at setup time into `external/repos/` (git-ignored) and are governed
by their own licenses.

| Component | Upstream | License | Pin / fetch |
| --- | --- | --- | --- |
| HypoSpace evaluator (boolean / causal / 3d domains) | [`CTT-Pavilion/_HypoSpace`](https://github.com/CTT-Pavilion/_HypoSpace) @ `c69e9318` (ICML 2026 spotlight, arXiv:2510.15614) | MIT | `external/repos.lock.yaml`, fetched by `scripts/setup_external.py` |
| Buchwald–Hartwig yield table + descriptors | [`doylelab/rxnpredict`](https://github.com/doylelab/rxnpredict) (master) | MIT | optional, `scripts/download_benchmarks.sh` |
| GB1 active-learning specialist assets | [`jsunn-y/ALDE`](https://github.com/jsunn-y/ALDE) (main) | MIT | optional, `scripts/download_benchmarks.sh` |
| GB1 clustering + full measured table (`Input/GB1.xlsx`, 149,361 variants) | [`WeilabMSU/CLADE`](https://github.com/WeilabMSU/CLADE) (main) | MIT | optional, `scripts/download_benchmarks.sh` |
| ResearchBench tasks + official `matched_score` (0–5) judge scorer | [`ankitala/ResearchBench`](https://github.com/ankitala/ResearchBench) (ACL Findings 2026, arXiv:2503.21248; dataset [`ankilok/ResearchBench`](https://huggingface.co/datasets/ankilok/ResearchBench)) | MIT | referenced by the `cases_exp` ResearchBench lane; not vendored |

Pins for every fetched checkout are recorded in `external/repos.lock.yaml`
(`python scripts/external_lock.py --verify` re-checks them against disk).

Python dependencies installed from PyPI (numpy, scipy, scikit-learn, pandas,
openpyxl, sympy, openai, …) remain under their own respective licenses; see
`pyproject.toml` and `requirements-benchmarks.txt` for the list.
