# MathTEC

> A three-stage pipeline that corrects errors in mathematical text mixing prose with inline LaTeX, together with **MathTEC-Bench**, a synthetic benchmark of 500 English mathematical statements covering five error types.

**Paper:** [MathTEC: A Multi-Agent Framework for Mathematical Text Error Correction](https://doi.org/10.1109/ICCCS69761.2026.11613421) — ICCCS 2026 (IEEE), pp. 1154–1159

## Results

Best configuration **G** (six-prompt detection + three-agent debate + diff audit + MathJax render gate) against the single-prompt single-agent baseline **A**:

| Config | Prec. | Rec. | F1 | Sent-Acc |
|---|---|---|---|---|
| A: Single prompt + single agent | 91.1 | 89.6 | 89.3 | 85.0 |
| D: DEEP detection + debate | 93.2 | 92.2 | 92.0 | 88.0 |
| **G: D + diff audit + MathJax** | **94.0** | **92.5** | **92.4** | **88.4** |

Improvement from A to G: **+3.1 char-F1** and **+3.4 sentence-accuracy** points.

Character-level F1 by error type (percentages):

| Config | Spell. | OCR-Sym. | OCR-Form. | Variable | Handwrite |
|---|---|---|---|---|---|
| A | 91.5 | 76.9 | 69.3 | 80.5 | 90.6 |
| D | 95.0 | 83.2 | 77.2 | 90.9 | 91.0 |
| G | 94.1 | 83.4 | 77.4 | 90.9 | 92.0 |

Every number above is reproduced by a command in [Reproducing the results](#reproducing-the-results) — **offline, with no API key**.

## Quick start

Verify the headline numbers in seconds — **no API key, no model download, no GPU, and no `pip install` beyond numpy**, because the released per-sample outputs and the evaluator both run offline:

```bash
git clone https://github.com/mikessslx/mathtec.git && cd mathtec
pip install numpy # all the offline path needs
python reproduce_tables.py
```

This prints Table 1 (benchmark statistics), Table 2 (main results) and Table 3 (F1 by error type).

## Installation

Tested with Python 3.11, 3.12 and 3.14; the reported experiments used 3.14. No GPU is required — all inference goes through LLM APIs. Node.js is needed only for the MathJax render check.

```bash
git clone https://github.com/mikessslx/mathtec.git && cd mathtec
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
npm install # optional: only for the MathJax/KaTeX render gate (E and G)
```

## Reproducing the results

### Without an API key (seconds)

The repository ships the per-sample outputs for all seven configurations, so the paper's tables are rebuilt offline directly from those files:

```bash
python reproduce_tables.py
```

| Paper element | Command | Runtime | Hardware |
|---|---|---|---|
| Table 1 — benchmark statistics | `python reproduce_tables.py` | <1 s | CPU |
| Table 2 — main ablation results | `python reproduce_tables.py` | <1 s | CPU |
| Table 3 — F1 by error type | `python reproduce_tables.py` | <1 s | CPU |

You can also re-run the evaluation script on any shipped result file directly:

```bash
python evaluate.py data/mathtec_bench.json results/G_diff_plus_renderer_results.json
```

### Re-running the experiments (requires API access)

This regenerates the per-sample outputs from scratch and **costs real API credit**. The reported run took about 7.4 hours of wall-clock time for all seven configurations (single-precision totals: 442 minutes, ~63 minutes per configuration at 3 concurrent requests).

```bash
export GATEWAY_BASE_URL="https://your-endpoint.example.com/v1"
export GATEWAY_API_KEY="..."
python run_gateway.py experiments # all seven configurations -> results/
```

To run one configuration, or a quick subset:

```bash
python run_experiments.py --experiment G
python run_experiments.py --max-samples 20
```

> **Note on models.** The paper's results were produced with `run_gateway.py`, which
> routes every role (detector, three debate agents, judge) to a single model — the
> default is `gpt-5.4` for all of them, as configured in `MODELS` at the top of
> `run_gateway.py`. The separate `DEFAULT_MODELS` table in `config.py` instead mixes
> providers (`claude-sonnet-4.6` for `agent_b`, `deepseek-v3.2` for `agent_c`);
> `run_experiments.py` uses that mixed table when run standalone. Reproducing the
> paper's numbers therefore requires the gateway path, or setting the environment
> variables `GATEWAY_MODEL` accordingly. The backbone model is not recorded inside
> the shipped `results/*.json` files.

## Data

`data/mathtec_bench.json` is the released benchmark: 500 samples, 745 injected errors, 59 clean samples (no injected error, used to measure over-correction).

Statistics reproduced by `python reproduce_tables.py`:

| Statistic | Value |
|---|---|
| Total samples | 500 |
| Clean samples | 59 (11.8%) |
| Injected errors | 745 (1.49 per sample) |
| Average words per sample | 24.2 |
| Average inline math expressions per sample | 3.85 |
| Injected error occurrences | spelling 221, OCR-symbol 175, OCR-formula 77, variable 38, handwriting 234 |
| Samples with 1 / 2 / 3 injected errors | 218 / 142 / 81 |

Each sample has the fields `id`, `original` (clean reference), `corrupted` (noisy input), `errors` (list of injected edits with type, spans and offsets), `num_errors` and `error_types`.

**Source of the text.** The clean statements were generated with GPT-5.4 and then corrupted by the rule-based `data/error_injector.py`. They are therefore synthetic, not scraped from textbooks or exam banks; see `data/construct_benchmark.py` for the generation and injection procedure.

**Two properties of the released data worth knowing:**

- **No topic labels.** The samples are drawn from 10 topics, but the released JSON does not record a per-sample topic field. The count of 10 is available only in `config.BenchmarkConfig.math_topics`. The paper notes this limitation.
- **Realised error mix differs from the nominal weights.** The generator requests spelling 0.25, OCR-symbol 0.25, OCR-formula 0.20, variable 0.05 and handwriting 0.25, but injection is rejected when no valid site exists in a given sentence and is not retried. Measured success rates per type are roughly spelling 97%, OCR-symbol 72%, OCR-formula 41%, variable 72% and handwriting 100%, which is why OCR-formula ends up at 10.3% of the 745 realised errors rather than 20%.

The benchmark is synthetic and released under CC BY 4.0 (see [License](#license)). It contains no personal data.

**Companion repository.** The QLoRA fine-tuning study that trains on this benchmark — and ships the same data as `source`/`target` train/val/test splits — is [mikessslx/mathcorrector](https://github.com/mikessslx/mathcorrector).

## Correct one sentence

Requires API access:

```python
import asyncio
from pipeline import MathTECPipeline

pipeline = MathTECPipeline()
text = "For all $x \\in \\mabb{R}$, the funciton $f(x) = x^{2}$ is continuos."
result = asyncio.run(pipeline.process(text))
print(result.corrected_text)
```

## Repository structure

```text
config.py              model configs, pipeline parameters, benchmark config
pipeline.py            three-stage orchestration
evaluate.py            char-level P/R/F1 and sentence accuracy
reproduce_tables.py    rebuild the paper's tables offline (no API key)
run_experiments.py     seven-configuration ablation runner
run_gateway.py         gateway entry point (the path used for the paper)
agents/                unified LLM client, detector, corrector, judge
validators/            diff audit, MathJax/KaTeX render check
tools/render_check.js  Node helper invoked by the render check
data/                  benchmark builder, error injector, released benchmark
results/               shipped per-sample outputs and summaries for A–G
```

## The seven configurations

| Config | Detection | Debate | Post-check |
|---|---|---|---|
| A | 1 prompt | — | — |
| B | 6 prompts (DEEP) | — | — |
| C | 1 prompt | 3 agents, 2 rounds | — |
| D | 6 prompts | 3 agents, 2 rounds | — |
| E | 6 prompts | 3 agents, 2 rounds | MathJax render gate |
| F | 6 prompts | 3 agents, 2 rounds | diff audit |
| G | 6 prompts | 3 agents, 2 rounds | diff audit + MathJax |

## Citation

```bibtex
@inproceedings{li2026mathtec,
  title     = {{MathTEC}: A Multi-Agent Framework for Mathematical Text Error Correction},
  author    = {Li, Xiang},
  booktitle = {2026 11th International Conference on Computer and Communication Systems (ICCCS)},
  pages     = {1154--1159},
  year      = {2026},
  publisher = {IEEE},
  doi       = {10.1109/ICCCS69761.2026.11613421}
}
```

GitHub also renders this from [`CITATION.cff`](CITATION.cff).

## License

- **Code** in this repository is released under the [MIT License](LICENSE). Copyright 2025-2026 Xiang Li.
- **MathTEC-Bench** (`data/mathtec_bench.json`) is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The paper itself is published by IEEE and is **not** redistributed here; please use the DOI link above.

## Acknowledgements

The author thanks Prof. Antti Rasila for guidance and support throughout this research.
