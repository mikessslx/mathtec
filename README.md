# MathTEC

> A multi-agent framework for mathematical text error correction. It combines DEEP-style multi-prompt detection, multi-agent debate, diff auditing and renderer-based LaTeX validation. All data reported in this paper were from MathTEC.

## What Is Included

```
mathtec/
├── config.py                  # API keys, model configs, pipeline params
├── pipeline.py                # Main 3-stage pipeline orchestration
├── evaluate.py                # Metrics: char-F1, sentence accuracy, GLEU
├── run_experiments.py         # Ablation experiment runner
├── run_gateway.py             # Gateway-based quick-start entry point
├── agents/
│   ├── base.py                # Unified LLM client (OpenAI/Anthropic/DeepSeek)
│   ├── detector.py            # Stage 1: DEEP-style multi-prompt detection
│   ├── corrector.py           # Stage 2: Multi-agent debate correction
│   └── judge.py               # Debate judge (selects best correction)
├── validators/
│   ├── diff_audit.py          # Diff-based minimal-edit audit
│   └── render_check.py        # KaTeX/MathJax renderer verification
├── tools/
│   └── render_check.js        # Node helper for renderer checks
├── data/
│   ├── error_injector.py      # 5-type error injection (spelling/OCR/variable/handwriting)
│   └── construct_benchmark.py # MathTEC-Bench builder
├── paper/
│   └── paper.tex              # ACL-format paper template
└── results/                   # Experiment outputs
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
npm install
```

2. Set API keys:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export DEEPSEEK_API_KEY="sk-..."
```

## Quick Start

Choose one of the following workflows.

### Option A: Gateway Workflow

> The workflow was used in this paper.

1. Edit `GATEWAY_BASE_URL`, `GATEWAY_API_KEY` and `MODELS` at the top of `run_gateway.py`.

2. Run a quick test:

```bash
python run_gateway.py test
```

3. Build the benchmark:

```bash
python run_gateway.py benchmark
```

4. Run the experiments:

```bash
python run_gateway.py experiments
```

### Option B: Direct Scripts

1. Build the benchmark:

```bash
python -m data.construct_benchmark
```

This generates `data/mathtec_bench.json` with roughly 500 samples across 10 math topics and 5 error types.

2. Run the full pipeline on a single text:

```python
import asyncio
from pipeline import MathTECPipeline

pipeline = MathTECPipeline()

text = "For all $x \\in \\mabb{R}$, the funciton $f(x) = x^{2}$ is continuos."
result = asyncio.run(pipeline.process(text))
print(result.corrected_text)
# -> For all $x \in \mathbb{R}$, the function $f(x) = x^{2}$ is continuous.
```

3. Run experiments:

```bash
python run_experiments.py --benchmark data/mathtec_bench.json --output-dir results

# Quick test with 20 samples:
python run_experiments.py --max-samples 20
```

For a single configuration:

```bash
python run_experiments.py --experiment E  # D + renderer check
python run_experiments.py --experiment F  # D + diff audit
python run_experiments.py --experiment G  # D + diff + renderer
python run_experiments.py --experiment A  # baseline
```

4. Evaluate results:

```bash
python evaluate.py data/mathtec_bench.json results/G_diff_plus_renderer_results.json results/eval.json
```


## Experiments

| Config | Detection | Debate | Post-check | Description |
|--------|-----------|--------|------------|-------------|
| A | 1 prompt | None | None | Single-model baseline |
| B | 6 prompts (DEEP) | None | None | Multi-prompt detection only |
| C | 1 prompt | 3 agents, 2 rounds | None | Multi-agent debate only |
| D | 6 prompts | 3 agents, 2 rounds | None | DEEP + debate baseline |
| E | 6 prompts | 3 agents, 2 rounds | MathJax render gate | D + renderer validation |
| F | 6 prompts | 3 agents, 2 rounds | Diff audit | D + minimal-edit diff validation |
| G | 6 prompts | 3 agents, 2 rounds | Diff audit + MathJax | Combined validation stack |

## Configuration

Create `config.yaml` to override defaults:

```yaml
models:
  detector:
    provider: openai
    model: gpt-5.4
    api_key_env: OPENAI_API_KEY
    temperature: 0.2
  agent_a:
    provider: openai
    model: gpt-5.4
    api_key_env: OPENAI_API_KEY
  agent_b:
    provider: anthropic
    model: claude-sonnet-4.6
    api_key_env: ANTHROPIC_API_KEY

pipeline:
  detection:
    num_prompts: 6
    agreement_threshold: 2
  debate:
    num_rounds: 2
    num_agents: 3
  render_check:
    enabled: true
    backend: mathjax
  diff_audit:
    enabled: true
    use_llm_review: true
```

## Notes

- Renderer validation tools require `npm install`.
- All data reported in this paper were generated primarily with `run_gateway.py`.
