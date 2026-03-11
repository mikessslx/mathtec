import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    DEFAULT_MODELS,
    BenchmarkConfig,
    DebateConfig,
    DetectionConfig,
    DiffAuditConfig,
    ModelConfig,
    PipelineConfig,
    RenderCheckConfig,
)
from evaluate import evaluate_benchmark
from pipeline import MathTECPipeline, save_results

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    name: str
    description: str
    pipeline_config: PipelineConfig
    models: Dict[str, ModelConfig]


def get_ablation_configs(
    base_models: Optional[Dict[str, ModelConfig]] = None,
) -> List[ExperimentConfig]:
    """Define all experiment configurations including ablations."""
    if base_models is None:
        base_models = DEFAULT_MODELS
    base_models = base_models.copy()

    experiments = []

    # Experiment A: Single-model single-prompt baseline
    single_model_only = base_models.copy()
    single_model_only.pop("agent_b", None)
    single_model_only.pop("agent_c", None)
    experiments.append(ExperimentConfig(
        name="A_single_prompt_baseline",
        description="Single model, single prompt, no debate rounds",
        pipeline_config=PipelineConfig(
            detection=DetectionConfig(num_prompts=1, agreement_threshold=1),
            debate=DebateConfig(num_rounds=0, num_agents=1),
        ),
        models=single_model_only,
    ))

    # Experiment B: Single model + DEEP multi-prompt detection
    experiments.append(ExperimentConfig(
        name="B_deep_detection",
        description="Single model, DEEP multi-prompt detection, no debate rounds",
        pipeline_config=PipelineConfig(
            detection=DetectionConfig(num_prompts=6, agreement_threshold=2),
            debate=DebateConfig(num_rounds=0, num_agents=1),
        ),
        models=single_model_only,
    ))

    # Experiment C: Multi-agent debate with a single detection prompt
    experiments.append(ExperimentConfig(
        name="C_multi_agent_debate",
        description="Multi-agent debate (3 models), single detection prompt",
        pipeline_config=PipelineConfig(
            detection=DetectionConfig(num_prompts=1, agreement_threshold=1),
            debate=DebateConfig(num_rounds=2, num_agents=3),
        ),
        models=base_models,
    ))

    # Experiment D: DEEP detection + multi-agent debate
    experiments.append(ExperimentConfig(
        name="D_deep_plus_debate",
        description="DEEP detection + multi-agent debate",
        pipeline_config=PipelineConfig(
            detection=DetectionConfig(num_prompts=6, agreement_threshold=2),
            debate=DebateConfig(num_rounds=2, num_agents=3),
        ),
        models=base_models,
    ))

    # Experiment E: Experiment D + renderer verification
    experiments.append(ExperimentConfig(
        name="E_renderer_check",
        description="D + MathJax renderer verification gate",
        pipeline_config=PipelineConfig(
            detection=DetectionConfig(num_prompts=6, agreement_threshold=2),
            debate=DebateConfig(num_rounds=2, num_agents=3),
            render_check=RenderCheckConfig(enabled=True, backend="mathjax"),
            diff_audit=DiffAuditConfig(enabled=False),
        ),
        models=base_models,
    ))

    # Experiment F: Experiment D + diff audit
    experiments.append(ExperimentConfig(
        name="F_diff_check",
        description="D + diff-based minimal-edit audit",
        pipeline_config=PipelineConfig(
            detection=DetectionConfig(num_prompts=6, agreement_threshold=2),
            debate=DebateConfig(num_rounds=2, num_agents=3),
            render_check=RenderCheckConfig(enabled=False),
            diff_audit=DiffAuditConfig(enabled=True),
        ),
        models=base_models,
    ))

    # Experiment G: Experiment D + diff audit + renderer verification
    experiments.append(ExperimentConfig(
        name="G_diff_plus_renderer",
        description="D + diff audit + MathJax renderer verification gate",
        pipeline_config=PipelineConfig(
            detection=DetectionConfig(num_prompts=6, agreement_threshold=2),
            debate=DebateConfig(num_rounds=2, num_agents=3),
            render_check=RenderCheckConfig(enabled=True, backend="mathjax"),
            diff_audit=DiffAuditConfig(enabled=True),
        ),
        models=base_models,
    ))

    return experiments


async def run_single_experiment(
    experiment: ExperimentConfig,
    benchmark_path: str,
    output_dir: str,
    max_samples: Optional[int] = None,
):
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {experiment.name}")
    logger.info(f"  {experiment.description}")
    logger.info(f"{'='*60}")

    with open(benchmark_path) as f:
        benchmark_data = json.load(f)
    data = benchmark_data["data"] if "data" in benchmark_data else benchmark_data

    if max_samples:
        data = data[:max_samples]

    texts = [item["corrupted"] for item in data]

    pipeline = MathTECPipeline(
        models=experiment.models,
        pipeline_config=experiment.pipeline_config,
    )

    start = time.time()
    results = await pipeline.process_batch(texts, max_concurrent=3)
    elapsed = time.time() - start

    results_path = f"{output_dir}/{experiment.name}_results.json"
    save_results(results, results_path)

    eval_path = f"{output_dir}/{experiment.name}_eval.json"
    metrics = evaluate_benchmark(benchmark_path, results_path, eval_path)

    summary = {
        "experiment": experiment.name,
        "description": experiment.description,
        "num_samples": len(texts),
        "total_time": elapsed,
        "avg_time_per_sample": elapsed / max(len(texts), 1),
        "sentence_accuracy": metrics.sentence_accuracy,
        "char_precision": metrics.char_precision,
        "char_recall": metrics.char_recall,
        "char_f1": metrics.char_f1,
        "by_error_type": metrics.by_error_type,
    }
    summary.update(_compute_postcheck_stats(results))

    summary_path = f"{output_dir}/{experiment.name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"  Sentence Accuracy: {metrics.sentence_accuracy:.4f}")
    logger.info(f"  Char F1:           {metrics.char_f1:.4f}")
    logger.info(f"  Time:              {elapsed:.1f}s")

    return summary


async def run_all_experiments(
    benchmark_path: str = "data/mathtec_bench.json",
    output_dir: str = "results",
    max_samples: Optional[int] = None,
    models: Optional[Dict[str, ModelConfig]] = None,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    experiments = get_ablation_configs(base_models=models)
    all_summaries = []

    for experiment in experiments:
        try:
            summary = await run_single_experiment(
                experiment, benchmark_path, output_dir, max_samples
            )
            all_summaries.append(summary)
        except Exception as e:
            logger.error(f"Experiment {experiment.name} failed: {e}")
            all_summaries.append({
                "experiment": experiment.name,
                "error": str(e),
            })

    comparison_path = f"{output_dir}/comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    _print_comparison_table(all_summaries)
    logger.info(f"\nAll results saved to {output_dir}/")


def _print_comparison_table(summaries: List[Dict]):
    print("\n" + "=" * 80)
    print(f"{'Experiment':<30} {'Sent-Acc':>10} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Time':>8}")
    print("-" * 80)
    for s in summaries:
        if "error" in s:
            print(f"{s['experiment']:<30} {'FAILED':>10}")
            continue
        print(
            f"{s['experiment']:<30} "
            f"{s['sentence_accuracy']:>10.4f} "
            f"{s['char_precision']:>8.4f} "
            f"{s['char_recall']:>8.4f} "
            f"{s['char_f1']:>8.4f} "
            f"{s.get('total_time', 0):>7.1f}s"
        )
    print("=" * 80)


def _compute_postcheck_stats(results) -> Dict[str, float]:
    if not results:
        return {}

    has_postcheck = any(r.validation_mode for r in results)
    if not has_postcheck:
        return {}

    total = len(results)
    judge_checked = [r for r in results if r.judge_candidate_passed_validation is not None]
    selected_checked = [r for r in results if r.selected_candidate_passed_validation is not None]
    diff_checked = [r for r in results if r.diff_check_passed is not None]
    render_checked = [r for r in results if r.render_check_passed is not None]

    stats = {
        "judge_validation_pass_rate": (
            sum(1 for r in judge_checked if r.judge_candidate_passed_validation) / max(len(judge_checked), 1)
        ),
        "selected_validation_pass_rate": (
            sum(1 for r in selected_checked if r.selected_candidate_passed_validation) / max(len(selected_checked), 1)
        ),
        "alternative_selection_rate": (
            sum(1 for r in results if r.used_alternative_candidate) / total
        ),
        "validation_fallback_rate": (
            sum(1 for r in results if r.validation_fallback_used) / total
        ),
    }

    if diff_checked:
        stats["diff_pass_rate"] = (
            sum(1 for r in diff_checked if r.diff_check_passed) / len(diff_checked)
        )
    if render_checked:
        stats["render_pass_rate"] = (
            sum(1 for r in render_checked if r.render_check_passed) / len(render_checked)
        )

    return stats


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run MathTEC experiments")
    parser.add_argument("--benchmark", default="data/mathtec_bench.json")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples for quick testing")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run single experiment by name (A/B/C/D/E/F/G)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.experiment:
        configs = get_ablation_configs()
        exp_arg = args.experiment.strip().upper()
        if exp_arg in {"A", "B", "C", "D", "E", "F", "G"}:
            matching = [c for c in configs if c.name.startswith(f"{exp_arg}_")]
        else:
            matching = [c for c in configs if c.name.upper() == exp_arg]
        if not matching:
            logger.error(f"Unknown experiment: {args.experiment}")
            return
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        for exp in matching:
            await run_single_experiment(exp, args.benchmark, args.output_dir, args.max_samples)
    else:
        await run_all_experiments(args.benchmark, args.output_dir, args.max_samples)


if __name__ == "__main__":
    asyncio.run(main())
