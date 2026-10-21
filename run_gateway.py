"""
Quick-start script for gateway users.

Configure the endpoint through environment variables (never hard-code a key):

    export GATEWAY_BASE_URL="https://your-endpoint.example.com/v1"
    export GATEWAY_API_KEY="..."
    export GATEWAY_MODEL="gpt-5.4"          # optional, defaults below

Then run: python run_gateway.py <command>
"""
import argparse
import asyncio
import logging
import os
from pathlib import Path


# Read gateway settings from the environment; the defaults match the setup used
# for the reported experiments but are safe to override.
GATEWAY_BASE_URL = os.environ.get("GATEWAY_BASE_URL", "http://127.0.0.1:8317/v1")
GATEWAY_API_KEY = os.environ.get("GATEWAY_API_KEY", "your-api-key-1")
_DEFAULT_MODEL = os.environ.get("GATEWAY_MODEL", "gpt-5.4")

MODELS = {
    "detector":       _DEFAULT_MODEL,
    "agent_a":        _DEFAULT_MODEL,
    "agent_b":        _DEFAULT_MODEL,
    "agent_c":        _DEFAULT_MODEL,
    "judge":          _DEFAULT_MODEL,
    "data_generator": _DEFAULT_MODEL,
}


if GATEWAY_API_KEY:
    os.environ["GATEWAY_API_KEY"] = GATEWAY_API_KEY

from config import (
    DiffAuditConfig,
    PipelineConfig,
    RenderCheckConfig,
    make_gateway_models,
)
from pipeline import MathTECPipeline

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure app logs and suppress noisy third-party HTTP request logs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    # Keep project INFO logs, hide per-request transport noise.
    noisy_loggers = ("httpx", "httpcore", "openai", "anthropic")
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def build_gateway_models():
    return make_gateway_models(
        base_url=GATEWAY_BASE_URL,
        api_key_env="GATEWAY_API_KEY",
        models=MODELS,
    )


def build_pipeline(verbose: bool = True) -> MathTECPipeline:
    return MathTECPipeline(
        models=build_gateway_models(),
        pipeline_config=PipelineConfig(
            verbose=verbose,
            render_check=RenderCheckConfig(enabled=True),
            diff_audit=DiffAuditConfig(enabled=True),
        ),
    )


async def quick_test():
    """Run a quick end-to-end test on a single sample."""
    pipeline = build_pipeline()

    test_input = (
        "Let $f: \\mabb{R} \\to \\mabb{R}$ be a continous funciton. "
        "By the Intermediate Value Theroem, if $f(a) < 0$ and $f(b) > 0$, "
        "then there exsts a $c \\in (a, b)$ such that $f(c) = 0$."
    )

    print("=" * 60)
    print("INPUT (with errors):")
    print(test_input)
    print("=" * 60)

    result = await pipeline.process(test_input)

    print("\nOUTPUT (corrected):")
    print(result.corrected_text)
    print(f"\nDetected errors: {len(result.detected_errors)}")
    for e in result.detected_errors:
        print(f"  - [{e['detector_id']}] '{e['span']}' → '{e['suggestion']}'")
    print(f"Processing time: {result.processing_time:.1f}s")


async def build_benchmark():
    """Build MathTEC-Bench using the gateway."""
    from data.construct_benchmark import BenchmarkBuilder
    from config import BenchmarkConfig

    builder = BenchmarkBuilder(
        config=BenchmarkConfig(),
        model_config=build_gateway_models()["data_generator"],
    )
    await builder.build("data/mathtec_bench.json")


async def run_experiments():
    """Run all ablation experiments using an existing benchmark."""
    from run_experiments import run_all_experiments

    models = build_gateway_models()

    benchmark_path = "data/mathtec_bench.json"
    if not Path(benchmark_path).exists():
        raise FileNotFoundError(
            f"Benchmark not found: {benchmark_path}. "
            "Run `python run_gateway.py benchmark` first."
        )

    print("Step 1: Running experiments...")
    await run_all_experiments(
        benchmark_path=benchmark_path,
        output_dir="results",
        models=models,
    )


async def interactive_chat(show_details: bool = False):
    """Run an interactive correction loop using the gateway-backed pipeline."""
    pipeline = build_pipeline(verbose=False)

    print("MathTEC interactive mode")
    print("Type your text and press Enter. Type 'exit' to quit.")

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break

        try:
            result = await pipeline.process(user_text)
        except Exception as exc:
            logger.error("Interactive chat request failed: %s", exc)
            print(f"Error: {exc}")
            continue

        print(f"Model: {result.corrected_text}")
        if show_details:
            print(f"Detected errors: {len(result.detected_errors)}")
            if result.detected_errors:
                for error in result.detected_errors:
                    print(
                        "  - "
                        f"[{error['detector_id']}] "
                        f"'{error['span']}' -> '{error['suggestion']}'"
                    )
            print(f"Processing time: {result.processing_time:.1f}s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gateway entry point for MathTEC.",
        epilog="Set GATEWAY_BASE_URL and GATEWAY_API_KEY in the environment before running.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("test", help="Quick test on one sample")
    subparsers.add_parser("benchmark", help="Build MathTEC-Bench")
    subparsers.add_parser("experiments", help="Run all experiments on the benchmark")

    chat_parser = subparsers.add_parser("chat", help="Interactive correction mode")
    chat_parser.add_argument(
        "--show-details",
        action="store_true",
        help="Show detected errors and latency for each request",
    )

    return parser


if __name__ == "__main__":
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "test":
        asyncio.run(quick_test())
    elif args.command == "benchmark":
        asyncio.run(build_benchmark())
    elif args.command == "experiments":
        asyncio.run(run_experiments())
    elif args.command == "chat":
        asyncio.run(interactive_chat(show_details=args.show_details))
    else:
        parser.print_help()
