import asyncio
import json
import logging
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

from config import BenchmarkConfig, ModelConfig, DEFAULT_MODELS
from agents.base import LLMAgent
from data.error_injector import ErrorInjector

logger = logging.getLogger(__name__)

MATH_TEXT_GENERATION_PROMPT = (
    "Generate {n} distinct, self-contained mathematical statements in English with LaTeX notation. "
    "Topic: {topic}.\n\n"
    "Requirements:\n"
    "- Each statement should be 1-3 sentences long\n"
    "- Use inline LaTeX (e.g., $x^{{2}}$, $\\forall x \\in \\mathbb{{R}}$)\n"
    "- Include a mix of: definitions, theorems, lemmas, propositions, corollaries\n"
    "- Cover different difficulty levels (undergraduate to graduate)\n"
    "- Statements must be mathematically CORRECT\n"
    "- Use proper mathematical terminology\n\n"
    "Return ONLY a JSON list of strings, each being one statement. No other text.\n"
    "Example format:\n"
    '["For all $x \\\\in \\\\mathbb{{R}}$, the function $f(x) = x^{{2}}$ is continuous.", '
    '"A matrix $A \\\\in \\\\mathbb{{R}}^{{n \\\\times n}}$ is symmetric if $A = A^T$."]\n\n'
    "Generate exactly {n} statements."
)


class BenchmarkBuilder:
    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        model_config: Optional[ModelConfig] = None,
    ):
        self.config = config or BenchmarkConfig()
        self.model_config = model_config or DEFAULT_MODELS["data_generator"]
        self.injector = ErrorInjector(seed=self.config.seed)
        self.agent = LLMAgent(
            config=self.model_config,
            role="data_generator",
            system_prompt="You are a mathematics professor generating precise mathematical statements.",
        )

    async def build(self, output_path: str = "data/mathtec_bench.json") -> List[Dict]:
        logger.info("=== Building MathTEC-Bench ===")

        source_texts = await self._generate_source_texts()
        total_source_texts = sum(len(texts) for texts in source_texts.values())
        logger.info(f"Generated {total_source_texts} source texts")

        benchmark = self._inject_errors(source_texts)
        logger.info(f"Created {len(benchmark)} benchmark samples")

        stats = self._compute_stats(benchmark)
        output = {
            "metadata": {
                "name": "MathTEC-Bench",
                "version": "1.0",
                "num_samples": len(benchmark),
                "config": {
                    "seed": self.config.seed,
                    "error_distribution": self.config.error_distribution,
                    "errors_per_sentence": list(self.config.errors_per_sentence),
                    "clean_sample_ratio": self.config.clean_sample_ratio,
                    "source_texts_per_topic": self.config.source_texts_per_topic,
                },
                "statistics": stats,
            },
            "data": benchmark,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Benchmark saved to {output_path}")
        self._print_stats(stats)
        return benchmark

    async def _generate_source_texts(self) -> Dict[str, List[str]]:
        all_texts: Dict[str, List[str]] = {}
        results = await asyncio.gather(
            *(self._generate_for_topic(topic) for topic in self.config.math_topics),
            return_exceptions=True,
        )

        for topic, result in zip(self.config.math_topics, results):
            if isinstance(result, BaseException):
                logger.warning("Topic '%s' failed; skipping this topic: %s", topic, result)
                all_texts[topic] = []
                continue

            all_texts[topic] = result

        return all_texts

    async def _generate_for_topic(self, topic: str) -> List[str]:
        target = self.config.source_texts_per_topic
        batch_target = max(1, math.ceil(target * 0.6))
        seen = set()
        collected: List[str] = []
        batch_errors: List[str] = []

        for batch_idx in range(2):
            prompt = MATH_TEXT_GENERATION_PROMPT.format(n=batch_target, topic=topic)
            try:
                response = await self.agent.call_with_retry(prompt, max_retries=3)
            except Exception as e:
                batch_errors.append(str(e))
                logger.warning(
                    "Topic '%s': batch %d/2 failed: %s",
                    topic, batch_idx + 1, e,
                )
                continue

            parsed = self._parse_generated_texts(response, topic)
            for text in parsed:
                norm = re.sub(r"\s+", " ", text.strip())
                if norm in seen:
                    continue
                seen.add(norm)
                collected.append(text.strip())

        if not collected and batch_errors:
            raise RuntimeError(
                f"Topic '{topic}' failed for all batches. Last error: {batch_errors[-1]}"
            )

        if len(collected) < target:
            logger.warning(
                "Topic '%s' generated only %d unique statements (< %d target).",
                topic, len(collected), target,
            )
        return collected

    def _parse_generated_texts(self, response: str, topic: str) -> List[str]:
        try:
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                texts = json.loads(json_match.group())
                return [t for t in texts if isinstance(t, str) and len(t.strip()) > 20]
        except json.JSONDecodeError:
            pass

        logger.warning(
            "Could not parse JSON for topic '%s', using line splitting",
            topic,
        )
        return [l.strip() for l in response.split("\n") if l.strip() and len(l.strip()) > 20]

    def _make_clean_sample(self, text: str) -> Dict:
        return {
            "id": 0,
            "original": text,
            "corrupted": text,
            "errors": [],
            "num_errors": 0,
            "error_types": [],
        }

    def _inject_errors(self, source_texts: Dict[str, List[str]]) -> List[Dict]:
        benchmark = []
        rng = random.Random(self.config.seed)
        min_errors, max_errors = self.config.errors_per_sentence
        clean_ratio = min(max(self.config.clean_sample_ratio, 0.0), 1.0)
        target = self.config.source_texts_per_topic
        topic_counts: Dict[str, int] = {}
        topic_clean_counts: Dict[str, int] = {}

        for topic in self.config.math_topics:
            topic_texts = source_texts.get(topic, [])
            topic_benchmark: List[Dict] = []
            failed_clean_candidates: List[Dict] = []
            target_clean_count = int(round(min(len(topic_texts), target) * clean_ratio))
            corrupted_target = max(target - target_clean_count, 0)

            for text in topic_texts:
                num_errors = rng.randint(min_errors, max_errors)
                corrupted, errors = self.injector.inject_errors(
                    text,
                    num_errors=num_errors,
                    error_weights=self.config.error_distribution,
                )

                if corrupted != text and errors:
                    topic_benchmark.append({
                        "id": 0,
                        "original": text,
                        "corrupted": corrupted,
                        "errors": [
                            {
                                "type": e.error_type,
                                "original": e.original_span,
                                "corrupted": e.corrupted_span,
                                "start": e.char_start,
                                "end": e.char_end,
                            }
                            for e in errors
                        ],
                        "num_errors": len(errors),
                        "error_types": list(set(e.error_type for e in errors)),
                    })
                else:
                    failed_clean_candidates.append(self._make_clean_sample(text))

            success_count = len(topic_benchmark)
            if success_count >= target:
                selected_successes = topic_benchmark[:target]
                revert_count = min(target_clean_count, len(selected_successes))
                revert_indices = (
                    set(rng.sample(range(len(selected_successes)), revert_count))
                    if revert_count
                    else set()
                )
                kept = [
                    self._make_clean_sample(sample["original"])
                    if idx in revert_indices
                    else sample
                    for idx, sample in enumerate(selected_successes)
                ]
            elif success_count > corrupted_target:
                revert_count = success_count - corrupted_target
                revert_indices = (
                    set(rng.sample(range(success_count), revert_count))
                    if revert_count
                    else set()
                )
                kept = [
                    self._make_clean_sample(sample["original"])
                    if idx in revert_indices
                    else sample
                    for idx, sample in enumerate(topic_benchmark)
                ]
                needed = target - len(kept)
                kept.extend(failed_clean_candidates[:needed])
            else:
                kept = list(topic_benchmark)
                needed = target - len(kept)
                kept.extend(failed_clean_candidates[:needed])

            topic_counts[topic] = len(kept)
            topic_clean_counts[topic] = sum(1 for sample in kept if sample["num_errors"] == 0)
            benchmark.extend(kept)

        for idx, sample in enumerate(benchmark):
            sample["id"] = idx

        overclean = [
            f"{topic}:{clean_count}/{max(topic_counts[topic], 1)}"
            for topic, clean_count in topic_clean_counts.items()
            if clean_count / max(topic_counts[topic], 1) > clean_ratio
        ]
        if overclean:
            logger.warning(
                "Topics exceeding clean ratio target: %s",
                ", ".join(overclean),
            )

        underfilled = [
            f"{topic}:{count}/{target}"
            for topic, count in topic_counts.items()
            if count < target
        ]
        if underfilled:
            logger.warning(
                "Topics underfilled: %s",
                ", ".join(underfilled),
            )

        if len(benchmark) < self.config.num_samples:
            logger.warning(
                "Benchmark underfilled: %d/%d samples.",
                len(benchmark), self.config.num_samples,
            )

        return benchmark[:self.config.num_samples]

    def _compute_stats(self, benchmark: List[Dict]) -> Dict:
        type_counts: Dict[str, int] = {}
        total_errors = 0
        for sample in benchmark:
            for err in sample["errors"]:
                etype = err["type"]
                type_counts[etype] = type_counts.get(etype, 0) + 1
                total_errors += 1

        return {
            "total_samples": len(benchmark),
            "total_errors": total_errors,
            "avg_errors_per_sample": round(total_errors / max(len(benchmark), 1), 2),
            "error_type_distribution": type_counts,
            "topics_covered": len(self.config.math_topics),
        }

    def _print_stats(self, stats: Dict):
        logger.info(f"  Total samples: {stats['total_samples']}")
        logger.info(f"  Total errors:  {stats['total_errors']}")
        logger.info(f"  Avg errors/sample: {stats['avg_errors_per_sample']}")
        for etype, count in stats["error_type_distribution"].items():
            logger.info(f"  {etype}: {count}")


async def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    builder = BenchmarkBuilder()
    await builder.build("data/mathtec_bench.json")


if __name__ == "__main__":
    asyncio.run(main())
