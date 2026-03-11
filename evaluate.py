import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SampleMetrics:
    sample_id: int
    sentence_correct: bool
    char_precision: float
    char_recall: float
    char_f1: float
    error_types: List[str]
    errors_found: int
    errors_total: int


@dataclass
class AggregateMetrics:
    sentence_accuracy: float
    char_precision: float
    char_recall: float
    char_f1: float
    total_samples: int
    by_error_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_num_errors: Dict[int, Dict[str, float]] = field(default_factory=dict)


def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip())


def compute_char_level_metrics(
    original: str, corrupted: str, predicted: str
) -> Tuple[float, float, float]:
    """
    Character-level precision/recall/F1.
    A "correction" is a character position where corrupted != original.
    We check if predicted matches original at those positions.
    """
    orig = normalize_whitespace(original)
    corr = normalize_whitespace(corrupted)
    pred = normalize_whitespace(predicted)

    max_len = max(len(orig), len(corr), len(pred))
    orig_padded = orig.ljust(max_len)
    corr_padded = corr.ljust(max_len)
    pred_padded = pred.ljust(max_len)

    error_positions = set()
    for i in range(min(len(orig_padded), len(corr_padded))):
        if i < len(orig) and i < len(corr) and orig[i] != corr[i]:
            error_positions.add(i)

    if not error_positions:
        is_exact = (pred == orig)
        return (1.0, 1.0, 1.0) if is_exact else (0.0, 0.0, 0.0)

    corrections_attempted = set()
    for i in range(min(len(pred_padded), len(corr_padded))):
        if i < len(pred) and i < len(corr) and pred[i] != corr[i]:
            corrections_attempted.add(i)

    true_positives = error_positions & corrections_attempted
    correct_corrections = set()
    for i in true_positives:
        if i < len(pred) and i < len(orig) and pred[i] == orig[i]:
            correct_corrections.add(i)

    precision = len(correct_corrections) / max(len(corrections_attempted), 1)
    recall = len(correct_corrections) / max(len(error_positions), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return precision, recall, f1


def compute_sentence_accuracy(original: str, predicted: str) -> bool:
    return normalize_whitespace(original) == normalize_whitespace(predicted)


def compute_gleu(
    source: str, reference: str, hypothesis: str, max_n: int = 4
) -> float:
    """
    GLEU score (from GEC literature).
    Computes n-gram overlap penalizing both missed corrections and overcorrections.
    """

    def get_ngrams(text: str, n: int) -> Dict[str, int]:
        tokens = text.split()
        ngrams: Dict[str, int] = {}
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i:i+n])
            ngrams[gram] = ngrams.get(gram, 0) + 1
        return ngrams

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = get_ngrams(reference, n)
        hyp_ngrams = get_ngrams(hypothesis, n)
        src_ngrams = get_ngrams(source, n)

        overlap = 0
        total = 0
        for gram, count in hyp_ngrams.items():
            ref_count = ref_ngrams.get(gram, 0)
            src_count = src_ngrams.get(gram, 0)
            clipped = min(count, max(ref_count - src_count, 0) + src_count)
            overlap += min(count, ref_count)
            total += count

        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(overlap / total)

    if not precisions or all(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(np.log(max(p, 1e-10)) for p in precisions) / len(precisions)
    return float(np.exp(log_avg))


def evaluate_benchmark(
    benchmark_path: str,
    results_path: str,
    output_path: Optional[str] = None,
) -> AggregateMetrics:
    with open(benchmark_path) as f:
        benchmark_data = json.load(f)
    benchmark = benchmark_data["data"] if "data" in benchmark_data else benchmark_data

    with open(results_path) as f:
        results = json.load(f)

    sample_metrics: List[SampleMetrics] = []

    for i, (sample, result) in enumerate(zip(benchmark, results)):
        original = sample["original"]
        corrupted = sample["corrupted"]
        predicted = result.get("corrected", result.get("corrected_text", ""))
        error_types = sample.get("error_types", [])

        sent_correct = compute_sentence_accuracy(original, predicted)
        precision, recall, f1 = compute_char_level_metrics(original, corrupted, predicted)

        sm = SampleMetrics(
            sample_id=i,
            sentence_correct=sent_correct,
            char_precision=precision,
            char_recall=recall,
            char_f1=f1,
            error_types=error_types,
            errors_found=sum(1 for _ in []),
            errors_total=sample.get("num_errors", 0),
        )
        sample_metrics.append(sm)

    agg = _aggregate_metrics(sample_metrics)

    report = _format_report(agg)
    print(report)

    if output_path:
        output = {
            "aggregate": {
                "sentence_accuracy": agg.sentence_accuracy,
                "char_precision": agg.char_precision,
                "char_recall": agg.char_recall,
                "char_f1": agg.char_f1,
                "total_samples": agg.total_samples,
            },
            "by_error_type": agg.by_error_type,
            "by_num_errors": agg.by_num_errors,
            "per_sample": [
                {
                    "id": sm.sample_id,
                    "sent_correct": sm.sentence_correct,
                    "precision": sm.char_precision,
                    "recall": sm.char_recall,
                    "f1": sm.char_f1,
                    "error_types": sm.error_types,
                }
                for sm in sample_metrics
            ],
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

    return agg


def _aggregate_metrics(samples: List[SampleMetrics]) -> AggregateMetrics:
    n = len(samples)
    if n == 0:
        return AggregateMetrics(0, 0, 0, 0, 0)

    sent_acc = sum(1 for s in samples if s.sentence_correct) / n
    avg_p = sum(s.char_precision for s in samples) / n
    avg_r = sum(s.char_recall for s in samples) / n
    avg_f1 = sum(s.char_f1 for s in samples) / n

    by_type: Dict[str, List[SampleMetrics]] = defaultdict(list)
    for s in samples:
        for etype in s.error_types:
            by_type[etype].append(s)

    by_type_agg = {}
    for etype, type_samples in by_type.items():
        tn = len(type_samples)
        by_type_agg[etype] = {
            "count": tn,
            "sentence_accuracy": sum(1 for s in type_samples if s.sentence_correct) / tn,
            "char_f1": sum(s.char_f1 for s in type_samples) / tn,
        }

    by_num: Dict[int, List[SampleMetrics]] = defaultdict(list)
    for s in samples:
        by_num[s.errors_total].append(s)

    by_num_agg = {}
    for num, num_samples in sorted(by_num.items()):
        nn = len(num_samples)
        by_num_agg[num] = {
            "count": nn,
            "sentence_accuracy": sum(1 for s in num_samples if s.sentence_correct) / nn,
            "char_f1": sum(s.char_f1 for s in num_samples) / nn,
        }

    return AggregateMetrics(
        sentence_accuracy=sent_acc,
        char_precision=avg_p,
        char_recall=avg_r,
        char_f1=avg_f1,
        total_samples=n,
        by_error_type=by_type_agg,
        by_num_errors=by_num_agg,
    )


def _format_report(agg: AggregateMetrics) -> str:
    lines = [
        "=" * 60,
        "  MathTEC Evaluation Report",
        "=" * 60,
        f"  Total samples:       {agg.total_samples}",
        f"  Sentence accuracy:   {agg.sentence_accuracy:.4f}",
        f"  Char precision:      {agg.char_precision:.4f}",
        f"  Char recall:         {agg.char_recall:.4f}",
        f"  Char F1:             {agg.char_f1:.4f}",
        "",
        "  By Error Type:",
    ]
    for etype, stats in agg.by_error_type.items():
        lines.append(
            f"    {etype:20s}  n={stats['count']:4d}  "
            f"sent_acc={stats['sentence_accuracy']:.3f}  "
            f"f1={stats['char_f1']:.3f}"
        )
    lines.append("")
    lines.append("  By Number of Errors:")
    for num, stats in agg.by_num_errors.items():
        lines.append(
            f"    {num} error(s):  n={stats['count']:4d}  "
            f"sent_acc={stats['sentence_accuracy']:.3f}  "
            f"f1={stats['char_f1']:.3f}"
        )
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <benchmark.json> <results.json> [output.json]")
        sys.exit(1)
    output = sys.argv[3] if len(sys.argv) > 3 else None
    evaluate_benchmark(sys.argv[1], sys.argv[2], output)
