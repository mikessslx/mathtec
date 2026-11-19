"""Rebuild the paper's tables from the shipped result files.

This script is fully offline: it reads only `data/mathtec_bench.json` and the
`results/*_results.json` files that are already in the repository. It does not
call any language-model API and needs no credentials.

Run from the repository root:

    python reproduce_tables.py

Outputs Table 1 (benchmark statistics), Table 2 (main ablation results) and
Table 3 (character-level F1 by error type).
"""

import json
import re
from collections import Counter
from typing import Dict, List

BENCHMARK_PATH = "data/mathtec_bench.json"

# Configuration key -> result-file stem. Ordered as in the paper.
CONFIGS = [
    ("A", "A_single_prompt_baseline", "Single prompt + single agent"),
    ("B", "B_deep_detection", "DEEP detection only"),
    ("C", "C_multi_agent_debate", "Debate only"),
    ("D", "D_deep_plus_debate", "DEEP + debate"),
    ("E", "E_renderer_check", "D + MathJax gate"),
    ("F", "F_diff_check", "D + diff audit"),
    ("G", "G_diff_plus_renderer", "D + diff audit + MathJax"),
]

ERROR_TYPES = [
    "spelling",
    "ocr_symbol",
    "ocr_formula",
    "variable",
    "handwriting_confusion",
]


def load_benchmark() -> List[Dict]:
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    return payload["data"] if "data" in payload else payload


def load_results(stem: str) -> List[Dict]:
    with open(f"results/{stem}_results.json", encoding="utf-8") as f:
        return json.load(f)


def table1(bench: List[Dict]) -> None:
    """Table 1: statistics of MathTEC-Bench."""
    n = len(bench)
    clean = sum(1 for s in bench if s["num_errors"] == 0)
    total_errors = sum(len(s["errors"]) for s in bench)
    occurrences = Counter(e["type"] for s in bench for e in s["errors"])
    words = sum(len(s["original"].split()) for s in bench) / n
    # Inline math expressions delimited by single dollar signs.
    math_expr = sum(len(re.findall(r"\$[^$]*\$", s["original"])) for s in bench) / n
    error_counts = Counter(s["num_errors"] for s in bench)

    print("Table 1: statistics of MathTEC-Bench")
    print(f"  Total samples            {n}")
    print(f"  Clean samples            {clean}  ({clean / n:.1%})")
    print(f"  Injected errors          {total_errors}  ({total_errors / n:.2f} / sample)")
    print(f"  Avg. words / sample      {words:.1f}")
    print(f"  Avg. math expr. / sample {math_expr:.2f}")
    print("  Injected error occurrences:")
    for name in ERROR_TYPES:
        count = occurrences[name]
        print(f"    {name:24s} {count:4d}  ({count / total_errors:.1%})")
    print(
        "  Samples with 1/2/3 errors: "
        f"{error_counts[1]}/{error_counts[2]}/{error_counts[3]}"
    )


def _evaluate(bench: List[Dict], results: List[Dict]) -> Dict:
    """Recompute char-level P/R/F1 and sentence accuracy.

    Mirrors evaluate.compute_char_level_metrics so this script stays independent
    of the numpy dependency but agrees with the shipped evaluation numbers.
    """
    from evaluate import compute_char_level_metrics, compute_sentence_accuracy

    rows = []
    for sample, result in zip(bench, results):
        predicted = result.get("corrected", result.get("corrected_text", ""))
        precision, recall, f1 = compute_char_level_metrics(
            sample["original"], sample["corrupted"], predicted
        )
        rows.append(
            {
                "sentence_correct": compute_sentence_accuracy(
                    sample["original"], predicted
                ),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "error_types": sample.get("error_types", []),
            }
        )

    def mean(values):
        return sum(values) / max(len(values), 1)

    by_type = {}
    for etype in ERROR_TYPES:
        subset = [r for r in rows if etype in r["error_types"]]
        by_type[etype] = {
            "count": len(subset),
            "char_f1": mean([r["f1"] for r in subset]),
            "sentence_accuracy": mean([1.0 if r["sentence_correct"] else 0.0 for r in subset]),
        }

    return {
        "precision": mean([r["precision"] for r in rows]),
        "recall": mean([r["recall"] for r in rows]),
        "f1": mean([r["f1"] for r in rows]),
        "sentence_accuracy": mean([1.0 if r["sentence_correct"] else 0.0 for r in rows]),
        "by_type": by_type,
    }


def table2_and_3(bench: List[Dict]) -> None:
    """Table 2: main results. Table 3: character-level F1 by error type."""
    metrics = {}
    for key, stem, description in CONFIGS:
        metrics[key] = _evaluate(bench, load_results(stem))

    print()
    print("Table 2: results on MathTEC-Bench (percentages)")
    print(f"  {'Cfg':<4}{'Prec.':>7}{'Rec.':>7}{'F1':>7}{'Sent-Acc':>10}  Description")
    for key, _stem, description in CONFIGS:
        m = metrics[key]
        print(
            f"  {key:<4}{m['precision'] * 100:>7.1f}{m['recall'] * 100:>7.1f}"
            f"{m['f1'] * 100:>7.1f}{m['sentence_accuracy'] * 100:>10.1f}  {description}"
        )

    baseline = metrics["A"]
    best = metrics["G"]
    print(
        f"  G minus A: F1 {best['f1'] * 100 - baseline['f1'] * 100:+.1f}, "
        f"Sent-Acc {best['sentence_accuracy'] * 100 - baseline['sentence_accuracy'] * 100:+.1f}"
    )

    print()
    print("Table 3: character-level F1 by error type, configurations A / D / G (percentages)")
    # Short labels matching the paper's table header.
    labels = [
        "Spell.",
        "OCR-Sym.",
        "OCR-Form.",
        "Variable",
        "Handwrite",
    ]
    header = "".join(f"{label:>11}" for label in labels)
    print(f"  {'Cfg':<4}{header}")
    for key in ("A", "D", "G"):
        cells = "".join(
            f"{metrics[key]['by_type'][name]['char_f1'] * 100:>11.1f}"
            for name in ERROR_TYPES
        )
        print(f"  {key:<4}{cells}")


def main() -> None:
    bench = load_benchmark()
    table1(bench)
    table2_and_3(bench)


if __name__ == "__main__":
    main()
