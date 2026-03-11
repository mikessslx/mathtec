import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List

from .base import LLMAgent
from config import ModelConfig, DetectionConfig

logger = logging.getLogger(__name__)

DETECTION_PROMPTS = [
    {
        "id": "spelling",
        "instruction": (
            "You are a spelling error detector for mathematical text. "
            "Examine the following mathematical text (which may contain LaTeX) and identify "
            "any misspelled English words. Mathematical symbols and LaTeX commands are NOT spelling errors. "
            "Focus on natural language words like 'theorem', 'continuous', 'differentiable', etc. "
            "The text may already be error-free.\n\n"
            "Text: {text}\n\n"
            "Return a JSON list of detected errors. Each error should have:\n"
            '- "span": the misspelled word exactly as it appears\n'
            '- "suggestion": the correct spelling\n'
            '- "confidence": a float between 0 and 1\n\n'
            "If no spelling errors are found, return an empty list: []\n"
            "Return ONLY the JSON list, no other text."
        ),
    },
    {
        "id": "latex_syntax",
        "instruction": (
            "You are a LaTeX syntax error detector. "
            "Examine the following mathematical text and identify any LaTeX syntax errors: "
            "mismatched braces, incorrect command names, malformed expressions, etc. "
            "The text may already be error-free.\n\n"
            "Text: {text}\n\n"
            "Return a JSON list of detected errors. Each error should have:\n"
            '- "span": the erroneous LaTeX fragment\n'
            '- "suggestion": the corrected LaTeX\n'
            '- "confidence": a float between 0 and 1\n\n'
            "If no errors are found, return an empty list: []\n"
            "Return ONLY the JSON list, no other text."
        ),
    },
    {
        "id": "math_validity",
        "instruction": (
            "You are a mathematical validity checker. "
            "Examine the following mathematical text and identify any expressions that are "
            "mathematically invalid, nonsensical, or likely OCR recognition errors "
            "(e.g., ≤ misread as <, ∀ misread as A, x² misread as x2, 1/l/I confusion). "
            "The text may already be error-free.\n\n"
            "Text: {text}\n\n"
            "Return a JSON list of detected errors. Each error should have:\n"
            '- "span": the erroneous expression\n'
            '- "suggestion": the correct expression\n'
            '- "confidence": a float between 0 and 1\n\n'
            "If no errors are found, return an empty list: []\n"
            "Return ONLY the JSON list, no other text."
        ),
    },
    {
        "id": "ocr_artifacts",
        "instruction": (
            "You are an OCR error detector specialized in mathematical documents. "
            "Examine the following text which was produced by OCR and look for typical OCR errors:\n"
            "- Symbols misrecognized: ∀→A, ∈→E, ∫→f, ×→x, ≤→<\n"
            "- Superscripts/subscripts lost: x²→x2, aₙ→an\n"
            "- LaTeX commands corrupted: \\frac→\\fac, \\sqrt→\\sqr\n"
            "- Handwriting confusions: 1↔l/I, 0↔O/Q, 2↔Z, 5↔S, 8↔B\n"
            "- Mathematical operators confused\n"
            "The text may already be error-free.\n\n"
            "Text: {text}\n\n"
            "Return a JSON list of detected errors. Each error should have:\n"
            '- "span": the OCR-corrupted text\n'
            '- "suggestion": what it should be\n'
            '- "confidence": a float between 0 and 1\n\n'
            "If no errors are found, return an empty list: []\n"
            "Return ONLY the JSON list, no other text."
        ),
    },
    {
        "id": "consistency",
        "instruction": (
            "You are a consistency checker for mathematical text. "
            "Examine the following text and identify any inconsistencies:\n"
            "- Variables used differently across expressions (e.g., f(x) then f(y))\n"
            "- Notation changes mid-text (e.g., switching between \\leq and <=)\n"
            "- Referenced but undefined symbols\n"
            "- Mismatched delimiters or parentheses\n"
            "The text may already be error-free.\n\n"
            "Text: {text}\n\n"
            "Return a JSON list of detected errors. Each error should have:\n"
            '- "span": the inconsistent text\n'
            '- "suggestion": the consistent version\n'
            '- "confidence": a float between 0 and 1\n\n'
            "If no errors are found, return an empty list: []\n"
            "Return ONLY the JSON list, no other text."
        ),
    },
    {
        "id": "handwriting_confusion",
        "instruction": (
            "You are a detector for handwriting-induced character confusions in ordinary words "
            "within mathematical prose. Ignore content inside LaTeX math spans such as $...$, "
            "\\(...\\), and \\[...\\]. Focus on visually similar glyph substitutions that often "
            "appear in handwritten notes or OCR from handwriting, such as: o↔0, l↔1↔I, s↔5, "
            "u↔n, and c↔e.\n\n"
            "The text may already be error-free.\n\n"
            "Text: {text}\n\n"
            "Return a JSON list of detected errors. Each error should have:\n"
            '- "span": the suspicious handwritten-confused word exactly as it appears\n'
            '- "suggestion": the most likely intended token\n'
            '- "confidence": a float between 0 and 1\n\n'
            "Only report high-likelihood ordinary-word confusions; avoid formula fragments, "
            "isolated variables, and normal stylistic variants.\n"
            "If no errors are found, return an empty list: []\n"
            "Return ONLY the JSON list, no other text."
        ),
    },
]


@dataclass
class DetectedError:
    span: str
    suggestion: str
    confidence: float
    detector_id: str
    vote_count: int = 1


class ErrorDetector:
    def __init__(self, model_config: ModelConfig, detection_config: DetectionConfig):
        self.config = detection_config
        num_prompts = max(1, min(self.config.num_prompts, len(DETECTION_PROMPTS)))
        if num_prompts != self.config.num_prompts:
            logger.warning(
                "DetectionConfig.num_prompts=%s is out of range; using %s prompts",
                self.config.num_prompts,
                num_prompts,
            )
        self.prompts = DETECTION_PROMPTS[:num_prompts]
        self.agents = [
            LLMAgent(
                config=model_config,
                role=f"detector_{p['id']}",
                system_prompt="You are a precise error detector. Always respond with valid JSON.",
            )
            for p in self.prompts
        ]
        self.agreement_threshold = min(self.config.agreement_threshold, len(self.prompts))
        if self.agreement_threshold != self.config.agreement_threshold:
            logger.warning(
                "DetectionConfig.agreement_threshold=%s exceeds prompt count=%s; using threshold=%s",
                self.config.agreement_threshold,
                len(self.prompts),
                self.agreement_threshold,
            )

    async def detect(self, text: str) -> List[DetectedError]:
        tasks = [
            self._run_single_detector(agent, prompt, text)
            for agent, prompt in zip(self.agents, self.prompts)
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        all_errors: List[DetectedError] = []
        for i, result in enumerate(raw_results):
            if isinstance(result, BaseException):
                logger.warning(f"Detector {self.prompts[i]['id']} failed: {result}")
                continue
            all_errors.extend(result)

        merged = self._merge_and_vote(all_errors)

        return [e for e in merged if e.vote_count >= self.agreement_threshold]

    async def _run_single_detector(
        self, agent: LLMAgent, prompt_template: Dict, text: str
    ) -> List[DetectedError]:
        prompt = prompt_template["instruction"].format(text=text)
        response = await agent.call_with_retry(prompt, temperature=self.config.temperature)

        errors = self._parse_response(response, prompt_template["id"])
        return errors

    def _parse_response(self, response: str, detector_id: str) -> List[DetectedError]:
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            return []

        try:
            items = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from detector {detector_id}")
            return []

        errors = []
        for item in items:
            if isinstance(item, dict) and "span" in item:
                errors.append(DetectedError(
                    span=item.get("span", ""),
                    suggestion=item.get("suggestion", ""),
                    confidence=float(item.get("confidence", 0.5)),
                    detector_id=detector_id,
                ))
        return errors

    def _merge_and_vote(self, errors: List[DetectedError]) -> List[DetectedError]:
        """Merge similar errors and count votes (how many detectors found them)."""
        if not errors:
            return []

        merged: List[DetectedError] = []
        used = [False] * len(errors)

        for i, err_i in enumerate(errors):
            if used[i]:
                continue
            used[i] = True
            best = err_i
            vote_count = 1

            for j, err_j in enumerate(errors):
                if j <= i or used[j]:
                    continue
                if self._errors_match(err_i, err_j):
                    used[j] = True
                    vote_count += 1
                    if err_j.confidence > best.confidence:
                        best = err_j

            best.vote_count = vote_count
            merged.append(best)

        merged.sort(key=lambda e: (e.vote_count, e.confidence), reverse=True)
        return merged

    def _errors_match(self, a: DetectedError, b: DetectedError) -> bool:
        if a.span.lower() == b.span.lower():
            return True

        shorter = min(a.span, b.span, key=len)
        longer = max(a.span, b.span, key=len)
        if len(shorter) > 2 and shorter.lower() in longer.lower():
            return True

        if a.suggestion and b.suggestion:
            if a.suggestion.lower() == b.suggestion.lower():
                return True

        return False
