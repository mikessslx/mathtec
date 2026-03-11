import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from agents.base import LLMAgent
from agents.detector import DetectedError
from config import DiffAuditConfig, ModelConfig

logger = logging.getLogger(__name__)


DIFF_AUDIT_SYSTEM_PROMPT = (
    "You audit OCR correction diffs for mathematical text. "
    "Approve only if the edits are minimal, justified by OCR cleanup, and do not "
    "rephrase or restyle already-correct math. Always return valid JSON."
)


DIFF_AUDIT_PROMPT = (
    "Audit whether the candidate correction made only reasonable OCR fixes.\n\n"
    "=== CORRUPTED INPUT ===\n{text}\n\n"
    "=== CANDIDATE OUTPUT ===\n{candidate}\n\n"
    "=== DETECTED ERRORS ===\n{errors}\n\n"
    "=== RULE-BASED DIFF SUMMARY ===\n{summary}\n\n"
    "Approve the candidate only if the edits are narrowly targeted, mathematically "
    "plausible, and stylistically conservative. Penalize unnecessary notation changes, "
    "structural rewrites, or edits far outside the detected spans.\n\n"
    "Return ONLY JSON in this schema:\n"
    "{{\n"
    "  \"approve\": true,\n"
    "  \"confidence\": 0.0,\n"
    "  \"issues\": [\"...\"],\n"
    "  \"reasoning\": \"short explanation\"\n"
    "}}"
)


SUSPICIOUS_INSERTIONS = (
    "ProductSet(",
    "Interval(",
    "FiniteSet(",
    "Matrix(",
    "Eq(",
    "StrictLessThan(",
    "StrictGreaterThan(",
)


@dataclass
class DiffChange:
    tag: str
    original_text: str
    corrected_text: str
    original_range: Tuple[int, int]
    corrected_range: Tuple[int, int]


@dataclass
class DiffAuditResult:
    approve: bool
    issues: List[str] = field(default_factory=list)
    confidence: float = 0.0
    changed_char_ratio: float = 0.0
    extra_edit_ratio: float = 0.0
    edit_block_count: int = 0
    hard_reject: bool = False
    reviewer_reasoning: str = ""
    raw_response: str = ""


class DiffAuditor:
    def __init__(
        self,
        model_config: ModelConfig,
        audit_config: DiffAuditConfig,
    ):
        self.config = audit_config
        self.reviewer = None
        if self.config.use_llm_review:
            self.reviewer = LLMAgent(
                config=model_config,
                role="diff_auditor",
                system_prompt=DIFF_AUDIT_SYSTEM_PROMPT,
            )

    async def audit(
        self,
        text: str,
        candidate: str,
        detected_errors: List[DetectedError],
    ) -> DiffAuditResult:
        changes = self._extract_changes(text, candidate)
        changed_chars = sum(
            max(
                change.original_range[1] - change.original_range[0],
                change.corrected_range[1] - change.corrected_range[0],
            )
            for change in changes
        )
        base_len = max(len(text), len(candidate), 1)
        changed_char_ratio = changed_chars / base_len
        extra_edit_ratio = self._compute_extra_edit_ratio(text, changes, detected_errors)

        issues: List[str] = []
        hard_reject = False
        if changed_char_ratio > self.config.max_changed_char_ratio:
            hard_reject = True
            issues.append(
                "Changed-text ratio is too high "
                f"({changed_char_ratio:.2%} > {self.config.max_changed_char_ratio:.2%})"
            )

        if len(changes) > self.config.max_edit_blocks:
            hard_reject = True
            issues.append(
                "Too many disjoint edit blocks "
                f"({len(changes)} > {self.config.max_edit_blocks})"
            )

        if detected_errors and extra_edit_ratio > self.config.max_extra_span_ratio:
            hard_reject = True
            issues.append(
                "Too many edits fall outside detected spans "
                f"({extra_edit_ratio:.2%} > {self.config.max_extra_span_ratio:.2%})"
            )

        suspicious = [token for token in SUSPICIOUS_INSERTIONS if token in candidate]
        if suspicious:
            issues.append(
                "Introduced CAS-style notation that is unusual for OCR cleanup: "
                + ", ".join(suspicious)
            )

        delimiter_delta = abs(text.count("$") - candidate.count("$"))
        if delimiter_delta > 0:
            issues.append("Changed the number of math delimiters '$'")

        result = DiffAuditResult(
            approve=not hard_reject,
            issues=issues,
            confidence=0.0 if hard_reject else 0.5,
            changed_char_ratio=changed_char_ratio,
            extra_edit_ratio=extra_edit_ratio,
            edit_block_count=len(changes),
            hard_reject=hard_reject,
        )

        if hard_reject or not self.reviewer:
            return result

        summary = self._format_summary(changes, result)
        prompt = DIFF_AUDIT_PROMPT.format(
            text=text,
            candidate=candidate,
            errors=self._format_errors(detected_errors),
            summary=summary,
        )

        try:
            response = await self.reviewer.call_with_retry(
                prompt,
                temperature=self.config.reviewer_temperature,
            )
            parsed = self._parse_json_response(response)
            result.raw_response = response
            if parsed is None:
                result.issues.append("Diff auditor returned invalid JSON; keeping rule-only decision")
                return result

            result.approve = bool(parsed.get("approve", result.approve))
            result.confidence = float(parsed.get("confidence", result.confidence or 0.5))
            result.reviewer_reasoning = str(parsed.get("reasoning", "")).strip()
            for issue in parsed.get("issues", []):
                issue_text = str(issue).strip()
                if issue_text and issue_text not in result.issues:
                    result.issues.append(issue_text)

            result.approve = result.approve and not result.hard_reject
            return result
        except Exception as exc:
            logger.warning("Diff audit reviewer failed: %s", exc)
            result.issues.append(f"Diff auditor failed: {exc}")
            return result

    def _extract_changes(self, source: str, corrected: str) -> List[DiffChange]:
        matcher = difflib.SequenceMatcher(a=source, b=corrected)
        changes: List[DiffChange] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            changes.append(DiffChange(
                tag=tag,
                original_text=source[i1:i2],
                corrected_text=corrected[j1:j2],
                original_range=(i1, i2),
                corrected_range=(j1, j2),
            ))
        return changes

    def _compute_extra_edit_ratio(
        self,
        text: str,
        changes: List[DiffChange],
        detected_errors: List[DetectedError],
    ) -> float:
        if not changes or not detected_errors:
            return 0.0

        span_ranges: List[Tuple[int, int]] = []
        for error in detected_errors:
            span_ranges.extend(self._find_span_ranges(text, error.span))

        if not span_ranges:
            return 0.0

        total = 0
        outside = 0
        for change in changes:
            if change.tag == "insert":
                segment_len = max(change.corrected_range[1] - change.corrected_range[0], 1)
                start = change.original_range[0]
                end = start + segment_len
            else:
                start, end = change.original_range
                segment_len = max(end - start, 1)

            total += segment_len
            overlap = any(self._ranges_overlap((start, end), span_range) for span_range in span_ranges)
            if not overlap:
                outside += segment_len

        return outside / max(total, 1)

    def _find_span_ranges(self, text: str, span: str) -> List[Tuple[int, int]]:
        if not span:
            return []
        ranges: List[Tuple[int, int]] = []
        start = 0
        while True:
            idx = text.find(span, start)
            if idx == -1:
                break
            ranges.append((idx, idx + len(span)))
            start = idx + max(len(span), 1)
        return ranges

    def _ranges_overlap(self, left: Tuple[int, int], right: Tuple[int, int]) -> bool:
        return left[0] < right[1] and right[0] < left[1]

    def _format_summary(
        self,
        changes: List[DiffChange],
        result: DiffAuditResult,
    ) -> str:
        lines = [
            f"changed_char_ratio: {result.changed_char_ratio:.2%}",
            f"extra_edit_ratio: {result.extra_edit_ratio:.2%}",
            f"edit_block_count: {result.edit_block_count}",
            f"rule_issues: {result.issues if result.issues else '[]'}",
            "edit_blocks:",
        ]
        if not changes:
            lines.append("- no changes")
            return "\n".join(lines)

        for idx, change in enumerate(changes, 1):
            lines.append(
                f"- {idx}. {change.tag} | "
                f"orig='{self._clip(change.original_text)}' -> "
                f"new='{self._clip(change.corrected_text)}'"
            )
        return "\n".join(lines)

    def _format_errors(self, detected_errors: List[DetectedError]) -> str:
        if not detected_errors:
            return "No detector-proposed errors"
        return "\n".join(
            f"- '{error.span}' -> '{error.suggestion}' (votes={error.vote_count}, conf={error.confidence:.2f})"
            for error in detected_errors
        )

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def _clip(self, text: str) -> str:
        clean = text.replace("\n", "\\n")
        if len(clean) <= self.config.max_snippet_chars:
            return clean
        return clean[: self.config.max_snippet_chars - 3] + "..."
