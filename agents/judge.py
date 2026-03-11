import json
import logging
import re
from typing import List

from .base import LLMAgent
from .corrector import CorrectionResult
from config import ModelConfig

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = (
    "You are an expert judge evaluating corrections of OCR-scanned mathematical text. "
    "You must select the best correction from multiple candidates based on: "
    "1) Accuracy of error fixes, 2) Minimal unnecessary changes, "
    "3) LaTeX correctness, 4) Mathematical validity."
)

JUDGE_PROMPT = (
    "Below is an original mathematical text with OCR errors, and {n} proposed corrections "
    "from different agents. Select the BEST correction.\n\n"
    "=== ORIGINAL TEXT ===\n{text}\n\n"
    "=== CORRECTIONS ===\n{corrections}\n\n"
    "Evaluate each correction on:\n"
    "1. Error coverage: Did it fix all the errors?\n"
    "2. Precision: Did it avoid unnecessary changes?\n"
    "3. LaTeX validity: Are all LaTeX expressions syntactically correct?\n"
    "4. Mathematical correctness: Are the mathematical statements valid?\n\n"
    "Respond in this format:\n"
    "ANALYSIS: <brief comparison of each correction>\n"
    "WINNER: <agent_id of the best correction>\n"
    "FINAL: <the winning corrected text, copied exactly>"
)


class DebateJudge:
    def __init__(self, model_config: ModelConfig):
        self.agent = LLMAgent(
            config=model_config,
            role="judge",
            system_prompt=JUDGE_SYSTEM_PROMPT,
        )

    async def judge(
        self, text: str, corrections: List[CorrectionResult]
    ) -> str:
        if len(corrections) == 0:
            return text
        if len(corrections) == 1:
            return corrections[0].corrected_text

        corrections_text = "\n\n".join(
            f"--- [{r.agent_id}] ---\n"
            f"Reasoning: {r.reasoning}\n"
            f"Text: {r.corrected_text}"
            for r in corrections
        )

        prompt = JUDGE_PROMPT.format(
            text=text,
            n=len(corrections),
            corrections=corrections_text,
        )
        response = await self.agent.call_with_retry(prompt)
        return self._extract_final(response, corrections)

    def _extract_final(
        self, response: str, corrections: List[CorrectionResult]
    ) -> str:
        final_match = re.search(r'FINAL:\s*(.+)', response, re.DOTALL)
        if final_match:
            return final_match.group(1).strip()

        winner_match = re.search(r'WINNER:\s*(\S+)', response)
        if winner_match:
            winner_id = winner_match.group(1).strip()
            for c in corrections:
                if c.agent_id == winner_id:
                    return c.corrected_text

        return corrections[0].corrected_text
