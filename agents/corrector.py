import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List

from .base import LLMAgent
from .detector import DetectedError
from config import ModelConfig, DebateConfig

logger = logging.getLogger(__name__)


CORRECTION_SYSTEM_PROMPT = (
    "You are an expert in mathematical text error correction for mathematical text containing LaTeX. "
    "Your task is to correct errors in OCR-scanned mathematical documents. "
    "The input may already be correct, so avoid unnecessary edits. "
    "Common errors include: misspelled mathematical terms, corrupted LaTeX commands, "
    "misrecognized mathematical symbols, handwriting glyph confusions (e.g., 1/l/I), "
    "and inconsistent variable usage. "
    "Make MINIMAL corrections — only fix actual errors, do not rephrase or restructure."
)

INITIAL_CORRECTION_PROMPT = (
    "The following mathematical text was produced by OCR and may contain errors. "
    "Detected errors are listed below.\n\n"
    "=== ORIGINAL TEXT ===\n{text}\n\n"
    "=== DETECTED ERRORS ===\n{errors}\n\n"
    "Please provide the corrected version of the text. "
    "Fix ONLY the identified errors and any other obvious errors you notice. "
    "If the text is already correct, return it unchanged. "
    "Do NOT rephrase, restructure, or add content. Make minimal edits.\n\n"
    "Return your response in this format:\n"
    "REASONING: <brief explanation of each correction>\n"
    "CORRECTED: <the full corrected text>"
)

DEBATE_PROMPT = (
    "You are reviewing corrections of an OCR-scanned mathematical text. "
    "The original text may already be correct. "
    "Below is the original text, the detected errors, and corrections proposed by other agents.\n\n"
    "=== ORIGINAL TEXT ===\n{text}\n\n"
    "=== DETECTED ERRORS ===\n{errors}\n\n"
    "=== OTHER AGENTS' CORRECTIONS ===\n{other_corrections}\n\n"
    "=== YOUR PREVIOUS CORRECTION ===\n{my_correction}\n\n"
    "Review the other agents' corrections. Consider:\n"
    "1. Did they catch errors you missed?\n"
    "2. Did they over-correct (change things that weren't errors)?\n"
    "3. Are their LaTeX corrections syntactically valid?\n"
    "4. Do you agree or disagree with specific changes?\n\n"
    "Provide your revised correction (or keep yours if you believe it's correct).\n\n"
    "REASONING: <explain agreements/disagreements with other agents>\n"
    "CORRECTED: <the full corrected text>"
)

@dataclass
class CorrectionResult:
    corrected_text: str
    reasoning: str
    agent_id: str
    round_num: int


@dataclass
class DebateResult:
    final_correction: str
    all_rounds: List[List[CorrectionResult]] = field(default_factory=list)
    judge_reasoning: str = ""


class MultiAgentCorrector:
    def __init__(
        self,
        agent_configs: Dict[str, ModelConfig],
        debate_config: DebateConfig,
    ):
        self.config = debate_config
        self.agents: List[LLMAgent] = []

        max_agents = max(1, self.config.num_agents)
        selected_agents = list(agent_configs.items())[:max_agents]
        if not selected_agents:
            raise ValueError("No correction agents configured")
        if len(selected_agents) < max_agents:
            logger.warning(
                "DebateConfig.num_agents=%s but only %s agent configs available",
                max_agents,
                len(selected_agents),
            )

        for agent_id, mcfg in selected_agents:
            self.agents.append(
                LLMAgent(
                    config=mcfg,
                    role=agent_id,
                    system_prompt=CORRECTION_SYSTEM_PROMPT,
                )
            )

    async def correct(
        self,
        text: str,
        detected_errors: List[DetectedError],
    ) -> DebateResult:
        error_desc = self._format_errors(detected_errors)
        all_rounds: List[List[CorrectionResult]] = []

        round0_tasks = [
            self._get_initial_correction(agent, text, error_desc)
            for agent in self.agents
        ]
        round0_raw = await asyncio.gather(*round0_tasks, return_exceptions=True)
        round0_results: List[CorrectionResult] = []
        for r in round0_raw:
            if isinstance(r, BaseException):
                logger.warning(f"Agent failed in round 0: {r}")
            else:
                round0_results.append(r)
        all_rounds.append(round0_results)

        current_corrections: Dict[str, CorrectionResult] = {
            r.agent_id: r for r in round0_results
        }

        for debate_round in range(1, self.config.num_rounds + 1):
            round_tasks = [
                self._get_debate_correction(
                    agent, text, error_desc, current_corrections, debate_round
                )
                for agent in self.agents
                if agent.role in current_corrections
            ]
            round_raw = await asyncio.gather(*round_tasks, return_exceptions=True)
            round_results: List[CorrectionResult] = []
            for r in round_raw:
                if isinstance(r, BaseException):
                    logger.warning(f"Agent failed in round {debate_round}: {r}")
                else:
                    round_results.append(r)
            all_rounds.append(round_results)

            for r in round_results:
                current_corrections[r.agent_id] = r

        return DebateResult(
            final_correction=self._select_best(list(current_corrections.values())),
            all_rounds=all_rounds,
        )

    async def _get_initial_correction(
        self, agent: LLMAgent, text: str, error_desc: str
    ) -> CorrectionResult:
        prompt = INITIAL_CORRECTION_PROMPT.format(text=text, errors=error_desc)
        response = await agent.call_with_retry(prompt)
        corrected = self._extract_corrected_text(response)
        reasoning = self._extract_reasoning(response)
        return CorrectionResult(
            corrected_text=corrected,
            reasoning=reasoning,
            agent_id=agent.role,
            round_num=0,
        )

    async def _get_debate_correction(
        self,
        agent: LLMAgent,
        text: str,
        error_desc: str,
        current_corrections: Dict[str, CorrectionResult],
        round_num: int,
    ) -> CorrectionResult:
        other_corrections = "\n\n".join(
            f"--- Agent {r.agent_id} (Round {r.round_num}) ---\n"
            f"Reasoning: {r.reasoning}\n"
            f"Correction: {r.corrected_text}"
            for aid, r in current_corrections.items()
            if aid != agent.role
        )
        my_prev = current_corrections.get(agent.role)
        my_correction = my_prev.corrected_text if my_prev else ""

        prompt = DEBATE_PROMPT.format(
            text=text,
            errors=error_desc,
            other_corrections=other_corrections,
            my_correction=my_correction,
        )
        response = await agent.call_with_retry(prompt)
        corrected = self._extract_corrected_text(response)
        reasoning = self._extract_reasoning(response)
        return CorrectionResult(
            corrected_text=corrected,
            reasoning=reasoning,
            agent_id=agent.role,
            round_num=round_num,
        )

    def _format_errors(self, errors: List[DetectedError]) -> str:
        if not errors:
            return (
                "No specific errors detected. "
                "The text may already be correct; keep it unchanged unless you find a clear error."
            )
        lines = []
        for i, e in enumerate(errors, 1):
            lines.append(
                f"{i}. [{e.detector_id}] '{e.span}' → suggested: '{e.suggestion}' "
                f"(confidence: {e.confidence:.2f}, votes: {e.vote_count})"
            )
        return "\n".join(lines)

    def _select_best(self, corrections: List[CorrectionResult]) -> str:
        if not corrections:
            return ""
        if len(corrections) == 1:
            return corrections[0].corrected_text
        return corrections[0].corrected_text

    def _extract_corrected_text(self, response: str) -> str:
        match = re.search(r'CORRECTED:\s*(.+)', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()

    def _extract_reasoning(self, response: str) -> str:
        match = re.search(r'REASONING:\s*(.+?)(?=CORRECTED:|$)', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
