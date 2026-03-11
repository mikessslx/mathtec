import asyncio
import difflib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    DEFAULT_MODELS,
    ModelConfig,
    PipelineConfig,
    load_config,
)
from agents.corrector import MultiAgentCorrector
from agents.detector import ErrorDetector
from agents.judge import DebateJudge
from validators import DiffAuditor, RendererChecker

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    original_text: str
    corrected_text: str
    detected_errors: List[Dict] = field(default_factory=list)
    debate_rounds: int = 0
    validation_mode: str = ""
    candidate_count: int = 0
    judge_candidate_passed_validation: Optional[bool] = None
    selected_candidate_passed_validation: Optional[bool] = None
    selected_candidate_source: str = "judge"
    used_alternative_candidate: bool = False
    validation_fallback_used: bool = False
    diff_check_passed: Optional[bool] = None
    diff_check_issues: List[str] = field(default_factory=list)
    render_check_passed: Optional[bool] = None
    render_check_issues: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)


class MathTECPipeline:
    def __init__(
        self,
        models: Optional[Dict[str, ModelConfig]] = None,
        pipeline_config: Optional[PipelineConfig] = None,
    ):
        self.models = models or DEFAULT_MODELS
        self.config = pipeline_config or PipelineConfig()

        self.detector = ErrorDetector(
            model_config=self.models["detector"],
            detection_config=self.config.detection,
        )

        agent_configs = {}
        for key in ["agent_a", "agent_b", "agent_c"]:
            if key in self.models:
                agent_configs[key] = self.models[key]
        self.corrector = MultiAgentCorrector(
            agent_configs=agent_configs,
            debate_config=self.config.debate,
        )

        self.judge = DebateJudge(model_config=self.models["judge"])

        self.renderer_checker = None
        if self.config.render_check.enabled:
            self.renderer_checker = RendererChecker(self.config.render_check)
        self.diff_auditor = None
        if self.config.diff_audit.enabled:
            self.diff_auditor = DiffAuditor(
                model_config=self.models["judge"],
                audit_config=self.config.diff_audit,
            )

    async def process(self, text: str) -> PipelineResult:
        start_time = time.time()
        result = PipelineResult(original_text=text, corrected_text=text)

        t0 = time.time()
        detected_errors = await self.detector.detect(text)
        result.stage_times["detection"] = time.time() - t0
        result.detected_errors = [
            {"span": e.span, "suggestion": e.suggestion,
             "confidence": e.confidence, "detector_id": e.detector_id,
             "vote_count": e.vote_count}
            for e in detected_errors
        ]

        if self.config.verbose:
            logger.info(f"Stage 1: Detected {len(detected_errors)} errors")

        t0 = time.time()
        debate_result = await self.corrector.correct(text, detected_errors)
        result.stage_times["debate"] = time.time() - t0

        final_corrections = []
        if debate_result.all_rounds:
            last_round = debate_result.all_rounds[-1]
            final_corrections = last_round

        if final_corrections:
            judged_text = await self.judge.judge(text, final_corrections)
        else:
            judged_text = debate_result.final_correction

        result.corrected_text = judged_text
        result.debate_rounds = max(len(debate_result.all_rounds) - 1, 0)
        candidate_pool = self._build_candidate_pool(judged_text, debate_result)
        result.candidate_count = len(candidate_pool)
        result.validation_mode = self._get_validation_mode()

        if self.config.verbose:
            logger.info(f"Stage 2: Debate completed ({result.debate_rounds} rounds)")

        if candidate_pool and (self.diff_auditor or self.renderer_checker):
            t0 = time.time()
            result = await self._apply_post_checks(text, detected_errors, candidate_pool, result)
            result.stage_times["postcheck"] = time.time() - t0

        result.processing_time = time.time() - start_time
        return result

    async def _apply_post_checks(
        self,
        original_text: str,
        detected_errors,
        candidates: List[Dict[str, str]],
        result: PipelineResult,
    ) -> PipelineResult:
        judge_assessment = await self._assess_candidate(
            original_text,
            candidates[0],
            detected_errors,
        )
        result.judge_candidate_passed_validation = judge_assessment["passed"]

        if judge_assessment["passed"]:
            return self._apply_assessment(result, judge_assessment, used_alternative=False)

        passing_alternatives = []
        for candidate in candidates[1:]:
            assessment = await self._assess_candidate(original_text, candidate, detected_errors)
            if assessment["passed"]:
                passing_alternatives.append(assessment)

        if passing_alternatives:
            selected = min(
                passing_alternatives,
                key=lambda item: (item["change_ratio"], item["source"]),
            )
            return self._apply_assessment(result, selected, used_alternative=True)

        result.validation_fallback_used = True
        return self._apply_assessment(result, judge_assessment, used_alternative=False)

    async def _assess_candidate(
        self,
        original_text: str,
        candidate: Dict[str, str],
        detected_errors,
    ) -> Dict:
        diff_result = None
        render_result = None
        if self.diff_auditor:
            diff_result = await self.diff_auditor.audit(
                original_text,
                candidate["text"],
                detected_errors,
            )
        if self.renderer_checker:
            render_result = self.renderer_checker.check_text(candidate["text"])

        passed = True
        if diff_result is not None:
            passed = passed and diff_result.approve
        if render_result is not None:
            passed = passed and render_result.is_valid

        return {
            "text": candidate["text"],
            "source": candidate["source"],
            "passed": passed,
            "change_ratio": self._candidate_change_ratio(original_text, candidate["text"], diff_result),
            "diff_result": diff_result,
            "render_result": render_result,
        }

    def _apply_assessment(
        self,
        result: PipelineResult,
        assessment: Dict,
        used_alternative: bool,
    ) -> PipelineResult:
        result.corrected_text = assessment["text"]
        result.selected_candidate_source = assessment["source"]
        result.used_alternative_candidate = used_alternative
        result.selected_candidate_passed_validation = assessment["passed"]

        diff_result = assessment.get("diff_result")
        if diff_result is not None:
            result.diff_check_passed = diff_result.approve
            result.diff_check_issues = diff_result.issues

        render_result = assessment.get("render_result")
        if render_result is not None:
            result.render_check_passed = render_result.is_valid
            result.render_check_issues = render_result.issues

        return result

    def _build_candidate_pool(self, judged_text: str, debate_result) -> List[Dict[str, str]]:
        candidates: List[Dict[str, str]] = []
        seen = set()

        def add_candidate(text: str, source: str) -> None:
            normalized = text.strip()
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            candidates.append({"text": normalized, "source": source})

        add_candidate(judged_text, "judge")

        for round_results in reversed(debate_result.all_rounds):
            for correction in round_results:
                add_candidate(correction.corrected_text, f"{correction.agent_id}_r{correction.round_num}")

        add_candidate(debate_result.final_correction, "debate_fallback")
        return candidates

    def _candidate_change_ratio(
        self,
        original_text: str,
        candidate_text: str,
        diff_result,
    ) -> float:
        if diff_result is not None:
            return diff_result.changed_char_ratio
        return 1.0 - difflib.SequenceMatcher(a=original_text, b=candidate_text).ratio()

    def _get_validation_mode(self) -> str:
        modes = []
        if self.config.diff_audit.enabled:
            modes.append("diff")
        if self.config.render_check.enabled:
            modes.append(self.config.render_check.backend)
        return "+".join(modes)

    async def process_batch(
        self, texts: List[str], max_concurrent: int = 5
    ) -> List[PipelineResult]:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(text: str) -> PipelineResult:
            async with semaphore:
                try:
                    return await self.process(text)
                except Exception as e:
                    logger.error(f"Failed to process text: {e}")
                    return PipelineResult(
                        original_text=text,
                        corrected_text=text,
                        diff_check_passed=False,
                        diff_check_issues=[str(e)],
                    )

        tasks = [process_with_semaphore(t) for t in texts]
        return await asyncio.gather(*tasks)


def save_results(results: List[PipelineResult], output_path: str):
    output = []
    for r in results:
        output.append({
            "original": r.original_text,
            "corrected": r.corrected_text,
            "detected_errors": r.detected_errors,
            "debate_rounds": r.debate_rounds,
            "validation_mode": r.validation_mode,
            "candidate_count": r.candidate_count,
            "judge_candidate_passed_validation": r.judge_candidate_passed_validation,
            "selected_candidate_passed_validation": r.selected_candidate_passed_validation,
            "selected_candidate_source": r.selected_candidate_source,
            "used_alternative_candidate": r.used_alternative_candidate,
            "validation_fallback_used": r.validation_fallback_used,
            "diff_check_passed": r.diff_check_passed,
            "diff_check_issues": r.diff_check_issues,
            "render_check_passed": r.render_check_passed,
            "render_check_issues": r.render_check_issues,
            "time": r.processing_time,
            "stage_times": r.stage_times,
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


async def run_pipeline(
    input_path: str,
    output_path: str,
    config_path: Optional[str] = None,
):
    config = load_config(config_path)
    pipeline = MathTECPipeline(
        models=config["models"],
        pipeline_config=config["pipeline"],
    )

    with open(input_path) as f:
        data = json.load(f)

    texts = [item["corrupted"] for item in data]

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Processing {len(texts)} samples...")

    results = await pipeline.process_batch(texts)
    save_results(results, output_path)
    logger.info(f"Results saved to {output_path}")
    return results
