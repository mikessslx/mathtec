import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from config import RenderCheckConfig

logger = logging.getLogger(__name__)


LATEX_PATTERN = re.compile(
    r'\$\$(.+?)\$\$'
    r'|\$(.+?)\$'
    r'|\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}'
    r'|\\begin\{align\*?\}(.+?)\\end\{align\*?\}'
    r'|\\\[(.+?)\\\]',
    re.DOTALL,
)


@dataclass
class RenderCheckResult:
    is_valid: bool
    backend: str
    checked_count: int = 0
    issues: List[str] = field(default_factory=list)


class RendererChecker:
    def __init__(self, render_config: RenderCheckConfig):
        self.config = render_config
        self.script_path = Path(__file__).resolve().parents[1] / "tools" / "render_check.js"
        self._cache: Dict[str, Dict] = {}

    def check_text(self, text: str) -> RenderCheckResult:
        expressions = self.extract_latex_expressions(text)
        if not expressions:
            return RenderCheckResult(
                is_valid=True,
                backend=self.config.backend,
                checked_count=0,
            )

        uncached = [expr for expr in expressions if expr not in self._cache]
        if uncached:
            payload = {"backend": self.config.backend, "expressions": uncached}
            cmd = [self.config.node_command, str(self.script_path)]
            completed = subprocess.run(
                cmd,
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                check=False,
            )
            if completed.returncode != 0:
                stderr = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(
                    f"Renderer check failed with backend '{self.config.backend}': {stderr}"
                )
            try:
                parsed = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Renderer check returned invalid JSON: {exc}") from exc

            if len(parsed) != len(uncached):
                raise RuntimeError("Renderer check returned mismatched result count")

            for expr, result in zip(uncached, parsed):
                self._cache[expr] = result

        issues: List[str] = []
        for expr in expressions:
            cached = self._cache[expr]
            if not cached.get("ok", False):
                message = cached.get("error", "unknown renderer error")
                issues.append(f"Render check failed for '${expr}$': {message}")

        return RenderCheckResult(
            is_valid=not issues,
            backend=self.config.backend,
            checked_count=len(expressions),
            issues=issues,
        )

    def extract_latex_expressions(self, text: str) -> List[str]:
        expressions: List[str] = []
        for match in LATEX_PATTERN.finditer(text):
            latex_content = next(group for group in match.groups() if group is not None)
            expressions.append(latex_content.strip())
        return expressions
