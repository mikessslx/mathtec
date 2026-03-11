"""
MathTEC Configuration

Configure API keys, model choices and pipeline parameters.
"""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


# Model configuration

@dataclass
class ModelConfig:
    """Configuration for a single LLM provider/model."""
    provider: str # "openai" | "anthropic" | "deepseek" | "google"
    model: str # e.g. "gpt-5.4", "claude-sonnet-4.6", "deepseek-v3.2"
    api_key_env: str # environment variable name for API key
    base_url: Optional[str] = None # custom endpoint (for DeepSeek etc.)
    temperature: float = 0.3
    max_tokens: int = 4096

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise EnvironmentError(
                f"API key not found: set environment variable '{self.api_key_env}'"
            )
        return key


# Pipeline configuration

@dataclass
class DetectionConfig:
    """Stage 1: Multi-perspective error detection (DEEP-style)."""
    num_prompts: int = 6
    agreement_threshold: int = 2 # min prompts that must agree to confirm error
    temperature: float = 0.2 # low temp for detection consistency


@dataclass
class DebateConfig:
    """Stage 2: Multi-agent debate correction."""
    num_rounds: int = 2 # debate rounds after initial correction
    num_agents: int = 3
    temperature: float = 0.5 # moderate creativity for correction


@dataclass
class RenderCheckConfig:
    """Post-check: LaTeX renderer verification using KaTeX or MathJax."""
    enabled: bool = False
    backend: str = "mathjax"
    node_command: str = "node"
    timeout: float = 8.0


@dataclass
class DiffAuditConfig:
    """Post-check: diff-based minimal-edit audit."""
    enabled: bool = False
    use_llm_review: bool = True
    max_changed_char_ratio: float = 0.35
    max_edit_blocks: int = 8
    max_extra_span_ratio: float = 0.65
    max_snippet_chars: int = 120
    reviewer_temperature: float = 0.1


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    render_check: RenderCheckConfig = field(default_factory=RenderCheckConfig)
    diff_audit: DiffAuditConfig = field(default_factory=DiffAuditConfig)
    verbose: bool = True
    log_dir: str = "logs"
    results_dir: str = "results"


# Default model assignments

DEFAULT_MODELS = {
    # Stage 1: Detection
    "detector": ModelConfig(
        provider="openai",
        model="gpt-5.4",
        api_key_env="OPENAI_API_KEY",
        temperature=0.2,
    ),
    # Stage 2: Debate agents
    "agent_a": ModelConfig(
        provider="openai",
        model="gpt-5.4",
        api_key_env="OPENAI_API_KEY",
        temperature=0.5,
    ),
    "agent_b": ModelConfig(
        provider="anthropic",
        model="claude-sonnet-4.6",
        api_key_env="ANTHROPIC_API_KEY",
        temperature=0.5,
    ),
    "agent_c": ModelConfig(
        provider="deepseek",
        model="deepseek-v3.2",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        temperature=0.5,
    ),
    # Judge model
    "judge": ModelConfig(
        provider="openai",
        model="gpt-5.4",
        api_key_env="OPENAI_API_KEY",
        temperature=0.1,
    ),
    # Benchmark generation
    "data_generator": ModelConfig(
        provider="openai",
        model="gpt-5.4",
        api_key_env="OPENAI_API_KEY",
        temperature=0.7,
    ),
}


def make_gateway_models(
    base_url: str,
    api_key_env: str = "GATEWAY_API_KEY",
    models: Optional[Dict[str, str]] = None,
) -> Dict[str, "ModelConfig"]:
    """
    One-line setup for an OpenAI-compatible gateway.

    Usage:
        # in your script or config.yaml
        models = make_gateway_models(
            base_url="https://your-gateway.example.com/v1",
            models={
                "detector":       "gpt-5.4",
                "agent_a":        "gpt-5.4",
                "agent_b":        "claude-sonnet-4.6",
                "agent_c":        "deepseek-v3.2",
                "judge":          "gpt-5.4",
                "data_generator": "gpt-5.4",
            },
        )
    """
    if models is None:
        models = {
            "detector":       "gpt-5.4",
            "agent_a":        "gpt-5.4",
            "agent_b":        "claude-sonnet-4.6",
            "agent_c":        "deepseek-v3.2",
            "judge":          "gpt-5.4",
            "data_generator": "gpt-5.4",
        }

    temp_map = {
        "detector": 0.2,
        "judge": 0.1,
        "data_generator": 0.7,
    }

    result = {}
    for role, model_name in models.items():
        result[role] = ModelConfig(
            provider="openai",
            model=model_name,
            api_key_env=api_key_env,
            base_url=base_url,
            temperature=temp_map.get(role, 0.5),
        )
    return result


# Optional YAML overrides

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file, falling back to defaults."""
    config = {
        "models": DEFAULT_MODELS,
        "pipeline": PipelineConfig(),
    }

    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            overrides = yaml.safe_load(f) or {}

        # Override model configs
        if "models" in overrides:
            for role, mcfg in overrides["models"].items():
                config["models"][role] = ModelConfig(**mcfg)

        # Override pipeline configs
        if "pipeline" in overrides:
            p = overrides["pipeline"]
            if "detection" in p:
                config["pipeline"].detection = DetectionConfig(**p["detection"])
            if "debate" in p:
                config["pipeline"].debate = DebateConfig(**p["debate"])
            if "render_check" in p:
                config["pipeline"].render_check = RenderCheckConfig(**p["render_check"])
            if "diff_audit" in p:
                config["pipeline"].diff_audit = DiffAuditConfig(**p["diff_audit"])

    return config


# Benchmark configuration

@dataclass
class BenchmarkConfig:
    """Configuration for MathTEC-Bench construction."""
    num_samples: int = 500
    error_distribution: Dict[str, float] = field(default_factory=lambda: {
        "spelling": 0.25, # 25% spelling errors
        "ocr_symbol": 0.25, # 25% OCR symbol errors
        "ocr_formula": 0.20, # 20% OCR formula errors
        "variable": 0.05, # 5% variable inconsistency
        "handwriting_confusion": 0.25, # 25% handwritten glyph confusions
    })
    errors_per_sentence: tuple = (1, 3) # min, max errors per sample
    clean_sample_ratio: float = 0.10
    seed: int = 42
    source_texts_per_topic: int = 50
    math_topics: List[str] = field(default_factory=lambda: [
        "calculus",
        "linear_algebra",
        "probability_statistics",
        "real_analysis",
        "abstract_algebra",
        "number_theory",
        "differential_equations",
        "topology",
        "discrete_mathematics",
        "optimization",
    ])
