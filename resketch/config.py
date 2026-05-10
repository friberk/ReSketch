from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from resketch.models import DecompositionMode, MatchMode, ProviderKind, SynthesisStrategy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class AppInfoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str


class MatchingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: MatchMode


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    temperature: float
    timeout_seconds: float
    max_retries: int
    max_candidates: int
    missing_confidence: float
    response_format: dict[str, Any]
    system_prompt: str
    user_prompt_template: str


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderKind
    fixture_path: str
    db_path: str
    cache_enabled: bool
    cache_path: str
    deduplicate: bool


class DecompositionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    mode: DecompositionMode
    allow_empty_hole_examples: bool
    max_soft_examples_per_hole: int
    max_assignments_per_example: int
    max_hole_capture_length: int
    max_repeat_unroll: int
    capture_policies: list[str]


class SemanticFitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    positive_match: float
    negative_reject: float
    hard_local_match: float
    hard_local_reject: float
    soft_positive_match: float
    tuple_negative_probe_reject: float


class SourceConfidenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weight: float


class SyntacticComplexityCostConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    literal: float
    dot: float
    category: float
    char_class: float
    concat: float
    alt: float
    repeat: float
    anchor: float
    raw_regex: float
    wildcard_penalty: float
    alternation_option_penalty: float
    length_penalty: float


class ScoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic_fit: SemanticFitConfig
    source_confidence: SourceConfidenceConfig
    syntactic_complexity_cost: SyntacticComplexityCostConfig
    strategy_penalty: dict[SynthesisStrategy, float]


class QuantifierRefinementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lower_delta: int
    upper_delta: int


class SearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_ast_size: int
    max_nodes_per_size: int
    max_behavior_classes: int
    max_generated_candidates: int
    max_seconds_per_hole: float
    max_constraint_probes_per_hole: int
    max_example_literals: int
    example_literal_max_length: int
    constructors: list[str]
    allowed_repeat_bounds: list[tuple[int, int | None]]
    separator_chars: str
    allow_raw_regex: bool
    behavioral_dedup: bool


class RefinementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_class_candidates: list[str]
    enable_anchor_toggle: bool
    enable_separator_insertion: bool
    enable_optional_literal: bool
    max_variants_per_operator: int


class GlobalPruningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    unknown_hole_pattern: str
    max_partial_beams: int


class AutomataPruningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    alphabet_policy: str
    max_unknown_hole_length: int
    max_states: int
    fail_open_on_unsupported: bool


class OutcomeLoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    path: str


class SynthesisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beam_size: int
    max_hole_options: int
    max_fragment_pairs: int
    max_refinement_variants: int
    max_candidate_length: int
    strategy_order: list[SynthesisStrategy]
    scoring: ScoringConfig
    quantifier_refinement: QuantifierRefinementConfig
    search: SearchConfig
    refinement: RefinementConfig
    global_pruning: GlobalPruningConfig
    automata: AutomataPruningConfig
    outcome_logging: OutcomeLoggingConfig


class CegisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interactive: bool
    max_rounds: int
    accept_tokens: list[str]
    reject_tokens: list[str]
    quit_tokens: list[str]
    positive_tokens: list[str]
    negative_tokens: list[str]
    prompt_accept: str
    prompt_counterexample: str
    prompt_counterexample_kind: str


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark_path: str


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app: AppInfoConfig
    matching: MatchingConfig
    llm: LLMConfig
    retrieval: RetrievalConfig
    decomposition: DecompositionConfig
    synthesis: SynthesisConfig
    cegis: CegisConfig
    evaluation: EvaluationConfig


def deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def set_dotted_value(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor = data
    for key in keys[:-1]:
        child = cursor.setdefault(key, {})
        if not isinstance(child, dict):
            msg = f"Cannot override non-object config key: {dotted_key}"
            raise ValueError(msg)
        cursor = child
    cursor[keys[-1]] = value


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = f"Config file must contain a mapping: {path}"
        raise ValueError(msg)
    return data


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    data = load_yaml(DEFAULT_CONFIG_PATH)
    if path is not None:
        data = deep_merge(data, load_yaml(Path(path)))

    if overrides:
        data = dict(data)
        for key, value in overrides.items():
            if value is not None:
                set_dotted_value(data, key, value)

    return AppConfig.model_validate(data)


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
