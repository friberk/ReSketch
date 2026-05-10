from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProviderKind(StrEnum):
    LLM = "llm"
    DB = "db"
    FIXTURE = "fixture"


class MatchMode(StrEnum):
    FULLMATCH = "fullmatch"
    MATCH = "match"
    SEARCH = "search"


class SynthesisStrategy(StrEnum):
    DIRECT_REUSE = "direct_reuse"
    COMPONENT_GENERALIZATION = "component_generalization"
    FRAGMENT_RECOMBINATION = "fragment_recombination"
    PARAMETERIZED_REFINEMENT = "parameterized_refinement"


class DecompositionMode(StrEnum):
    OFF = "off"
    EXPLICIT_ONLY = "explicit-only"
    HARD_ONLY = "hard-only"
    HARD_AND_SOFT = "hard-and-soft"


class EvidenceKind(StrEnum):
    HARD_POSITIVE = "hard_positive"
    SOFT_POSITIVE = "soft_positive"
    EXPLICIT_POSITIVE = "explicit_positive"
    EXPLICIT_NEGATIVE = "explicit_negative"
    INFERRED_NEGATIVE = "inferred_negative"


class CompletenessStatus(StrEnum):
    COMPLETE_WITHIN_BOUNDS = "complete_within_bounds"
    INCOMPLETE_DUE_TO_TIMEOUT = "incomplete_due_to_timeout"
    INCOMPLETE_DUE_TO_LIMIT = "incomplete_due_to_limit"
    INCOMPLETE_DUE_TO_RAW_REGEX = "incomplete_due_to_raw_regex"
    INCOMPLETE_DUE_TO_UNSUPPORTED_AUTOMATA = "incomplete_due_to_unsupported_automata"
    INCOMPLETE_DUE_TO_NON_FULLMATCH = "incomplete_due_to_non_fullmatch"


class Examples(BaseModel):
    model_config = ConfigDict(extra="forbid")

    positive: list[str] = Field(default_factory=list)
    negative: list[str] = Field(default_factory=list)


class SynthesisSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    global_examples: Examples = Field(default_factory=Examples)
    explicit_hole_examples: dict[str, Examples] = Field(default_factory=dict)


class Hole(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identifier: str
    semantic_type: str
    start: int
    end: int


class RegexComponent(BaseModel):
    model_config = ConfigDict(extra="allow")

    regex: str
    type: str
    description: str
    confidence: float | None = None
    source_id: str | None = None
    examples: Examples = Field(default_factory=Examples)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HoleEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    source_example: str
    kind: EvidenceKind
    confidence: float
    policies: list[str] = Field(default_factory=list)
    reason: str


class HoleExampleSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hard: Examples = Field(default_factory=Examples)
    soft_positive: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    evidence: list[HoleEvidence] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)


class CandidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hole: Hole
    sketch: str
    global_examples: Examples
    hole_examples: HoleExampleSet
    max_candidates: int


class DecompositionStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    global_positive_count: int = 0
    matched_positive_count: int = 0
    unmatched_positive_count: int = 0
    holes_with_hard_examples: int = 0
    holes_with_only_soft_examples: int = 0
    hard_evidence_count: int = 0
    soft_evidence_count: int = 0
    ambiguous_example_count: int = 0
    ambiguity_group_count: int = 0
    repeated_hole_constraint_count: int = 0
    explicit_evidence_count: int = 0
    inferred_negative_count: int = 0
    tuple_negative_constraint_count: int = 0
    total_assignments_considered: int = 0
    examples_with_assignments: int = 0
    truncated_assignment_count: int = 0
    repeat_unroll_truncation_count: int = 0
    average_assignments_per_example: float = 0.0
    decomposition_success_rate: float = 0.0
    hard_coverage_rate: float = 0.0

    def recompute_rates(self) -> None:
        if self.examples_with_assignments == 0:
            self.average_assignments_per_example = 0.0
        else:
            self.average_assignments_per_example = (
                self.total_assignments_considered / self.examples_with_assignments
            )

        if self.global_positive_count == 0:
            self.decomposition_success_rate = 0.0
        else:
            self.decomposition_success_rate = (
                self.matched_positive_count / self.global_positive_count
            )

        total_holes = self.holes_with_hard_examples + self.holes_with_only_soft_examples
        if total_holes == 0:
            self.hard_coverage_rate = 0.0
        else:
            self.hard_coverage_rate = self.holes_with_hard_examples / total_holes


class TupleNegativeConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_example: str
    hole_values: dict[str, list[str]]
    reason: str
    confidence: float = 1.0


class AmbiguityGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_example: str
    choices: list[dict[str, list[str]]] = Field(default_factory=list)
    reason: str
    confidence: float = 0.0


class RepeatedHoleConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_example: str
    hole_id: str
    occurrence_count: int
    values: list[str] = Field(default_factory=list)
    reason: str
    confidence: float = 1.0


class ConstraintSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tuple_negative_constraints: list[TupleNegativeConstraint] = Field(default_factory=list)
    ambiguity_groups: list[AmbiguityGroup] = Field(default_factory=list)
    repeated_hole_constraints: list[RepeatedHoleConstraint] = Field(default_factory=list)


class DecompositionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    by_hole: dict[str, HoleExampleSet] = Field(default_factory=dict)
    tuple_negative_constraints: list[TupleNegativeConstraint] = Field(default_factory=list)
    ambiguity_groups: list[AmbiguityGroup] = Field(default_factory=list)
    repeated_hole_constraints: list[RepeatedHoleConstraint] = Field(default_factory=list)
    constraint_set: ConstraintSet = Field(default_factory=ConstraintSet)
    unmatched_positive: list[str] = Field(default_factory=list)
    stats: DecompositionStats = Field(default_factory=DecompositionStats)
    diagnostics: list[str] = Field(default_factory=list)


class CandidateSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderKind
    hole: Hole
    components: list[RegexComponent]
    trace: dict[str, Any] = Field(default_factory=dict)


class CandidateEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    positive_total: int
    positive_matched: int
    negative_total: int
    negative_rejected: int
    failures: list[str] = Field(default_factory=list)

    @property
    def success(self) -> bool:
        return (
            self.positive_total == self.positive_matched
            and self.negative_total == self.negative_rejected
        )


class SearchStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_candidates: int = 0
    canonicalized_candidates: int = 0
    evaluated_candidates: int = 0
    pruned_invalid_regexes: int = 0
    pruned_local_example_failures: int = 0
    pruned_duplicates: int = 0
    pruned_canonical_duplicates: int = 0
    pruned_behavior_duplicates: int = 0
    pruned_constraint_failures: int = 0
    pruned_global_positive_feasibility: int = 0
    pruned_global_negative_feasibility: int = 0
    pruned_tuple_negative_constraints: int = 0
    constraint_probe_count: int = 0
    raw_regex_candidates: int = 0
    frontier_size: int = 0
    bounds_hit: bool = False
    timed_out: bool = False
    elapsed_seconds: float = 0.0
    strategy_counts: dict[str, int] = Field(default_factory=dict)


class ProvenanceStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: SynthesisStrategy
    operation: str
    inputs: list[str] = Field(default_factory=list)
    output: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompletenessScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    regex_subset: str
    status: CompletenessStatus = CompletenessStatus.COMPLETE_WITHIN_BOUNDS
    incomplete_reasons: list[str] = Field(default_factory=list)
    max_ast_size: int
    max_generated_candidates_per_hole: int
    max_seconds_per_hole: float
    constructors: list[str] = Field(default_factory=list)
    automata_pruning_enabled: bool = False
    automata_alphabet_policy: str | None = None
    automata_alphabet_size: int = 0
    component_count: int = 0
    candidate_count_by_hole: dict[str, int] = Field(default_factory=dict)
    timed_out: bool = False
    limits_hit: bool = False
    raw_regex_candidate_count: int = 0
    unsupported_automata_reasons: list[str] = Field(default_factory=list)
    complete_within_bounds: bool = True


class AutomataStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    alphabet_policy: str | None = None
    alphabet_size: int = 0
    feasibility_checks: int = 0
    negative_feasibility_checks: int = 0
    pruned_positive_feasibility: int = 0
    pruned_negative_feasibility: int = 0
    fail_open_count: int = 0
    max_states_seen: int = 0
    unsupported_reasons: list[str] = Field(default_factory=list)


class CandidateScoreBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic_fit_score: float
    source_confidence_score: float
    ast_complexity_cost: float
    length_complexity_cost: float
    syntactic_complexity_cost: float
    strategy_penalty: float
    total_score: float


class CandidateFeatureVector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    regex: str
    strategy: SynthesisStrategy
    confidence: float | None = None
    source_id: str | None = None
    length: int
    score_breakdown: CandidateScoreBreakdown
    constraint_probe_matches: int = 0
    constraint_probe_rejections: int = 0


class SynthesisTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_sets: list[CandidateSet] = Field(default_factory=list)
    candidates_explored: int
    strategies_used: list[SynthesisStrategy] = Field(default_factory=list)
    decomposition: DecompositionResult | None = None
    hole_assignments: dict[str, str] = Field(default_factory=dict)
    hole_order: list[str] = Field(default_factory=list)
    hole_witnesses: dict[str, list[str]] = Field(default_factory=dict)
    search_stats: dict[str, SearchStats] = Field(default_factory=dict)
    automata_stats: AutomataStats = Field(default_factory=AutomataStats)
    candidate_features: dict[str, list[CandidateFeatureVector]] = Field(default_factory=dict)
    selected_option_provenance: dict[str, dict[str, Any]] = Field(default_factory=dict)
    selected_provenance_steps: dict[str, list[ProvenanceStep]] = Field(default_factory=dict)
    completeness_scope: CompletenessScope | None = None
    pruned_partial_beams: int = 0
    pruned_tuple_negative_constraints: int = 0
    diagnostics: list[str] = Field(default_factory=list)


class SynthesisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    regex: str | None
    success: bool
    score: float
    evaluation: CandidateEvaluation | None = None
    trace: SynthesisTrace
