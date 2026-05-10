from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from resketch.automata_engine import (
    AutomataFeasibilityResult,
    check_complete_negative_rejection,
    check_partial_positive_feasibility,
)
from resketch.config import AppConfig, resolve_project_path
from resketch.decomposition import capture_witnesses, decompose_examples
from resketch.models import (
    AutomataStats,
    CandidateEvaluation,
    CandidateFeatureVector,
    CandidateRequest,
    CandidateScoreBreakdown,
    CandidateSet,
    CompletenessScope,
    CompletenessStatus,
    DecompositionResult,
    Examples,
    Hole,
    HoleExampleSet,
    MatchMode,
    ProvenanceStep,
    RegexComponent,
    RepeatedHoleConstraint,
    SearchStats,
    SynthesisResult,
    SynthesisSpec,
    SynthesisStrategy,
    SynthesisTrace,
    TupleNegativeConstraint,
)
from resketch.regex_engine import (
    RegexValidationError,
    evaluate_examples,
    regex_matches,
)
from resketch.retrieval.base import CandidateProvider
from resketch.scoring import candidate_score_breakdown
from resketch.search import RegexSearchEngine, SearchOption, SearchResult
from resketch.sketch import Sketch, SketchHole, parse_sketch


@dataclass(frozen=True)
class HoleOption:
    regex: str
    strategy: SynthesisStrategy
    confidence: float | None
    source_id: str | None
    semantic_fit_score: float
    ast_complexity_cost: float
    provenance: dict[str, Any]
    provenance_steps: tuple[ProvenanceStep, ...]
    behavior_signature: tuple[object, ...]
    constraint_probe_matches: int = 0
    constraint_probe_rejections: int = 0


@dataclass(frozen=True)
class BeamState:
    assignments: dict[str, str]
    strategies: tuple[SynthesisStrategy, ...]
    provenance: tuple[tuple[str, dict[str, Any]], ...]
    provenance_steps: tuple[tuple[str, tuple[ProvenanceStep, ...]], ...]
    score: float
    candidates_explored: int


class Synthesizer:
    def __init__(self, config: AppConfig, provider: CandidateProvider) -> None:
        self._config = config
        self._provider = provider
        self._search = RegexSearchEngine(config)
        self._candidate_cache: dict[str, CandidateSet] = {}
        self._search_cache: dict[str, SearchResult] = {}

    def synthesize(
        self,
        sketch_source: str,
        spec_or_examples: SynthesisSpec | Examples,
    ) -> SynthesisResult:
        spec = _normalize_spec(spec_or_examples)
        sketch = parse_sketch(sketch_source)
        if not sketch.holes:
            result = self._evaluate_finished_candidate(
                sketch_source,
                spec.global_examples,
                [],
                [],
                decomposition=None,
                hole_assignments={},
                sketch=None,
                diagnostics=[],
            )
            self._log_outcome(sketch_source, spec, result)
            return result

        decomposition = decompose_examples(
            self._config,
            sketch,
            spec.global_examples,
            spec.explicit_hole_examples,
        )
        candidate_sets = self._retrieve_candidate_sets(sketch, spec.global_examples, decomposition)
        search_results = {}
        for candidate_set in candidate_sets:
            components = self._embedding_safe_components(
                sketch,
                candidate_set.hole.identifier,
                candidate_set.components,
            )
            negative_probes = _negative_probes_for_hole(
                decomposition,
                candidate_set.hole.identifier,
                self._config.synthesis.search.max_constraint_probes_per_hole,
            )
            search_results[candidate_set.hole.identifier] = self._search_candidate_set(
                components,
                decomposition.by_hole.get(candidate_set.hole.identifier, HoleExampleSet()),
                negative_probes,
            )
        options_by_hole = {
            hole_id: self._embedding_safe_options(
                sketch,
                hole_id,
                [_option_from_search_option(option) for option in result.options],
            )
            for hole_id, result in search_results.items()
        }
        candidate_features = {
            hole_id: [
                self._candidate_feature(option)
                for option in options
            ]
            for hole_id, options in options_by_hole.items()
        }
        search_stats = {
            hole_id: result.stats
            for hole_id, result in search_results.items()
        }
        automata_stats = AutomataStats(
            enabled=self._config.synthesis.automata.enabled,
            alphabet_policy=self._config.synthesis.automata.alphabet_policy,
        )
        completeness_scope = self._completeness_scope(
            candidate_sets,
            search_stats,
            automata_stats,
        )

        diagnostics: list[str] = [
            *decomposition.diagnostics,
            *[
                f"positive example did not match decomposition skeleton: {example!r}"
                for example in decomposition.unmatched_positive
            ],
        ]
        for hole in sketch.holes:
            if not options_by_hole.get(hole.identifier):
                diagnostics.append(
                    f"No candidates available for hole {hole.identifier}:{hole.semantic_type}"
                )

        initial_candidates_explored = sum(
            stats.evaluated_candidates
            for stats in search_stats.values()
        )
        beams = [
            BeamState(
                assignments={},
                strategies=(),
                provenance=(),
                provenance_steps=(),
                score=0.0,
                candidates_explored=initial_candidates_explored,
            )
        ]
        pruned_partial_beams = 0
        pruned_tuple_negative_constraints = 0
        hole_order = self._ordered_holes(sketch, decomposition, options_by_hole)
        for hole in hole_order:
            next_beams: list[BeamState] = []
            for beam in beams:
                for option in options_by_hole.get(hole.identifier, []):
                    assignments = dict(beam.assignments)
                    assignments[hole.identifier] = option.regex
                    if self._violates_tuple_negative_constraints(
                        assignments,
                        decomposition.tuple_negative_constraints,
                    ):
                        pruned_tuple_negative_constraints += 1
                        search_stats[hole.identifier].pruned_tuple_negative_constraints += 1
                        continue
                    if self._violates_repeated_hole_constraints(
                        assignments,
                        decomposition.repeated_hole_constraints,
                    ):
                        search_stats[hole.identifier].pruned_constraint_failures += 1
                        continue
                    if self._config.synthesis.global_pruning.enabled and not (
                        self._partial_assignment_feasible(
                            sketch,
                            assignments,
                            spec.global_examples,
                            automata_stats,
                        )
                    ):
                        pruned_partial_beams += 1
                        search_stats[hole.identifier].pruned_global_positive_feasibility += 1
                        continue
                    if (
                        self._config.synthesis.global_pruning.enabled
                        and len(assignments) == len(sketch.holes)
                        and not self._complete_assignment_rejects_negatives(
                            sketch,
                            assignments,
                            spec.global_examples,
                            automata_stats,
                        )
                    ):
                        search_stats[hole.identifier].pruned_global_negative_feasibility += 1
                        continue
                    score = beam.score + self._candidate_ranking_score(option)
                    next_beams.append(
                        BeamState(
                            assignments=assignments,
                            strategies=beam.strategies + (option.strategy,),
                            provenance=beam.provenance + (
                                (hole.identifier, option.provenance),
                            ),
                            provenance_steps=beam.provenance_steps + (
                                (hole.identifier, option.provenance_steps),
                            ),
                            score=score,
                            candidates_explored=beam.candidates_explored + 1,
                        )
                    )
            beam_limit = self._config.synthesis.beam_size
            if self._config.synthesis.global_pruning.enabled:
                beam_limit = min(
                    beam_limit,
                    self._config.synthesis.global_pruning.max_partial_beams,
                )
            beams = sorted(next_beams, key=lambda beam: beam.score, reverse=True)[:beam_limit]

        self._update_completeness_scope(completeness_scope, search_stats, automata_stats)
        results = [
            self._evaluate_beam(
                sketch,
                spec.global_examples,
                beam,
                candidate_sets,
                decomposition,
                search_stats,
                diagnostics,
                automata_stats,
                candidate_features,
                [hole.identifier for hole in hole_order],
                completeness_scope,
                pruned_partial_beams,
                pruned_tuple_negative_constraints,
            )
            for beam in beams
        ]
        if not results:
            negative_prunes = sum(
                stats.pruned_global_negative_feasibility
                for stats in search_stats.values()
            )
            result = SynthesisResult(
                regex=None,
                success=False,
                score=0.0,
                evaluation=None,
                trace=SynthesisTrace(
                    candidate_sets=candidate_sets,
                    candidates_explored=sum(
                        stats.evaluated_candidates for stats in search_stats.values()
                    ),
                    strategies_used=[],
                    decomposition=decomposition,
                    hole_order=[hole.identifier for hole in hole_order],
                    search_stats=search_stats,
                    automata_stats=automata_stats,
                    candidate_features=candidate_features,
                    completeness_scope=completeness_scope,
                    pruned_partial_beams=pruned_partial_beams,
                    pruned_tuple_negative_constraints=pruned_tuple_negative_constraints,
                    diagnostics=[
                        *diagnostics,
                        "No complete assignment survived global assembly.",
                        (
                            "Pruned assignments: "
                            f"partial_positive={pruned_partial_beams}, "
                            f"tuple_negative={pruned_tuple_negative_constraints}, "
                            f"global_negative={negative_prunes}."
                        ),
                    ],
                ),
            )
            self._log_outcome(sketch_source, spec, result)
            return result

        results.sort(key=lambda result: (result.success, result.score), reverse=True)
        self._log_outcome(sketch_source, spec, results[0])
        return results[0]

    def _retrieve_candidate_sets(
        self,
        sketch: Sketch,
        global_examples: Examples,
        decomposition: DecompositionResult,
    ) -> list[CandidateSet]:
        candidate_sets: list[CandidateSet] = []
        for hole in sketch.holes:
            request = CandidateRequest(
                hole=hole,
                sketch=sketch.source,
                global_examples=global_examples,
                hole_examples=decomposition.by_hole.get(hole.identifier, HoleExampleSet()),
                max_candidates=self._config.llm.max_candidates,
            )
            cache_key = _json_cache_key(request.model_dump(mode="json"))
            cached = self._candidate_cache.get(cache_key)
            if cached is not None:
                candidate_set = deepcopy(cached)
                candidate_set.trace["from_synthesis_cache"] = True
            else:
                candidate_set = self._provider.retrieve(request)
                self._candidate_cache[cache_key] = deepcopy(candidate_set)
            candidate_sets.append(candidate_set)
        return candidate_sets

    def _search_candidate_set(
        self,
        components: list[RegexComponent],
        hole_examples: HoleExampleSet,
        negative_probes: list[str],
    ) -> SearchResult:
        cache_key = _json_cache_key(
            {
                "components": [
                    component.model_dump(mode="json")
                    for component in components
                ],
                "hole_examples": hole_examples.model_dump(mode="json"),
                "negative_probes": negative_probes,
                "strategy_order": [
                    strategy.value
                    for strategy in self._config.synthesis.strategy_order
                ],
                "scoring": self._config.synthesis.scoring.model_dump(mode="json"),
                "search": self._config.synthesis.search.model_dump(mode="json"),
                "refinement": self._config.synthesis.refinement.model_dump(mode="json"),
            }
        )
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            return deepcopy(cached)
        result = self._search.search(
            components,
            hole_examples,
            negative_probes=negative_probes,
        )
        self._search_cache[cache_key] = deepcopy(result)
        return result

    def _candidate_ranking_score(self, option: HoleOption) -> float:
        return self._candidate_score_breakdown(option).total_score

    def _candidate_score_breakdown(self, option: HoleOption) -> CandidateScoreBreakdown:
        return candidate_score_breakdown(
            regex=option.regex,
            strategy=option.strategy,
            confidence=option.confidence,
            semantic_fit_score=option.semantic_fit_score,
            ast_complexity_cost=option.ast_complexity_cost,
            config=self._config,
        )

    def _evaluate_beam(
        self,
        sketch: Sketch,
        examples: Examples,
        beam: BeamState,
        candidate_sets: list[CandidateSet],
        decomposition: DecompositionResult,
        search_stats: dict[str, SearchStats],
        diagnostics: list[str],
        automata_stats: AutomataStats,
        candidate_features: dict[str, list[CandidateFeatureVector]],
        hole_order: list[str],
        completeness_scope: CompletenessScope,
        pruned_partial_beams: int,
        pruned_tuple_negative_constraints: int,
    ) -> SynthesisResult:
        regex = sketch.render(beam.assignments)
        result = self._evaluate_finished_candidate(
            regex,
            examples,
            list(beam.strategies),
            candidate_sets,
            decomposition=decomposition,
            hole_assignments=beam.assignments,
            selected_option_provenance=dict(beam.provenance),
            selected_option_provenance_steps=dict(beam.provenance_steps),
            search_stats=search_stats,
            sketch=sketch,
            diagnostics=diagnostics,
            automata_stats=automata_stats,
            candidate_features=candidate_features,
            hole_order=hole_order,
            candidates_explored=beam.candidates_explored,
            completeness_scope=completeness_scope,
            pruned_partial_beams=pruned_partial_beams,
            pruned_tuple_negative_constraints=pruned_tuple_negative_constraints,
        )
        result.score += beam.score
        return result

    def _evaluate_finished_candidate(
        self,
        regex: str,
        examples: Examples,
        strategies: list[SynthesisStrategy],
        candidate_sets: list[CandidateSet],
        *,
        decomposition: DecompositionResult | None,
        hole_assignments: dict[str, str],
        diagnostics: list[str],
        sketch: Sketch | None = None,
        selected_option_provenance: dict[str, dict[str, Any]] | None = None,
        selected_option_provenance_steps: dict[str, tuple[ProvenanceStep, ...]] | None = None,
        search_stats: dict[str, SearchStats] | None = None,
        automata_stats: AutomataStats | None = None,
        candidate_features: dict[str, list[CandidateFeatureVector]] | None = None,
        hole_order: list[str] | None = None,
        candidates_explored: int = 1,
        completeness_scope: CompletenessScope | None = None,
        pruned_partial_beams: int = 0,
        pruned_tuple_negative_constraints: int = 0,
    ) -> SynthesisResult:
        try:
            evaluation = evaluate_examples(regex, examples, self._config.matching.mode)
        except RegexValidationError as exc:
            return SynthesisResult(
                regex=regex,
                success=False,
                score=float("-inf"),
                evaluation=None,
                trace=SynthesisTrace(
                    candidate_sets=candidate_sets,
                    candidates_explored=candidates_explored,
                    strategies_used=strategies,
                    decomposition=decomposition,
                    hole_assignments=hole_assignments,
                    hole_order=hole_order or [],
                    hole_witnesses={},
                    search_stats=search_stats or {},
                    automata_stats=automata_stats or AutomataStats(),
                    candidate_features=candidate_features or {},
                    selected_option_provenance=selected_option_provenance or {},
                    selected_provenance_steps=_dump_provenance_steps(
                        selected_option_provenance_steps
                    ),
                    completeness_scope=completeness_scope,
                    pruned_partial_beams=pruned_partial_beams,
                    pruned_tuple_negative_constraints=pruned_tuple_negative_constraints,
                    diagnostics=[*diagnostics, str(exc)],
                ),
            )

        return SynthesisResult(
            regex=regex,
            success=evaluation.success,
            score=self._score_evaluation(regex, evaluation),
            evaluation=evaluation,
            trace=SynthesisTrace(
                candidate_sets=candidate_sets,
                candidates_explored=candidates_explored,
                strategies_used=strategies,
                decomposition=decomposition,
                hole_assignments=hole_assignments,
                hole_order=hole_order or [],
                hole_witnesses=(
                    capture_witnesses(
                        self._config,
                        sketch,
                        hole_assignments,
                        examples,
                    )
                    if sketch is not None and hole_assignments
                    else {}
                ),
                search_stats=search_stats or {},
                automata_stats=automata_stats or AutomataStats(),
                candidate_features=candidate_features or {},
                selected_option_provenance=selected_option_provenance or {},
                selected_provenance_steps=_dump_provenance_steps(
                    selected_option_provenance_steps
                ),
                completeness_scope=completeness_scope,
                pruned_partial_beams=pruned_partial_beams,
                pruned_tuple_negative_constraints=pruned_tuple_negative_constraints,
                diagnostics=diagnostics,
            ),
        )

    def _score_evaluation(self, regex: str, evaluation: CandidateEvaluation) -> float:
        semantic_fit = self._config.synthesis.scoring.semantic_fit
        length_penalty = (
            len(regex)
            * self._config.synthesis.scoring.syntactic_complexity_cost.length_penalty
        )
        return (
            evaluation.positive_matched * semantic_fit.positive_match
            + evaluation.negative_rejected * semantic_fit.negative_reject
            - length_penalty
        )

    def _partial_assignment_feasible(
        self,
        sketch: Sketch,
        assignments: dict[str, str],
        examples: Examples,
        automata_stats: AutomataStats,
    ) -> bool:
        if not examples.positive:
            return True
        if self._config.synthesis.automata.enabled:
            automata_stats.feasibility_checks += 1
            result = check_partial_positive_feasibility(
                self._config,
                sketch,
                assignments,
                examples,
            )
            self._record_automata_result(automata_stats, result)
            if not result.supported:
                if not self._config.synthesis.automata.fail_open_on_unsupported:
                    return result.feasible
                return self._regex_partial_assignment_feasible(sketch, assignments, examples)
            if not result.feasible:
                automata_stats.pruned_positive_feasibility += 1
            return result.feasible
        return self._regex_partial_assignment_feasible(sketch, assignments, examples)

    def _complete_assignment_rejects_negatives(
        self,
        sketch: Sketch,
        assignments: dict[str, str],
        examples: Examples,
        automata_stats: AutomataStats,
    ) -> bool:
        if not examples.negative:
            return True
        if self._config.synthesis.automata.enabled:
            automata_stats.negative_feasibility_checks += 1
            result = check_complete_negative_rejection(
                self._config,
                sketch,
                assignments,
                examples,
            )
            self._record_automata_result(automata_stats, result)
            if not result.supported:
                if not self._config.synthesis.automata.fail_open_on_unsupported:
                    return result.feasible
                return self._regex_complete_assignment_rejects_negatives(
                    sketch,
                    assignments,
                    examples,
                )
            if not result.feasible:
                automata_stats.pruned_negative_feasibility += 1
            return result.feasible
        return self._regex_complete_assignment_rejects_negatives(sketch, assignments, examples)

    def _record_automata_result(
        self,
        automata_stats: AutomataStats,
        result: AutomataFeasibilityResult,
    ) -> None:
        automata_stats.alphabet_size = max(automata_stats.alphabet_size, result.alphabet_size)
        automata_stats.max_states_seen = max(automata_stats.max_states_seen, result.state_count)
        if result.supported:
            return
        automata_stats.fail_open_count += 1
        for reason in result.reasons:
            if reason not in automata_stats.unsupported_reasons:
                automata_stats.unsupported_reasons.append(reason)

    def _regex_partial_assignment_feasible(
        self,
        sketch: Sketch,
        assignments: dict[str, str],
        examples: Examples,
    ) -> bool:
        try:
            partial_regex = sketch.render_partial(
                assignments,
                self._config.synthesis.global_pruning.unknown_hole_pattern,
            )
            return all(
                regex_matches(partial_regex, example, self._config.matching.mode)
                for example in examples.positive
            )
        except RegexValidationError:
            return False

    def _regex_complete_assignment_rejects_negatives(
        self,
        sketch: Sketch,
        assignments: dict[str, str],
        examples: Examples,
    ) -> bool:
        try:
            regex = sketch.render(assignments)
            return all(
                not regex_matches(regex, example, self._config.matching.mode)
                for example in examples.negative
            )
        except (KeyError, RegexValidationError):
            return False

    def _completeness_scope(
        self,
        candidate_sets: list[CandidateSet],
        search_stats: dict[str, SearchStats],
        automata_stats: AutomataStats,
    ) -> CompletenessScope:
        timed_out = any(stats.timed_out for stats in search_stats.values())
        limits_hit = any(stats.bounds_hit for stats in search_stats.values())
        raw_regex_count = sum(stats.raw_regex_candidates for stats in search_stats.values())
        return CompletenessScope(
            regex_subset="regular",
            max_ast_size=self._config.synthesis.search.max_ast_size,
            max_generated_candidates_per_hole=(
                self._config.synthesis.search.max_generated_candidates
            ),
            max_seconds_per_hole=self._config.synthesis.search.max_seconds_per_hole,
            constructors=self._config.synthesis.search.constructors,
            automata_pruning_enabled=automata_stats.enabled,
            automata_alphabet_policy=automata_stats.alphabet_policy,
            automata_alphabet_size=automata_stats.alphabet_size,
            component_count=sum(len(candidate_set.components) for candidate_set in candidate_sets),
            candidate_count_by_hole={
                candidate_set.hole.identifier: len(candidate_set.components)
                for candidate_set in candidate_sets
            },
            timed_out=timed_out,
            limits_hit=limits_hit,
            raw_regex_candidate_count=raw_regex_count,
            unsupported_automata_reasons=automata_stats.unsupported_reasons,
            complete_within_bounds=not timed_out and not limits_hit,
        )

    def _update_completeness_scope(
        self,
        scope: CompletenessScope,
        search_stats: dict[str, SearchStats],
        automata_stats: AutomataStats,
    ) -> None:
        timed_out = any(stats.timed_out for stats in search_stats.values())
        limits_hit = any(stats.bounds_hit for stats in search_stats.values())
        raw_regex_count = sum(stats.raw_regex_candidates for stats in search_stats.values())
        reasons: list[str] = []
        status = CompletenessStatus.COMPLETE_WITHIN_BOUNDS
        if timed_out:
            status = CompletenessStatus.INCOMPLETE_DUE_TO_TIMEOUT
            reasons.append("at least one hole search timed out")
        if limits_hit:
            if status is CompletenessStatus.COMPLETE_WITHIN_BOUNDS:
                status = CompletenessStatus.INCOMPLETE_DUE_TO_LIMIT
            reasons.append("at least one configured search bound was hit")
        if raw_regex_count:
            if status is CompletenessStatus.COMPLETE_WITHIN_BOUNDS:
                status = CompletenessStatus.INCOMPLETE_DUE_TO_RAW_REGEX
            reasons.append("raw regex candidates are outside the formal subset")
        if (
            self._config.synthesis.automata.enabled
            and self._config.matching.mode is not MatchMode.FULLMATCH
        ):
            if status is CompletenessStatus.COMPLETE_WITHIN_BOUNDS:
                status = CompletenessStatus.INCOMPLETE_DUE_TO_NON_FULLMATCH
            reasons.append("automata pruning is formalized only for fullmatch mode")
        if automata_stats.unsupported_reasons:
            if status is CompletenessStatus.COMPLETE_WITHIN_BOUNDS:
                status = CompletenessStatus.INCOMPLETE_DUE_TO_UNSUPPORTED_AUTOMATA
            reasons.extend(
                f"unsupported automata construct: {reason}"
                for reason in automata_stats.unsupported_reasons
            )

        scope.status = status
        scope.incomplete_reasons = reasons
        scope.timed_out = timed_out
        scope.limits_hit = limits_hit
        scope.raw_regex_candidate_count = raw_regex_count
        scope.unsupported_automata_reasons = list(automata_stats.unsupported_reasons)
        scope.automata_alphabet_size = automata_stats.alphabet_size
        scope.complete_within_bounds = not reasons

    def _candidate_feature(self, option: HoleOption) -> CandidateFeatureVector:
        return CandidateFeatureVector(
            regex=option.regex,
            strategy=option.strategy,
            confidence=option.confidence,
            source_id=option.source_id,
            length=len(option.regex),
            score_breakdown=self._candidate_score_breakdown(option),
            constraint_probe_matches=option.constraint_probe_matches,
            constraint_probe_rejections=option.constraint_probe_rejections,
        )

    def _embedding_safe_components(
        self,
        sketch: Sketch,
        hole_id: str,
        components: list[RegexComponent],
    ) -> list[RegexComponent]:
        if _hole_is_entire_sketch(sketch, hole_id):
            return components

        normalized: list[RegexComponent] = []
        for component in components:
            stripped = _strip_outer_anchors(component.regex)
            if stripped == component.regex or not stripped:
                normalized.append(component)
                continue
            metadata = dict(component.metadata)
            metadata["embedding_anchor_stripped_from"] = component.regex
            normalized.append(
                component.model_copy(
                    update={
                        "regex": stripped,
                        "metadata": metadata,
                    }
                )
            )
        return normalized

    def _embedding_safe_options(
        self,
        sketch: Sketch,
        hole_id: str,
        options: list[HoleOption],
    ) -> list[HoleOption]:
        if _hole_is_entire_sketch(sketch, hole_id):
            return _deduplicate_options(options)

        normalized: list[HoleOption] = []
        for option in options:
            stripped = _strip_outer_anchors(option.regex)
            if stripped == option.regex:
                normalized.append(option)
                continue
            if not stripped:
                continue
            normalized.append(
                HoleOption(
                    regex=stripped,
                    strategy=option.strategy,
                    confidence=option.confidence,
                    source_id=option.source_id,
                    semantic_fit_score=option.semantic_fit_score,
                    ast_complexity_cost=option.ast_complexity_cost,
                    provenance={
                        **option.provenance,
                        "embedding_anchor_stripped_from": option.regex,
                    },
                    provenance_steps=option.provenance_steps
                    + (
                        ProvenanceStep(
                            strategy=option.strategy,
                            operation="strip_embedding_anchors",
                            inputs=[option.regex],
                            output=stripped,
                            source_ids=[option.source_id] if option.source_id else [],
                        ),
                    ),
                    behavior_signature=option.behavior_signature,
                    constraint_probe_matches=option.constraint_probe_matches,
                    constraint_probe_rejections=option.constraint_probe_rejections,
                )
            )
        return _deduplicate_options(normalized)

    def _ordered_holes(
        self,
        sketch: Sketch,
        decomposition: DecompositionResult,
        options_by_hole: dict[str, list[HoleOption]],
    ) -> list[Hole]:
        tuple_participation: dict[str, int] = {}
        for constraint in decomposition.tuple_negative_constraints:
            for hole_id in constraint.hole_values:
                tuple_participation[hole_id] = tuple_participation.get(hole_id, 0) + 1

        def key(hole: Hole) -> tuple[int, int, int, int]:
            hole_examples = decomposition.by_hole.get(hole.identifier, HoleExampleSet())
            hard_count = len(hole_examples.hard.positive) + len(hole_examples.hard.negative)
            return (
                -hard_count,
                -tuple_participation.get(hole.identifier, 0),
                len(options_by_hole.get(hole.identifier, [])),
                hole.start,
            )

        return sorted(sketch.holes, key=key)

    def _violates_tuple_negative_constraints(
        self,
        assignments: dict[str, str],
        constraints: list[TupleNegativeConstraint],
    ) -> bool:
        for constraint in constraints:
            if any(hole_id not in assignments for hole_id in constraint.hole_values):
                continue
            try:
                if all(
                    all(
                        regex_matches(assignments[hole_id], value, MatchMode.FULLMATCH)
                        for value in values
                    )
                    for hole_id, values in constraint.hole_values.items()
                ):
                    return True
            except RegexValidationError:
                return True
        return False

    def _violates_repeated_hole_constraints(
        self,
        assignments: dict[str, str],
        constraints: list[RepeatedHoleConstraint],
    ) -> bool:
        for constraint in constraints:
            pattern = assignments.get(constraint.hole_id)
            if pattern is None:
                continue
            try:
                if any(
                    not regex_matches(pattern, value, MatchMode.FULLMATCH)
                    for value in constraint.values
                ):
                    return True
            except RegexValidationError:
                return True
        return False

    def _log_outcome(
        self,
        sketch_source: str,
        spec: SynthesisSpec,
        result: SynthesisResult,
    ) -> None:
        if not self._config.synthesis.outcome_logging.enabled:
            return
        path = resolve_project_path(self._config.synthesis.outcome_logging.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "sketch": sketch_source,
            "spec": spec.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _deduplicate_options(options: list[HoleOption]) -> list[HoleOption]:
    deduplicated: list[HoleOption] = []
    seen: set[str] = set()
    for option in options:
        if option.regex in seen:
            continue
        seen.add(option.regex)
        deduplicated.append(option)
    return deduplicated


def _option_from_search_option(option: SearchOption) -> HoleOption:
    return HoleOption(
        regex=option.regex,
        strategy=option.strategy,
        confidence=option.confidence,
        source_id=option.source_id,
        semantic_fit_score=option.semantic_fit_score,
        ast_complexity_cost=option.ast_complexity_cost,
        provenance=option.provenance,
        provenance_steps=option.provenance_steps,
        behavior_signature=option.behavior_signature,
        constraint_probe_matches=option.constraint_probe_matches,
        constraint_probe_rejections=option.constraint_probe_rejections,
    )


def _dump_provenance_steps(
    steps_by_hole: dict[str, tuple[ProvenanceStep, ...]] | None,
) -> dict[str, list[ProvenanceStep]]:
    if steps_by_hole is None:
        return {}
    return {
        hole_id: list(steps)
        for hole_id, steps in steps_by_hole.items()
    }


def _normalize_spec(spec_or_examples: SynthesisSpec | Examples) -> SynthesisSpec:
    if isinstance(spec_or_examples, SynthesisSpec):
        return spec_or_examples
    return SynthesisSpec(global_examples=spec_or_examples)


def _negative_probes_for_hole(
    decomposition: DecompositionResult,
    hole_id: str,
    limit: int,
) -> list[str]:
    probes: list[str] = []
    for constraint in decomposition.tuple_negative_constraints:
        for value in constraint.hole_values.get(hole_id, []):
            if value not in probes:
                probes.append(value)
            if len(probes) >= limit:
                return probes
    return probes


def _hole_is_entire_sketch(sketch: Sketch, hole_id: str) -> bool:
    return (
        isinstance(sketch.root, SketchHole)
        and sketch.root.hole.identifier == hole_id
    )


def _strip_outer_anchors(pattern: str) -> str:
    stripped = pattern
    while True:
        if stripped.startswith("^"):
            stripped = stripped[1:]
        elif stripped.startswith(r"\A"):
            stripped = stripped[2:]
        else:
            break

    while True:
        if stripped.endswith(r"\Z"):
            stripped = stripped[:-2]
        elif stripped.endswith("$") and not _is_escaped(stripped, len(stripped) - 1):
            stripped = stripped[:-1]
        else:
            break
    return stripped


def _is_escaped(value: str, index: int) -> bool:
    slash_count = 0
    cursor = index - 1
    while cursor >= 0 and value[cursor] == "\\":
        slash_count += 1
        cursor -= 1
    return slash_count % 2 == 1


def _json_cache_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
