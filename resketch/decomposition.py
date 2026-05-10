from __future__ import annotations

from dataclasses import dataclass

from resketch.config import AppConfig
from resketch.models import (
    AmbiguityGroup,
    ConstraintSet,
    DecompositionMode,
    DecompositionResult,
    DecompositionStats,
    EvidenceKind,
    Examples,
    Hole,
    HoleEvidence,
    HoleExampleSet,
    MatchMode,
    RepeatedHoleConstraint,
    TupleNegativeConstraint,
)
from resketch.regex_engine import RegexValidationError, regex_matches
from resketch.sketch import (
    Sketch,
    SketchAlt,
    SketchAnchor,
    SketchCategory,
    SketchCharClass,
    SketchConcat,
    SketchDot,
    SketchHole,
    SketchLiteral,
    SketchNode,
    SketchRepeat,
    render_sketch_node,
)


@dataclass(frozen=True)
class CaptureAssignment:
    captures: dict[str, tuple[str, ...]]
    source_example: str


@dataclass(frozen=True)
class MatchState:
    position: int
    captures: dict[str, tuple[str, ...]]


@dataclass
class MatchContext:
    config: AppConfig
    stats: DecompositionStats
    hole_patterns: dict[str, str] | None = None


def decompose_examples(
    config: AppConfig,
    sketch: Sketch,
    global_examples: Examples,
    explicit_hole_examples: dict[str, Examples] | None = None,
) -> DecompositionResult:
    by_hole = {hole.identifier: HoleExampleSet() for hole in sketch.holes}
    explicit = explicit_hole_examples or {}
    mode = _active_mode(config)
    stats = DecompositionStats(global_positive_count=len(global_examples.positive))
    diagnostics: list[str] = []
    unmatched_positive: list[str] = []
    ambiguity_groups: list[AmbiguityGroup] = []
    repeated_hole_constraints: list[RepeatedHoleConstraint] = []

    if mode is DecompositionMode.OFF:
        return DecompositionResult(
            by_hole=by_hole,
            stats=stats,
            diagnostics=["decomposition disabled"],
        )

    if mode is not DecompositionMode.EXPLICIT_ONLY:
        _infer_positive_evidence(
            config=config,
            sketch=sketch,
            examples=global_examples.positive,
            mode=mode,
            by_hole=by_hole,
            stats=stats,
            unmatched_positive=unmatched_positive,
            ambiguity_groups=ambiguity_groups,
            repeated_hole_constraints=repeated_hole_constraints,
        )
        tuple_negative_constraints = _infer_negative_constraints(
            config=config,
            sketch=sketch,
            examples=global_examples.negative,
            by_hole=by_hole,
            stats=stats,
        )
    else:
        tuple_negative_constraints = []

    _merge_explicit_examples(by_hole, explicit, stats)
    diagnostics.extend(_unknown_explicit_holes(sketch, explicit))
    if stats.truncated_assignment_count:
        diagnostics.append(
            f"decomposition assignment enumeration truncated "
            f"{stats.truncated_assignment_count} time(s)"
        )
    if stats.repeat_unroll_truncation_count:
        diagnostics.append(
            f"repeat unrolling truncated {stats.repeat_unroll_truncation_count} time(s)"
        )
    _refresh_derived_examples(by_hole, stats)
    stats.recompute_rates()

    return DecompositionResult(
        by_hole=by_hole,
        tuple_negative_constraints=tuple_negative_constraints,
        ambiguity_groups=ambiguity_groups,
        repeated_hole_constraints=repeated_hole_constraints,
        constraint_set=ConstraintSet(
            tuple_negative_constraints=tuple_negative_constraints,
            ambiguity_groups=ambiguity_groups,
            repeated_hole_constraints=repeated_hole_constraints,
        ),
        unmatched_positive=unmatched_positive,
        stats=stats,
        diagnostics=diagnostics,
    )


def capture_witnesses(
    config: AppConfig,
    sketch: Sketch,
    assignments: dict[str, str],
    examples: Examples,
) -> dict[str, list[str]]:
    stats = DecompositionStats()
    context = MatchContext(config=config, stats=stats, hole_patterns=assignments)
    witnesses: dict[str, list[str]] = {hole.identifier: [] for hole in sketch.holes}
    for example in examples.positive:
        for assignment in _match_root(context, sketch.root, example):
            for hole in sketch.holes:
                for value in assignment.captures.get(hole.identifier, ()):
                    _append_unique(witnesses[hole.identifier], value)
    return witnesses


def _infer_positive_evidence(
    config: AppConfig,
    sketch: Sketch,
    examples: list[str],
    mode: DecompositionMode,
    by_hole: dict[str, HoleExampleSet],
    stats: DecompositionStats,
    unmatched_positive: list[str],
    ambiguity_groups: list[AmbiguityGroup],
    repeated_hole_constraints: list[RepeatedHoleConstraint],
) -> None:
    context = MatchContext(config=config, stats=stats)
    for example in examples:
        assignments = _match_root(context, sketch.root, example)
        if not assignments:
            stats.unmatched_positive_count += 1
            unmatched_positive.append(example)
            continue

        stats.matched_positive_count += 1
        stats.examples_with_assignments += 1
        stats.total_assignments_considered += len(assignments)

        is_ambiguous = _example_is_ambiguous(sketch, assignments, config)
        if is_ambiguous:
            stats.ambiguous_example_count += 1
            ambiguity_group = _ambiguity_group(config, example, assignments)
            if ambiguity_group.choices:
                ambiguity_groups.append(ambiguity_group)
                stats.ambiguity_group_count += 1
        else:
            _record_repeated_hole_constraints(
                config,
                assignments,
                repeated_hole_constraints,
                stats,
            )
        for hole in sketch.holes:
            _classify_hole_captures(config, mode, by_hole[hole.identifier], hole, assignments)


def _infer_negative_constraints(
    config: AppConfig,
    sketch: Sketch,
    examples: list[str],
    by_hole: dict[str, HoleExampleSet],
    stats: DecompositionStats,
) -> list[TupleNegativeConstraint]:
    context = MatchContext(config=config, stats=stats)
    constraints: list[TupleNegativeConstraint] = []
    for example in examples:
        assignments = _match_root(context, sketch.root, example)
        if len(assignments) != 1:
            continue
        hole_values = _assignment_hole_values(config, assignments[0])
        if not hole_values:
            continue
        if len(hole_values) == 1:
            hole_id, values = next(iter(hole_values.items()))
            for value in values:
                stats.inferred_negative_count += 1
                _add_evidence(
                    by_hole[hole_id],
                    HoleEvidence(
                        value=value,
                        source_example=example,
                        kind=EvidenceKind.INFERRED_NEGATIVE,
                        confidence=1.0,
                        policies=["ast"],
                        reason="unique negative AST assignment",
                    ),
                )
            continue
        constraints.append(
            TupleNegativeConstraint(
                source_example=example,
                hole_values=hole_values,
                reason="unique multi-hole negative AST assignment",
                confidence=1.0,
            )
        )
        stats.tuple_negative_constraint_count += 1
    return constraints


def _match_root(
    context: MatchContext,
    node: SketchNode,
    example: str,
) -> list[CaptureAssignment]:
    starts = [0]
    if context.config.matching.mode is MatchMode.SEARCH:
        starts = list(range(len(example) + 1))

    assignments: list[CaptureAssignment] = []
    for start in starts:
        states = _match_node(
            context,
            node,
            example,
            MatchState(position=start, captures={}),
        )
        for state in states:
            if (
                context.config.matching.mode is MatchMode.FULLMATCH
                and state.position != len(example)
            ):
                continue
            assignments.append(CaptureAssignment(captures=state.captures, source_example=example))
            if len(assignments) >= context.config.decomposition.max_assignments_per_example:
                return _deduplicate_assignments(assignments)
    return _deduplicate_assignments(assignments)


def _match_node(
    context: MatchContext,
    node: SketchNode,
    example: str,
    state: MatchState,
) -> list[MatchState]:
    if isinstance(node, SketchLiteral):
        if example.startswith(node.value, state.position):
            return [
                MatchState(
                    position=state.position + len(node.value),
                    captures=state.captures,
                )
            ]
        return []
    if isinstance(node, SketchDot):
        if state.position < len(example) and example[state.position] != "\n":
            return [MatchState(position=state.position + 1, captures=state.captures)]
        return []
    if isinstance(node, SketchCategory | SketchCharClass):
        return _match_single_char_regex(context, node, example, state)
    if isinstance(node, SketchAnchor):
        return _match_anchor(node, example, state)
    if isinstance(node, SketchHole):
        return _match_hole(context, node.hole, example, state)
    if isinstance(node, SketchConcat):
        return _match_concat(context, node.parts, example, state)
    if isinstance(node, SketchAlt):
        return _match_alt(context, node.options, example, state)
    if isinstance(node, SketchRepeat):
        return _match_repeat(context, node, example, state)


def _match_single_char_regex(
    context: MatchContext,
    node: SketchCategory | SketchCharClass,
    example: str,
    state: MatchState,
) -> list[MatchState]:
    if state.position >= len(example):
        return []
    pattern = render_sketch_node(node, {})
    try:
        if regex_matches(pattern, example[state.position], MatchMode.FULLMATCH):
            return [MatchState(position=state.position + 1, captures=state.captures)]
    except RegexValidationError:
        context.stats.truncated_assignment_count += 1
    return []


def _match_anchor(
    node: SketchAnchor,
    example: str,
    state: MatchState,
) -> list[MatchState]:
    if node.pattern in {"^", r"\A"} and state.position == 0:
        return [state]
    if node.pattern in {"$", r"\Z"} and state.position == len(example):
        return [state]
    return []


def _match_hole(
    context: MatchContext,
    hole: Hole,
    example: str,
    state: MatchState,
) -> list[MatchState]:
    min_end = (
        state.position
        if context.config.decomposition.allow_empty_hole_examples
        else state.position + 1
    )
    max_end = min(
        len(example),
        state.position + context.config.decomposition.max_hole_capture_length,
    )
    results: list[MatchState] = []
    pattern = context.hole_patterns.get(hole.identifier) if context.hole_patterns else None
    for end in range(min_end, max_end + 1):
        value = example[state.position:end]
        if pattern is not None:
            try:
                if not regex_matches(pattern, value, MatchMode.FULLMATCH):
                    continue
            except RegexValidationError:
                return []
        next_captures = dict(state.captures)
        next_captures[hole.identifier] = (*next_captures.get(hole.identifier, ()), value)
        _append_state(context, results, MatchState(position=end, captures=next_captures))
    return results


def _match_concat(
    context: MatchContext,
    parts: tuple[SketchNode, ...],
    example: str,
    state: MatchState,
) -> list[MatchState]:
    states = [state]
    for part in parts:
        next_states: list[MatchState] = []
        for current in states:
            for matched in _match_node(context, part, example, current):
                _append_state(context, next_states, matched)
        states = next_states
        if not states:
            return []
    return states


def _match_alt(
    context: MatchContext,
    options: tuple[SketchNode, ...],
    example: str,
    state: MatchState,
) -> list[MatchState]:
    states: list[MatchState] = []
    for option in options:
        for matched in _match_node(context, option, example, state):
            _append_state(context, states, matched)
    return states


def _match_repeat(
    context: MatchContext,
    node: SketchRepeat,
    example: str,
    state: MatchState,
) -> list[MatchState]:
    upper = _bounded_repeat_upper(context, node)
    if upper < node.min_repeat:
        return []

    results: list[MatchState] = []
    frontier = [state]
    if node.min_repeat == 0:
        _append_state(context, results, state)

    for repeat_count in range(1, upper + 1):
        next_frontier: list[MatchState] = []
        for current in frontier:
            for matched in _match_node(context, node.child, example, current):
                _append_state(context, next_frontier, matched)
        frontier = next_frontier
        if not frontier:
            break
        if repeat_count >= node.min_repeat:
            for matched in frontier:
                _append_state(context, results, matched)
    return results


def _bounded_repeat_upper(context: MatchContext, node: SketchRepeat) -> int:
    configured_limit = context.config.decomposition.max_repeat_unroll
    natural_upper = node.max_repeat if node.max_repeat is not None else configured_limit
    if natural_upper > configured_limit:
        context.stats.repeat_unroll_truncation_count += 1
        return configured_limit
    if node.max_repeat is None:
        context.stats.repeat_unroll_truncation_count += 1
    return natural_upper


def _append_state(
    context: MatchContext,
    states: list[MatchState],
    state: MatchState,
) -> None:
    if len(states) < context.config.decomposition.max_assignments_per_example:
        states.append(state)
    else:
        context.stats.truncated_assignment_count += 1


def _classify_hole_captures(
    config: AppConfig,
    mode: DecompositionMode,
    hole_examples: HoleExampleSet,
    hole: Hole,
    assignments: list[CaptureAssignment],
) -> None:
    capture_tuples = [
        _filtered_capture_values(config, assignment, hole.identifier)
        for assignment in assignments
    ]
    non_empty_capture_tuples = [values for values in capture_tuples if values]
    if not non_empty_capture_tuples:
        hole_examples.diagnostics.append(
            f"empty inferred capture ignored for hole {hole.identifier}"
        )
        return

    unique_capture_tuples = set(capture_tuples)
    if len(unique_capture_tuples) == 1:
        source_example = assignments[0].source_example if assignments else ""
        values = next(iter(unique_capture_tuples))
        for value in _unique_in_order(values):
            _add_evidence(
                hole_examples,
                HoleEvidence(
                    value=value,
                    source_example=source_example or value,
                    kind=EvidenceKind.HARD_POSITIVE,
                    confidence=1.0,
                    policies=["ast"],
                    reason="all AST assignments agree",
                ),
            )
        return

    hole_examples.diagnostics.append(f"ambiguous capture for hole {hole.identifier}")
    if mode is not DecompositionMode.HARD_AND_SOFT:
        return
    source_example = assignments[0].source_example if assignments else ""
    confidence = 1.0 / max(2, len(unique_capture_tuples))
    for value in _all_capture_values_in_order(config, assignments, hole.identifier)[
        : config.decomposition.max_soft_examples_per_hole
    ]:
        _add_evidence(
            hole_examples,
            HoleEvidence(
                value=value,
                source_example=source_example or value,
                kind=EvidenceKind.SOFT_POSITIVE,
                confidence=confidence,
                policies=["ast"],
                reason=f"multiple AST assignments possible ({len(unique_capture_tuples)} choices)",
            ),
        )


def _filtered_capture_values(
    config: AppConfig,
    assignment: CaptureAssignment,
    hole_id: str,
) -> tuple[str, ...]:
    return tuple(
        value
        for value in assignment.captures.get(hole_id, ())
        if value or config.decomposition.allow_empty_hole_examples
    )


def _all_capture_values_in_order(
    config: AppConfig,
    assignments: list[CaptureAssignment],
    hole_id: str,
) -> list[str]:
    values: list[str] = []
    for assignment in assignments:
        for value in _filtered_capture_values(config, assignment, hole_id):
            _append_unique(values, value)
    return values


def _assignment_hole_values(
    config: AppConfig,
    assignment: CaptureAssignment,
) -> dict[str, list[str]]:
    values_by_hole: dict[str, list[str]] = {}
    for hole_id in assignment.captures:
        values = _filtered_capture_values(config, assignment, hole_id)
        if values:
            values_by_hole[hole_id] = _unique_in_order(values)
    return values_by_hole


def _example_is_ambiguous(
    sketch: Sketch,
    assignments: list[CaptureAssignment],
    config: AppConfig,
) -> bool:
    for hole in sketch.holes:
        values = {
            _filtered_capture_values(config, assignment, hole.identifier)
            for assignment in assignments
        }
        if any(values) and len(values) > 1:
            return True
    return False


def _ambiguity_group(
    config: AppConfig,
    source_example: str,
    assignments: list[CaptureAssignment],
) -> AmbiguityGroup:
    choices: list[dict[str, list[str]]] = []
    seen: set[tuple[tuple[str, tuple[str, ...]], ...]] = set()
    for assignment in assignments:
        values_by_hole = _assignment_hole_values(config, assignment)
        if not values_by_hole:
            continue
        key = tuple(
            sorted((hole_id, tuple(values)) for hole_id, values in values_by_hole.items())
        )
        if key in seen:
            continue
        seen.add(key)
        choices.append(values_by_hole)
    return AmbiguityGroup(
        source_example=source_example,
        choices=choices,
        reason=f"{len(choices)} distinct AST capture assignments",
        confidence=1.0 / max(1, len(choices)),
    )


def _record_repeated_hole_constraints(
    config: AppConfig,
    assignments: list[CaptureAssignment],
    constraints: list[RepeatedHoleConstraint],
    stats: DecompositionStats,
) -> None:
    for assignment in assignments:
        for hole_id in assignment.captures:
            filtered = _filtered_capture_values(config, assignment, hole_id)
            if len(filtered) <= 1:
                continue
            constraint = RepeatedHoleConstraint(
                source_example=assignment.source_example,
                hole_id=hole_id,
                occurrence_count=len(filtered),
                values=_unique_in_order(filtered),
                reason="same sketch hole captured multiple AST occurrences",
                confidence=1.0,
            )
            if _add_repeated_hole_constraint(constraints, constraint):
                stats.repeated_hole_constraint_count += 1


def _add_repeated_hole_constraint(
    constraints: list[RepeatedHoleConstraint],
    constraint: RepeatedHoleConstraint,
) -> bool:
    key = (
        constraint.source_example,
        constraint.hole_id,
        constraint.occurrence_count,
        tuple(constraint.values),
    )
    for existing in constraints:
        existing_key = (
            existing.source_example,
            existing.hole_id,
            existing.occurrence_count,
            tuple(existing.values),
        )
        if existing_key == key:
            return False
    constraints.append(constraint)
    return True


def _merge_explicit_examples(
    by_hole: dict[str, HoleExampleSet],
    explicit: dict[str, Examples],
    stats: DecompositionStats,
) -> None:
    for hole_id, examples in explicit.items():
        hole_examples = by_hole.setdefault(hole_id, HoleExampleSet())
        for positive in examples.positive:
            stats.explicit_evidence_count += 1
            _add_evidence(
                hole_examples,
                HoleEvidence(
                    value=positive,
                    source_example=positive,
                    kind=EvidenceKind.EXPLICIT_POSITIVE,
                    confidence=1.0,
                    policies=["explicit"],
                    reason="user-provided hole positive",
                ),
            )
        for negative in examples.negative:
            stats.explicit_evidence_count += 1
            _add_evidence(
                hole_examples,
                HoleEvidence(
                    value=negative,
                    source_example=negative,
                    kind=EvidenceKind.EXPLICIT_NEGATIVE,
                    confidence=1.0,
                    policies=["explicit"],
                    reason="user-provided hole negative",
                ),
            )


def _refresh_derived_examples(
    by_hole: dict[str, HoleExampleSet],
    stats: DecompositionStats,
) -> None:
    stats.hard_evidence_count = 0
    stats.soft_evidence_count = 0
    stats.holes_with_hard_examples = 0
    stats.holes_with_only_soft_examples = 0

    for hole_examples in by_hole.values():
        hole_examples.hard = Examples()
        hole_examples.soft_positive = []
        for evidence in hole_examples.evidence:
            if evidence.kind in {
                EvidenceKind.HARD_POSITIVE,
                EvidenceKind.EXPLICIT_POSITIVE,
            }:
                _append_unique(hole_examples.hard.positive, evidence.value)
            elif evidence.kind in {
                EvidenceKind.EXPLICIT_NEGATIVE,
                EvidenceKind.INFERRED_NEGATIVE,
            }:
                _append_unique(hole_examples.hard.negative, evidence.value)
            elif evidence.kind is EvidenceKind.SOFT_POSITIVE:
                _append_unique(hole_examples.soft_positive, evidence.value)

        stats.hard_evidence_count += sum(
            1 for evidence in hole_examples.evidence if evidence.kind is EvidenceKind.HARD_POSITIVE
        )
        stats.soft_evidence_count += sum(
            1 for evidence in hole_examples.evidence if evidence.kind is EvidenceKind.SOFT_POSITIVE
        )

        hard_count = len(hole_examples.hard.positive) + len(hole_examples.hard.negative)
        soft_count = len(hole_examples.soft_positive)
        if hard_count:
            stats.holes_with_hard_examples += 1
            hole_examples.confidence = 1.0
        elif soft_count:
            stats.holes_with_only_soft_examples += 1
            hole_examples.confidence = 0.5
        else:
            hole_examples.confidence = 0.0


def _unknown_explicit_holes(sketch: Sketch, explicit: dict[str, Examples]) -> list[str]:
    known_holes = {hole.identifier for hole in sketch.holes}
    return [
        f"explicit examples reference unknown hole {hole_id!r}"
        for hole_id in explicit
        if hole_id not in known_holes
    ]


def _active_mode(config: AppConfig) -> DecompositionMode:
    if not config.decomposition.enabled:
        return DecompositionMode.OFF
    return config.decomposition.mode


def _deduplicate_assignments(assignments: list[CaptureAssignment]) -> list[CaptureAssignment]:
    deduplicated: list[CaptureAssignment] = []
    seen: set[tuple[tuple[str, tuple[str, ...]], ...]] = set()
    for assignment in assignments:
        key = tuple(sorted(assignment.captures.items()))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(assignment)
    return deduplicated


def _add_evidence(hole_examples: HoleExampleSet, evidence: HoleEvidence) -> None:
    key = (evidence.kind, evidence.value, evidence.source_example, tuple(evidence.policies))
    for existing in hole_examples.evidence:
        existing_key = (
            existing.kind,
            existing.value,
            existing.source_example,
            tuple(existing.policies),
        )
        if existing_key == key:
            return
    hole_examples.evidence.append(evidence)


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _unique_in_order(values: tuple[str, ...]) -> list[str]:
    unique: list[str] = []
    for value in values:
        _append_unique(unique, value)
    return unique
