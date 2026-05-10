from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from resketch.config import AppConfig
from resketch.models import (
    HoleExampleSet,
    MatchMode,
    ProvenanceStep,
    RegexComponent,
    SearchStats,
    SynthesisStrategy,
)
from resketch.regex_ast import (
    Alt,
    Anchor,
    Category,
    CharClass,
    Concat,
    Literal,
    RawRegex,
    RegexNode,
    Repeat,
    extract_subnodes,
    make_alt,
    make_concat,
    node_size,
    normalize_node,
    parse_regex_to_ast,
    render_regex,
)
from resketch.regex_engine import RegexValidationError, evaluate_examples, regex_matches
from resketch.scoring import candidate_score_breakdown, node_syntactic_complexity_cost


@dataclass(frozen=True)
class SearchOption:
    regex: str
    node: RegexNode
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
class SearchResult:
    options: list[SearchOption]
    stats: SearchStats


@dataclass(frozen=True)
class _CandidateSeed:
    node: RegexNode
    strategy: SynthesisStrategy
    confidence: float | None
    source_id: str | None
    provenance: dict[str, Any]
    order: int


class RegexSearchEngine:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def search(
        self,
        components: list[RegexComponent],
        hole_examples: HoleExampleSet,
        *,
        negative_probes: list[str] | None = None,
    ) -> SearchResult:
        start = time.perf_counter()
        stats = SearchStats()
        probes = list(dict.fromkeys(negative_probes or []))[
            : self._config.synthesis.search.max_constraint_probes_per_hole
        ]
        stats.constraint_probe_count = len(probes)
        context = _SearchContext(
            config=self._config,
            hole_examples=hole_examples,
            negative_probes=probes,
            stats=stats,
            start_time=start,
            best_by_signature={},
            best_by_canonical={},
        )

        parsed_components = self._parse_components(components, context)
        base_nodes = self._base_nodes(parsed_components, hole_examples)

        seeds: list[_CandidateSeed] = []
        for strategy in self._config.synthesis.strategy_order:
            if len(seeds) >= self._config.synthesis.search.max_generated_candidates:
                stats.bounds_hit = True
                break
            if strategy is SynthesisStrategy.DIRECT_REUSE:
                seeds.extend(self._direct_seeds(parsed_components, len(seeds)))
            elif strategy is SynthesisStrategy.COMPONENT_GENERALIZATION:
                seeds.extend(self._generalization_seeds(parsed_components, len(seeds)))
            elif strategy is SynthesisStrategy.FRAGMENT_RECOMBINATION:
                seeds.extend(self._recombination_seeds(base_nodes, len(seeds)))
            elif strategy is SynthesisStrategy.PARAMETERIZED_REFINEMENT:
                seeds.extend(
                    self._refinement_seeds(parsed_components, base_nodes, context, len(seeds))
                )

        stats.frontier_size = len(seeds)
        for seed in sorted(seeds, key=self._seed_priority):
            if context.should_stop():
                break
            context.add_candidate(
                node=seed.node,
                strategy=seed.strategy,
                confidence=seed.confidence,
                source_id=seed.source_id,
                provenance={**seed.provenance, "frontier_order": seed.order},
            )

        stats.elapsed_seconds = time.perf_counter() - start
        options = sorted(
            context.best_by_signature.values(),
            key=lambda option: (
                _candidate_ranking_score(option, self._config),
                -len(option.regex),
            ),
            reverse=True,
        )
        selected_options = _select_options(
            options,
            self._config.synthesis.max_hole_options,
            len(probes),
        )
        if len(selected_options) < len(options):
            stats.bounds_hit = True
        return SearchResult(
            options=selected_options,
            stats=stats,
        )

    def _seed_priority(self, seed: _CandidateSeed) -> tuple[float, float, int, str]:
        try:
            node = normalize_node(seed.node)
            rendered = render_regex(node)
            ast_complexity_cost = node_syntactic_complexity_cost(
                node,
                self._config.synthesis.scoring.syntactic_complexity_cost,
            )
        except RegexValidationError:
            rendered = ""
            ast_complexity_cost = float("inf")
        return (
            ast_complexity_cost
            + self._config.synthesis.scoring.strategy_penalty[seed.strategy],
            len(rendered),
            seed.order,
            rendered,
        )

    def _parse_components(
        self,
        components: list[RegexComponent],
        context: _SearchContext,
    ) -> list[tuple[RegexComponent, RegexNode]]:
        parsed: list[tuple[RegexComponent, RegexNode]] = []
        for component in components:
            if len(component.regex) > self._config.synthesis.max_candidate_length:
                continue
            try:
                node = parse_regex_to_ast(
                    component.regex,
                    allow_raw_regex=self._config.synthesis.search.allow_raw_regex,
                )
            except RegexValidationError:
                context.stats.pruned_invalid_regexes += 1
                continue
            if isinstance(node, RawRegex):
                context.stats.raw_regex_candidates += 1
            parsed.append((component, node))
        return parsed

    def _base_nodes(
        self,
        parsed_components: list[tuple[RegexComponent, RegexNode]],
        hole_examples: HoleExampleSet,
    ) -> list[RegexNode]:
        nodes: list[RegexNode] = []
        for _, node in parsed_components:
            nodes.extend(extract_subnodes(node))
        nodes.extend(
            self._example_primitives(
                hole_examples,
                include_full_literals=not parsed_components,
            )
        )

        deduped: dict[str, RegexNode] = {}
        for node in nodes:
            if node_size(node) <= self._config.synthesis.search.max_ast_size:
                deduped.setdefault(render_regex(node), node)
        return list(deduped.values())

    def _direct_seeds(
        self,
        parsed_components: list[tuple[RegexComponent, RegexNode]],
        start_order: int,
    ) -> list[_CandidateSeed]:
        seeds: list[_CandidateSeed] = []
        for component, node in parsed_components:
            seeds.append(
                _CandidateSeed(
                    node=node,
                    strategy=SynthesisStrategy.DIRECT_REUSE,
                    confidence=component.confidence,
                    source_id=component.source_id,
                    provenance=_component_provenance(component),
                    order=start_order + len(seeds),
                )
            )
        return seeds

    def _generalization_seeds(
        self,
        parsed_components: list[tuple[RegexComponent, RegexNode]],
        start_order: int,
    ) -> list[_CandidateSeed]:
        seeds: list[_CandidateSeed] = []
        pair_count = 0
        for left_index, (left_component, left_node) in enumerate(parsed_components):
            for right_component, right_node in parsed_components[left_index + 1 :]:
                if pair_count >= self._config.synthesis.max_fragment_pairs:
                    return seeds
                pair_count += 1
                generalized = _anti_unify(left_node, right_node)
                if generalized is None:
                    continue
                if node_size(generalized) > self._config.synthesis.search.max_ast_size:
                    continue
                seeds.append(
                    _CandidateSeed(
                        node=generalized,
                        strategy=SynthesisStrategy.COMPONENT_GENERALIZATION,
                        confidence=_average_confidence(
                            left_component.confidence,
                            right_component.confidence,
                        ),
                        source_id=_join_source_ids(
                            left_component.source_id,
                            right_component.source_id,
                        ),
                        provenance={
                            "operation": "anti_unify",
                            "source_regexes": [
                                _source_regex(left_component),
                                _source_regex(right_component),
                            ],
                            "normalized_source_regexes": [
                                left_component.regex,
                                right_component.regex,
                            ],
                            "source_ids": [
                                source_id
                                for source_id in [
                                    left_component.source_id,
                                    right_component.source_id,
                                ]
                                if source_id is not None
                            ],
                        },
                        order=start_order + len(seeds),
                    )
                )
        return seeds

    def _recombination_seeds(
        self,
        base_nodes: list[RegexNode],
        start_order: int,
    ) -> list[_CandidateSeed]:
        seeds: list[_CandidateSeed] = []
        constructors = set(self._config.synthesis.search.constructors)
        nodes_by_size: dict[int, list[RegexNode]] = defaultdict(list)
        for node in base_nodes:
            nodes_by_size[node_size(node)].append(node)
            seeds.append(
                _CandidateSeed(
                    node=node,
                    strategy=SynthesisStrategy.FRAGMENT_RECOMBINATION,
                    confidence=None,
                    source_id=None,
                    provenance={"operation": "fragment", "source_regexes": [render_regex(node)]},
                    order=start_order + len(seeds),
                )
            )

        for size in range(2, self._config.synthesis.search.max_ast_size + 1):
            generated_for_size: dict[str, RegexNode] = {}
            if "concat" in constructors or "alt" in constructors:
                for left_size in range(1, size):
                    right_size = size - left_size - 1
                    if right_size < 1:
                        continue
                    for left in nodes_by_size.get(left_size, []):
                        for right in nodes_by_size.get(right_size, []):
                            if "concat" in constructors:
                                node = make_concat(left, right)
                                generated_for_size.setdefault(render_regex(node), node)
                            if "alt" in constructors:
                                node = make_alt(left, right)
                                generated_for_size.setdefault(render_regex(node), node)
                            if (
                                len(generated_for_size)
                                >= self._config.synthesis.search.max_nodes_per_size
                            ):
                                break
                        if (
                            len(generated_for_size)
                            >= self._config.synthesis.search.max_nodes_per_size
                        ):
                            break
                    if (
                        len(generated_for_size)
                        >= self._config.synthesis.search.max_nodes_per_size
                    ):
                        break

            child_size = size - 1
            if child_size >= 1 and ("repeat" in constructors or "optional" in constructors):
                for child in nodes_by_size.get(child_size, []):
                    for (
                        min_repeat,
                        max_repeat,
                    ) in self._config.synthesis.search.allowed_repeat_bounds:
                        if "optional" not in constructors and (min_repeat, max_repeat) == (0, 1):
                            continue
                        if "repeat" not in constructors and (min_repeat, max_repeat) != (0, 1):
                            continue
                        node = Repeat(child=child, min_repeat=min_repeat, max_repeat=max_repeat)
                        generated_for_size.setdefault(render_regex(node), node)
                        if (
                            len(generated_for_size)
                            >= self._config.synthesis.search.max_nodes_per_size
                        ):
                            break

            nodes_by_size[size] = list(generated_for_size.values())
            for node in nodes_by_size[size]:
                seeds.append(
                    _CandidateSeed(
                        node=node,
                        strategy=SynthesisStrategy.FRAGMENT_RECOMBINATION,
                        confidence=None,
                        source_id=None,
                        provenance={"operation": "bottom_up", "size": size},
                        order=start_order + len(seeds),
                    )
                )
        return seeds

    def _refinement_seeds(
        self,
        parsed_components: list[tuple[RegexComponent, RegexNode]],
        base_nodes: list[RegexNode],
        context: _SearchContext,
        start_order: int,
    ) -> list[_CandidateSeed]:
        seeds: list[_CandidateSeed] = []
        for component, node in parsed_components:
            for refined in self._refine_node(node, context):
                seeds.append(
                    _CandidateSeed(
                        node=refined,
                        strategy=SynthesisStrategy.PARAMETERIZED_REFINEMENT,
                        confidence=component.confidence,
                        source_id=component.source_id,
                        provenance={
                            **_component_provenance(component),
                            "operation": "refine_component",
                        },
                        order=start_order + len(seeds),
                    )
                )
        for node in base_nodes:
            for refined in self._refine_node(node, context):
                seeds.append(
                    _CandidateSeed(
                        node=refined,
                        strategy=SynthesisStrategy.PARAMETERIZED_REFINEMENT,
                        confidence=None,
                        source_id=None,
                        provenance={"operation": "refine_fragment"},
                        order=start_order + len(seeds),
                    )
                )
        return seeds

    def _example_primitives(
        self,
        hole_examples: HoleExampleSet,
        *,
        include_full_literals: bool,
    ) -> list[RegexNode]:
        nodes: list[RegexNode] = []
        examples = [*hole_examples.hard.positive, *hole_examples.soft_positive]
        for example in examples[: self._config.synthesis.search.max_example_literals]:
            if (
                include_full_literals
                and example
                and len(example) <= self._config.synthesis.search.example_literal_max_length
            ):
                nodes.append(Literal(example))
            for char in example:
                if char in self._config.synthesis.search.separator_chars:
                    nodes.append(Literal(char))
        for pattern in self._config.synthesis.refinement.character_class_candidates:
            try:
                nodes.append(
                    parse_regex_to_ast(
                        pattern,
                        allow_raw_regex=self._config.synthesis.search.allow_raw_regex,
                    )
                )
            except RegexValidationError:
                continue
        return nodes

    def _add_direct_options(
        self,
        parsed_components: list[tuple[RegexComponent, RegexNode]],
        context: _SearchContext,
    ) -> None:
        for component, node in parsed_components:
            context.add_candidate(
                node=node,
                strategy=SynthesisStrategy.DIRECT_REUSE,
                confidence=component.confidence,
                source_id=component.source_id,
                provenance=_component_provenance(component),
            )

    def _add_generalizations(
        self,
        parsed_components: list[tuple[RegexComponent, RegexNode]],
        context: _SearchContext,
    ) -> None:
        pair_count = 0
        for left_index, (left_component, left_node) in enumerate(parsed_components):
            for right_component, right_node in parsed_components[left_index + 1 :]:
                if pair_count >= self._config.synthesis.max_fragment_pairs:
                    return
                pair_count += 1
                generalized = _anti_unify(left_node, right_node)
                if generalized is None:
                    continue
                if node_size(generalized) > self._config.synthesis.search.max_ast_size:
                    continue
                context.add_candidate(
                    node=generalized,
                    strategy=SynthesisStrategy.COMPONENT_GENERALIZATION,
                    confidence=_average_confidence(
                        left_component.confidence,
                        right_component.confidence,
                    ),
                    source_id=_join_source_ids(left_component.source_id, right_component.source_id),
                    provenance={
                        "operation": "anti_unify",
                        "source_regexes": [
                            _source_regex(left_component),
                            _source_regex(right_component),
                        ],
                        "normalized_source_regexes": [
                            left_component.regex,
                            right_component.regex,
                        ],
                        "source_ids": [
                            source_id
                            for source_id in [
                                left_component.source_id,
                                right_component.source_id,
                            ]
                            if source_id is not None
                        ],
                    },
                )

    def _enumerate_recombinations(
        self,
        base_nodes: list[RegexNode],
        context: _SearchContext,
    ) -> None:
        constructors = set(self._config.synthesis.search.constructors)
        nodes_by_size: dict[int, list[RegexNode]] = defaultdict(list)
        for node in base_nodes:
            nodes_by_size[node_size(node)].append(node)
            context.add_candidate(
                node=node,
                strategy=SynthesisStrategy.FRAGMENT_RECOMBINATION,
                confidence=None,
                source_id=None,
                provenance={"operation": "fragment", "source_regexes": [render_regex(node)]},
            )

        for size in range(2, self._config.synthesis.search.max_ast_size + 1):
            if context.should_stop():
                return
            generated_for_size: dict[str, RegexNode] = {}
            if "concat" in constructors or "alt" in constructors:
                for left_size in range(1, size):
                    right_size = size - left_size - 1
                    if right_size < 1:
                        continue
                    for left in nodes_by_size.get(left_size, []):
                        for right in nodes_by_size.get(right_size, []):
                            if "concat" in constructors:
                                generated_for_size.setdefault(
                                    render_regex(make_concat(left, right)),
                                    make_concat(left, right),
                                )
                            if "alt" in constructors:
                                generated_for_size.setdefault(
                                    render_regex(make_alt(left, right)),
                                    make_alt(left, right),
                                )
                            if (
                                len(generated_for_size)
                                >= self._config.synthesis.search.max_nodes_per_size
                            ):
                                break
                        if (
                            len(generated_for_size)
                            >= self._config.synthesis.search.max_nodes_per_size
                        ):
                            break
                    if (
                        len(generated_for_size)
                        >= self._config.synthesis.search.max_nodes_per_size
                    ):
                        break

            child_size = size - 1
            if child_size >= 1 and ("repeat" in constructors or "optional" in constructors):
                for child in nodes_by_size.get(child_size, []):
                    for (
                        min_repeat,
                        max_repeat,
                    ) in self._config.synthesis.search.allowed_repeat_bounds:
                        if "optional" not in constructors and (min_repeat, max_repeat) == (0, 1):
                            continue
                        if "repeat" not in constructors and (min_repeat, max_repeat) != (0, 1):
                            continue
                        node = Repeat(child=child, min_repeat=min_repeat, max_repeat=max_repeat)
                        generated_for_size.setdefault(render_regex(node), node)
                        if (
                            len(generated_for_size)
                            >= self._config.synthesis.search.max_nodes_per_size
                        ):
                            break

            nodes_by_size[size] = list(generated_for_size.values())
            for node in nodes_by_size[size]:
                context.add_candidate(
                    node=node,
                    strategy=SynthesisStrategy.FRAGMENT_RECOMBINATION,
                    confidence=None,
                    source_id=None,
                    provenance={"operation": "bottom_up", "size": size},
                )

    def _add_refinements(
        self,
        parsed_components: list[tuple[RegexComponent, RegexNode]],
        base_nodes: list[RegexNode],
        context: _SearchContext,
    ) -> None:
        for component, node in parsed_components:
            for refined in self._refine_node(node, context):
                context.add_candidate(
                    node=refined,
                    strategy=SynthesisStrategy.PARAMETERIZED_REFINEMENT,
                    confidence=component.confidence,
                    source_id=component.source_id,
                    provenance={
                        **_component_provenance(component),
                        "operation": "refine_component",
                    },
                )
        for node in base_nodes:
            for refined in self._refine_node(node, context):
                context.add_candidate(
                    node=refined,
                    strategy=SynthesisStrategy.PARAMETERIZED_REFINEMENT,
                    confidence=None,
                    source_id=None,
                    provenance={"operation": "refine_fragment"},
                )

    def _refine_node(self, node: RegexNode, context: _SearchContext) -> list[RegexNode]:
        refined: list[RegexNode] = []
        if isinstance(node, Repeat):
            for min_repeat, max_repeat in self._nearby_bounds(node):
                refined.append(
                    Repeat(child=node.child, min_repeat=min_repeat, max_repeat=max_repeat)
                )
            refined.extend(_example_repeat_bounds(node, context.hole_examples, self._config))
        elif isinstance(node, Category | CharClass):
            for pattern in self._config.synthesis.refinement.character_class_candidates:
                try:
                    refined.append(
                        parse_regex_to_ast(
                            pattern,
                            allow_raw_regex=self._config.synthesis.search.allow_raw_regex,
                        )
                    )
                except RegexValidationError:
                    continue
            refined.extend(_example_character_class_refinements(context.hole_examples))
        if self._config.synthesis.refinement.enable_anchor_toggle:
            refined.extend(_anchor_refinements(node))
        if self._config.synthesis.refinement.enable_separator_insertion:
            refined.extend(_separator_refinements(node, context.hole_examples))
        if self._config.synthesis.refinement.enable_optional_literal:
            refined.extend(_optional_literal_refinements(node, context.hole_examples))
        return _limit_unique_nodes(
            refined,
            self._config.synthesis.refinement.max_variants_per_operator,
        )

    def _nearby_bounds(self, node: Repeat) -> list[tuple[int, int | None]]:
        bounds = list(self._config.synthesis.search.allowed_repeat_bounds)
        lower_delta = self._config.synthesis.quantifier_refinement.lower_delta
        upper_delta = self._config.synthesis.quantifier_refinement.upper_delta
        finite_max = node.min_repeat + upper_delta if node.max_repeat is None else node.max_repeat
        for new_min in range(
            max(0, node.min_repeat - lower_delta),
            node.min_repeat + lower_delta + 1,
        ):
            for new_max in range(
                max(new_min, finite_max - upper_delta),
                finite_max + upper_delta + 1,
            ):
                bounds.append((new_min, new_max))
        return list(dict.fromkeys(bounds))


@dataclass
class _SearchContext:
    config: AppConfig
    hole_examples: HoleExampleSet
    negative_probes: list[str]
    stats: SearchStats
    start_time: float
    best_by_signature: dict[tuple[object, ...], SearchOption]
    best_by_canonical: dict[str, SearchOption]

    def should_stop(self) -> bool:
        if self.stats.generated_candidates >= self.config.synthesis.search.max_generated_candidates:
            self.stats.bounds_hit = True
            return True
        if len(self.best_by_signature) >= self.config.synthesis.search.max_behavior_classes:
            self.stats.bounds_hit = True
            return True
        elapsed = time.perf_counter() - self.start_time
        if elapsed >= self.config.synthesis.search.max_seconds_per_hole:
            self.stats.timed_out = True
            self.stats.bounds_hit = True
            return True
        return False

    def add_candidate(
        self,
        node: RegexNode,
        strategy: SynthesisStrategy,
        confidence: float | None,
        source_id: str | None,
        provenance: dict[str, Any],
    ) -> None:
        if self.should_stop():
            return
        self.stats.generated_candidates += 1
        self.stats.strategy_counts[str(strategy)] = (
            self.stats.strategy_counts.get(str(strategy), 0) + 1
        )

        try:
            node = normalize_node(node)
            regex = render_regex(node)
            self.stats.canonicalized_candidates += 1
            if len(regex) > self.config.synthesis.max_candidate_length:
                self.stats.pruned_invalid_regexes += 1
                return
            semantic_fit_score, signature, probe_matches, probe_rejections = _evaluate_local(
                regex,
                self.hole_examples,
                self.negative_probes,
                self.config,
            )
        except RegexValidationError:
            self.stats.pruned_invalid_regexes += 1
            return

        self.stats.evaluated_candidates += 1
        if semantic_fit_score is None:
            self.stats.pruned_local_example_failures += 1
            self.stats.pruned_constraint_failures += 1
            return

        ast_complexity_cost = node_syntactic_complexity_cost(
            node,
            self.config.synthesis.scoring.syntactic_complexity_cost,
        )
        option = SearchOption(
            regex=regex,
            node=node,
            strategy=strategy,
            confidence=confidence,
            source_id=source_id,
            semantic_fit_score=semantic_fit_score,
            ast_complexity_cost=ast_complexity_cost,
            provenance={
                **provenance,
                "ast_complexity_cost": ast_complexity_cost,
                "rendered": regex,
            },
            provenance_steps=_provenance_steps(
                strategy=strategy,
                provenance={**provenance, "ast_complexity_cost": ast_complexity_cost},
                output=regex,
            ),
            behavior_signature=signature,
            constraint_probe_matches=probe_matches,
            constraint_probe_rejections=probe_rejections,
        )

        canonical_existing = self.best_by_canonical.get(regex)
        if canonical_existing is not None and _candidate_ranking_score(
            canonical_existing,
            self.config,
        ) >= _candidate_ranking_score(option, self.config):
            self.stats.pruned_canonical_duplicates += 1
            self.stats.pruned_duplicates += 1
            return
        if canonical_existing is not None:
            self.stats.pruned_canonical_duplicates += 1
            self.stats.pruned_duplicates += 1
        self.best_by_canonical[regex] = option

        dedup_key = (
            (str(strategy), *signature)
            if self.config.synthesis.search.behavioral_dedup
            else ("regex", regex)
        )
        existing = self.best_by_signature.get(dedup_key)
        if existing is not None and _candidate_ranking_score(
            existing,
            self.config,
        ) >= _candidate_ranking_score(option, self.config):
            self.stats.pruned_behavior_duplicates += 1
            self.stats.pruned_duplicates += 1
            return
        if existing is not None:
            self.stats.pruned_behavior_duplicates += 1
            self.stats.pruned_duplicates += 1
        self.best_by_signature[dedup_key] = option


def _evaluate_local(
    regex: str,
    hole_examples: HoleExampleSet,
    negative_probes: list[str],
    config: AppConfig,
) -> tuple[float | None, tuple[object, ...], int, int]:
    hard_evaluation = evaluate_examples(regex, hole_examples.hard, config.matching.mode)
    if not hard_evaluation.success:
        return None, (), 0, 0

    semantic_fit = config.synthesis.scoring.semantic_fit
    semantic_fit_score = (
        hard_evaluation.positive_matched * semantic_fit.hard_local_match
        + hard_evaluation.negative_rejected * semantic_fit.hard_local_reject
    )
    signature_values: list[object] = []
    signature_examples = [
        *hole_examples.hard.positive,
        *hole_examples.hard.negative,
        *hole_examples.soft_positive,
    ]
    for example in signature_examples:
        matched = regex_matches(regex, example, config.matching.mode)
        signature_values.append(matched)

    soft_matches = 0
    for soft_positive in hole_examples.soft_positive:
        if regex_matches(regex, soft_positive, config.matching.mode):
            soft_matches += 1
    semantic_fit_score += soft_matches * semantic_fit.soft_positive_match

    probe_matches = 0
    probe_rejections = 0
    for probe in negative_probes:
        matched = regex_matches(regex, probe, MatchMode.FULLMATCH)
        signature_values.append(matched)
        if matched:
            probe_matches += 1
        else:
            probe_rejections += 1
    semantic_fit_score += probe_rejections * semantic_fit.tuple_negative_probe_reject

    if not signature_values:
        signature_values = ["regex", regex]
    return semantic_fit_score, tuple(signature_values), probe_matches, probe_rejections


def _candidate_ranking_score(option: SearchOption, config: AppConfig) -> float:
    return candidate_score_breakdown(
        regex=option.regex,
        strategy=option.strategy,
        confidence=option.confidence,
        semantic_fit_score=option.semantic_fit_score,
        ast_complexity_cost=option.ast_complexity_cost,
        config=config,
    ).total_score


def _select_options(
    options: list[SearchOption],
    limit: int,
    probe_count: int,
) -> list[SearchOption]:
    if len(options) <= limit:
        return options

    selected: list[SearchOption] = []
    selected_regexes: set[str] = set()

    def append(option: SearchOption) -> bool:
        if option.regex in selected_regexes:
            return len(selected) >= limit
        selected.append(option)
        selected_regexes.add(option.regex)
        return len(selected) >= limit

    for option in options:
        if option.strategy is SynthesisStrategy.DIRECT_REUSE and append(option):
            return selected

    seen_probe_behaviors: set[tuple[object, ...]] = set()
    if probe_count:
        for option in options:
            probe_behavior = option.behavior_signature[-probe_count:]
            if probe_behavior in seen_probe_behaviors:
                continue
            seen_probe_behaviors.add(probe_behavior)
            if append(option):
                return selected

    for option in options:
        if append(option):
            return selected
    return selected


def _component_provenance(component: RegexComponent) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "source_regex": _source_regex(component),
        "source_type": component.type,
        "source_id": component.source_id,
    }
    if provenance["source_regex"] != component.regex:
        provenance["normalized_source_regex"] = component.regex
    return provenance


def _source_regex(component: RegexComponent) -> str:
    original = component.metadata.get("embedding_anchor_stripped_from")
    return original if isinstance(original, str) else component.regex


def _anti_unify(left: RegexNode, right: RegexNode) -> RegexNode | None:
    if render_regex(left) == render_regex(right):
        return left
    class_join = _generalize_character_classes(left, right)
    if class_join is not None:
        return class_join
    if isinstance(left, Alt) and isinstance(right, Alt):
        options = tuple(left.options + right.options)
        return normalize_node(Alt(options))
    if isinstance(left, Alt):
        return normalize_node(Alt((*left.options, right)))
    if isinstance(right, Alt):
        return normalize_node(Alt((left, *right.options)))
    if isinstance(left, Repeat) and isinstance(right, Repeat):
        child = _anti_unify(left.child, right.child)
        if child is None:
            return None
        max_repeat = None if left.max_repeat is None or right.max_repeat is None else max(
            left.max_repeat,
            right.max_repeat,
        )
        return normalize_node(Repeat(
            child=child,
            min_repeat=min(left.min_repeat, right.min_repeat),
            max_repeat=max_repeat,
        ))
    if (
        isinstance(left, Concat)
        and isinstance(right, Concat)
    ):
        return _generalize_sequences(left.parts, right.parts)
    if isinstance(left, Literal) and isinstance(right, Literal):
        return _generalize_literals(left.value, right.value)
    return None


def _generalize_character_classes(left: RegexNode, right: RegexNode) -> RegexNode | None:
    left_family = _character_family(left)
    right_family = _character_family(right)
    if left_family is None or right_family is None:
        return None
    families = {left_family, right_family}
    if families == {"digit"}:
        return Category(r"\d")
    if families <= {"digit", "upper", "lower", "alpha", "alnum", "word"}:
        if "word" in families:
            return Category(r"\w")
        digit_with_alpha = "digit" in families and (
            "upper" in families or "lower" in families
        )
        if "alnum" in families or digit_with_alpha:
            return CharClass("A-Za-z0-9")
        if "alpha" in families or {"upper", "lower"} <= families:
            return CharClass("A-Za-z")
        if "upper" in families:
            return CharClass("A-Z")
        if "lower" in families:
            return CharClass("a-z")
    return None


def _character_family(node: RegexNode) -> str | None:
    rendered = render_regex(node)
    mapping = {
        r"\d": "digit",
        "[0-9]": "digit",
        "[A-Z]": "upper",
        "[a-z]": "lower",
        "[A-Za-z]": "alpha",
        "[A-Za-z0-9]": "alnum",
        r"\w": "word",
    }
    return mapping.get(rendered)


def _generalize_sequences(
    left_parts: tuple[RegexNode, ...],
    right_parts: tuple[RegexNode, ...],
) -> RegexNode | None:
    if len(left_parts) == len(right_parts):
        parts = tuple(
            _anti_unify(lpart, rpart)
            for lpart, rpart in zip(left_parts, right_parts, strict=True)
        )
        if any(part is None for part in parts):
            return None
        return normalize_node(Concat(tuple(part for part in parts if part is not None)))

    left_rendered = [render_regex(part) for part in left_parts]
    right_rendered = [render_regex(part) for part in right_parts]
    matcher = SequenceMatcher(None, left_rendered, right_rendered, autojunk=False)
    generalized: list[RegexNode] = []
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        left_slice = left_parts[left_start:left_end]
        right_slice = right_parts[right_start:right_end]
        if tag == "equal":
            generalized.extend(left_slice)
        elif tag == "replace" and len(left_slice) == len(right_slice):
            for lpart, rpart in zip(left_slice, right_slice, strict=True):
                part = _anti_unify(lpart, rpart)
                if part is None:
                    return None
                generalized.append(part)
        elif tag == "delete":
            generalized.append(_optional_sequence(left_slice))
        elif tag == "insert":
            generalized.append(_optional_sequence(right_slice))
        else:
            return None
    return normalize_node(Concat(tuple(generalized)))


def _optional_sequence(parts: tuple[RegexNode, ...]) -> RegexNode:
    child = parts[0] if len(parts) == 1 else normalize_node(Concat(parts))
    return Repeat(child=child, min_repeat=0, max_repeat=1)


def _generalize_literals(left: str, right: str) -> RegexNode | None:
    if left == right:
        return Literal(left)
    if len(left) != len(right):
        repeated = _generalize_repeated_literal_set(left, right)
        if repeated is not None:
            return repeated
        optional = _generalize_literals_with_optional_segments(left, right)
        if optional is not None:
            return optional
        return None
    parts = tuple(
        _generalize_char(lchar, rchar)
        for lchar, rchar in zip(left, right, strict=True)
    )
    if any(part is None for part in parts):
        return None
    typed_parts = tuple(part for part in parts if part is not None)
    if len(typed_parts) == 1:
        return typed_parts[0]
    return normalize_node(Concat(typed_parts))


def _generalize_literals_with_optional_segments(left: str, right: str) -> RegexNode | None:
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    parts: list[RegexNode] = []
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        left_text = left[left_start:left_end]
        right_text = right[right_start:right_end]
        if tag == "equal":
            parts.append(Literal(left_text))
        elif tag == "replace" and len(left_text) == len(right_text):
            for lchar, rchar in zip(left_text, right_text, strict=True):
                part = _generalize_char(lchar, rchar)
                if part is None:
                    return None
                parts.append(part)
        elif tag == "delete" and left_text:
            parts.append(Repeat(Literal(left_text), min_repeat=0, max_repeat=1))
        elif tag == "insert" and right_text:
            parts.append(Repeat(Literal(right_text), min_repeat=0, max_repeat=1))
        else:
            return None
    return normalize_node(Concat(tuple(parts)))


def _generalize_repeated_literal_set(left: str, right: str) -> RegexNode | None:
    values = [left, right]
    chars = "".join(values)
    if not chars:
        return None
    child: RegexNode
    if all(char.isdigit() for char in chars):
        child = Category(r"\d")
    elif all(char.isupper() for char in chars):
        child = CharClass("A-Z")
    elif all(char.islower() for char in chars):
        child = CharClass("a-z")
    elif all(char.isalpha() for char in chars):
        child = CharClass("A-Za-z")
    else:
        return None
    return Repeat(
        child=child,
        min_repeat=min(len(value) for value in values),
        max_repeat=max(len(value) for value in values),
    )


def _generalize_char(left: str, right: str) -> RegexNode | None:
    if left == right:
        return Literal(left)
    if left.isdigit() and right.isdigit():
        return Category(r"\d")
    if left.isupper() and right.isupper():
        return CharClass("A-Z")
    if left.islower() and right.islower():
        return CharClass("a-z")
    if left.isalpha() and right.isalpha():
        return CharClass("A-Za-z")
    if left.isalnum() and right.isalnum():
        return CharClass("A-Za-z0-9")
    return CharClass(f"{_escape_class_char(left)}{_escape_class_char(right)}")


def _anchor_refinements(node: RegexNode) -> list[RegexNode]:
    if isinstance(node, Anchor):
        return []
    rendered = render_regex(node)
    has_start_anchor = rendered.startswith(("^", r"\A"))
    has_end_anchor = rendered.endswith(("$", r"\Z"))
    refined: list[RegexNode] = []
    if not has_start_anchor:
        refined.append(Concat((Anchor("^"), node)))
    if not has_end_anchor:
        refined.append(Concat((node, Anchor("$"))))
    if not has_start_anchor and not has_end_anchor:
        refined.append(Concat((Anchor("^"), node, Anchor("$"))))
    return refined


def _separator_refinements(node: RegexNode, hole_examples: HoleExampleSet) -> list[RegexNode]:
    refined: list[RegexNode] = []
    for separator in _example_separators(hole_examples):
        optional_separator = Repeat(Literal(separator), min_repeat=0, max_repeat=1)
        refined.append(Concat((node, optional_separator)))
        refined.append(Concat((optional_separator, node)))
    return refined


def _optional_literal_refinements(
    node: RegexNode,
    hole_examples: HoleExampleSet,
) -> list[RegexNode]:
    refined: list[RegexNode] = []
    for literal in _edge_literals(hole_examples):
        optional_literal = Repeat(Literal(literal), min_repeat=0, max_repeat=1)
        refined.append(Concat((optional_literal, node)))
        refined.append(Concat((node, optional_literal)))
    return refined


def _example_repeat_bounds(
    node: Repeat,
    hole_examples: HoleExampleSet,
    config: AppConfig,
) -> list[RegexNode]:
    positives = hole_examples.hard.positive or hole_examples.soft_positive
    if not positives:
        return []
    child_regex = render_regex(node.child)
    if not all(
        regex_matches(child_regex, char, MatchMode.FULLMATCH)
        for example in positives
        for char in example
    ):
        return []
    min_repeat = min(len(example) for example in positives)
    max_repeat = max(len(example) for example in positives)
    if max_repeat > config.synthesis.search.max_ast_size * config.synthesis.max_candidate_length:
        return []
    return [Repeat(child=node.child, min_repeat=min_repeat, max_repeat=max_repeat)]


def _example_character_class_refinements(hole_examples: HoleExampleSet) -> list[RegexNode]:
    positives = [char for example in hole_examples.hard.positive for char in example]
    negatives = [char for example in hole_examples.hard.negative for char in example]
    if not positives:
        return []
    patterns = [
        r"\d",
        "[A-Z]",
        "[a-z]",
        "[A-Za-z]",
        "[A-Za-z0-9]",
        r"\w",
    ]
    refined: list[RegexNode] = []
    for pattern in patterns:
        try:
            if not all(regex_matches(pattern, char, MatchMode.FULLMATCH) for char in positives):
                continue
            if any(regex_matches(pattern, char, MatchMode.FULLMATCH) for char in negatives):
                continue
            refined.append(parse_regex_to_ast(pattern, allow_raw_regex=False))
        except RegexValidationError:
            continue
    return refined


def _example_separators(hole_examples: HoleExampleSet) -> list[str]:
    separators: list[str] = []
    examples = [*hole_examples.hard.positive, *hole_examples.soft_positive]
    for example in examples:
        for char in example:
            if not char.isalnum():
                _append_unique(separators, char)
    return separators


def _edge_literals(hole_examples: HoleExampleSet) -> list[str]:
    literals: list[str] = []
    examples = [*hole_examples.hard.positive, *hole_examples.soft_positive]
    for example in examples:
        if not example:
            continue
        for char in (example[0], example[-1]):
            if not char.isalnum():
                _append_unique(literals, char)
    return literals


def _limit_unique_nodes(nodes: list[RegexNode], limit: int) -> list[RegexNode]:
    limited: list[RegexNode] = []
    seen: set[str] = set()
    for node in nodes:
        rendered = render_regex(node)
        if rendered in seen:
            continue
        seen.add(rendered)
        limited.append(node)
        if len(limited) >= limit:
            break
    return limited


def _average_confidence(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _join_source_ids(left: str | None, right: str | None) -> str | None:
    values = [value for value in (left, right) if value is not None]
    return "+".join(values) if values else None


def _provenance_step(
    strategy: SynthesisStrategy,
    provenance: dict[str, Any],
    output: str,
) -> ProvenanceStep:
    source_regexes = provenance.get("source_regexes", [])
    if isinstance(provenance.get("source_regex"), str):
        source_regexes = [provenance["source_regex"]]
    source_ids = provenance.get("source_ids", [])
    if isinstance(provenance.get("source_id"), str):
        source_ids = [provenance["source_id"]]
    return ProvenanceStep(
        strategy=strategy,
        operation=str(provenance.get("operation", strategy.value)),
        inputs=[str(value) for value in source_regexes if isinstance(value, str)],
        output=output,
        source_ids=[str(value) for value in source_ids if isinstance(value, str)],
        metadata={
            key: value
            for key, value in provenance.items()
            if key not in {"source_regex", "source_regexes", "source_id", "source_ids"}
        },
    )


def _provenance_steps(
    strategy: SynthesisStrategy,
    provenance: dict[str, Any],
    output: str,
) -> tuple[ProvenanceStep, ...]:
    steps = [
        _provenance_step(
            strategy=strategy,
            provenance=provenance,
            output=output,
        )
    ]
    source_regex = provenance.get("source_regex")
    normalized_regex = provenance.get("normalized_source_regex")
    if (
        strategy is SynthesisStrategy.DIRECT_REUSE
        and isinstance(source_regex, str)
        and isinstance(normalized_regex, str)
        and source_regex != normalized_regex
    ):
        source_id = provenance.get("source_id")
        steps.append(
            ProvenanceStep(
                strategy=strategy,
                operation="strip_embedding_anchors",
                inputs=[source_regex],
                output=normalized_regex,
                source_ids=[source_id] if isinstance(source_id, str) else [],
            )
        )
    return tuple(steps)


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _escape_class_char(value: str) -> str:
    if value in {"\\", "]", "^", "-"}:
        return "\\" + value
    return value
