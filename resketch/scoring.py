from __future__ import annotations

from resketch.config import AppConfig, SyntacticComplexityCostConfig
from resketch.models import CandidateScoreBreakdown, SynthesisStrategy
from resketch.regex_ast import (
    Alt,
    Anchor,
    Category,
    CharClass,
    Concat,
    Dot,
    Literal,
    RawRegex,
    RegexNode,
    Repeat,
    is_broad_wildcard,
)


def node_syntactic_complexity_cost(
    node: RegexNode,
    cost_config: SyntacticComplexityCostConfig,
) -> float:
    if isinstance(node, Literal):
        return cost_config.literal * max(1, len(node.value))
    if isinstance(node, Dot):
        return cost_config.dot + cost_config.wildcard_penalty
    if isinstance(node, Category):
        return cost_config.category
    if isinstance(node, CharClass):
        return cost_config.char_class
    if isinstance(node, Anchor):
        return cost_config.anchor
    if isinstance(node, RawRegex):
        return cost_config.raw_regex
    if isinstance(node, Concat):
        return cost_config.concat + sum(
            node_syntactic_complexity_cost(part, cost_config)
            for part in node.parts
        )
    if isinstance(node, Alt):
        return (
            cost_config.alt
            + sum(
                node_syntactic_complexity_cost(option, cost_config)
                for option in node.options
            )
            + len(node.options) * cost_config.alternation_option_penalty
        )
    if isinstance(node, Repeat):
        wildcard_penalty = (
            cost_config.wildcard_penalty if is_broad_wildcard(node) else 0.0
        )
        return (
            cost_config.repeat
            + node_syntactic_complexity_cost(node.child, cost_config)
            + wildcard_penalty
        )


def candidate_score_breakdown(
    *,
    regex: str,
    strategy: SynthesisStrategy,
    confidence: float | None,
    semantic_fit_score: float,
    ast_complexity_cost: float,
    config: AppConfig,
) -> CandidateScoreBreakdown:
    if confidence is None:
        confidence = config.llm.missing_confidence
    scoring = config.synthesis.scoring
    source_confidence_score = confidence * scoring.source_confidence.weight
    length_complexity_cost = (
        len(regex) * scoring.syntactic_complexity_cost.length_penalty
    )
    syntactic_complexity_cost = ast_complexity_cost + length_complexity_cost
    strategy_penalty = scoring.strategy_penalty[strategy]
    total_score = (
        semantic_fit_score
        + source_confidence_score
        - syntactic_complexity_cost
        - strategy_penalty
    )
    return CandidateScoreBreakdown(
        semantic_fit_score=semantic_fit_score,
        source_confidence_score=source_confidence_score,
        ast_complexity_cost=ast_complexity_cost,
        length_complexity_cost=length_complexity_cost,
        syntactic_complexity_cost=syntactic_complexity_cost,
        strategy_penalty=strategy_penalty,
        total_score=total_score,
    )
