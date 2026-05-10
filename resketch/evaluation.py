from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from resketch.config import AppConfig, resolve_project_path
from resketch.models import DecompositionStats, Examples, SynthesisSpec
from resketch.synthesis import Synthesizer


class BenchmarkTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    sketch: str
    positive: list[str] = Field(default_factory=list)
    negative: list[str] = Field(default_factory=list)
    hole_examples: dict[str, Examples] = Field(default_factory=dict)


class BenchmarkSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: list[BenchmarkTask]


def load_benchmark(path: str | Path) -> BenchmarkSuite:
    resolved = resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return BenchmarkSuite.model_validate(data)


def run_benchmark(
    config: AppConfig,
    synthesizer: Synthesizer,
    path: str | Path | None = None,
) -> dict[str, Any]:
    suite = load_benchmark(path or config.evaluation.benchmark_path)
    task_results: list[dict[str, Any]] = []
    solved = 0
    explored = 0
    aggregate_decomposition = DecompositionStats()
    start = time.perf_counter()

    for task in suite.tasks:
        spec = SynthesisSpec(
            global_examples=Examples(positive=task.positive, negative=task.negative),
            explicit_hole_examples=task.hole_examples,
        )
        result = synthesizer.synthesize(task.sketch, spec)
        if result.success:
            solved += 1
        explored += result.trace.candidates_explored
        if result.trace.decomposition is not None:
            _add_decomposition_stats(aggregate_decomposition, result.trace.decomposition.stats)
        task_results.append(
            {
                "name": task.name,
                "success": result.success,
                "regex": result.regex,
                "score": result.score,
                "candidates_explored": result.trace.candidates_explored,
                "decomposition": (
                    result.trace.decomposition.stats.model_dump(mode="json")
                    if result.trace.decomposition
                    else None
                ),
                "failures": result.evaluation.failures if result.evaluation else [],
                "strategies_used": [s.value for s in result.trace.strategies_used],
            }
        )

    elapsed_seconds = time.perf_counter() - start
    aggregate_decomposition.recompute_rates()
    return {
        "tasks": task_results,
        "solved": solved,
        "total": len(suite.tasks),
        "candidates_explored": explored,
        "decomposition": aggregate_decomposition.model_dump(mode="json"),
        "elapsed_seconds": elapsed_seconds,
    }


def _add_decomposition_stats(target: DecompositionStats, source: DecompositionStats) -> None:
    target.global_positive_count += source.global_positive_count
    target.matched_positive_count += source.matched_positive_count
    target.unmatched_positive_count += source.unmatched_positive_count
    target.holes_with_hard_examples += source.holes_with_hard_examples
    target.holes_with_only_soft_examples += source.holes_with_only_soft_examples
    target.hard_evidence_count += source.hard_evidence_count
    target.soft_evidence_count += source.soft_evidence_count
    target.ambiguous_example_count += source.ambiguous_example_count
    target.explicit_evidence_count += source.explicit_evidence_count
    target.inferred_negative_count += source.inferred_negative_count
    target.tuple_negative_constraint_count += source.tuple_negative_constraint_count
    target.total_assignments_considered += source.total_assignments_considered
    target.examples_with_assignments += source.examples_with_assignments
    target.truncated_assignment_count += source.truncated_assignment_count
    target.repeat_unroll_truncation_count += source.repeat_unroll_truncation_count
