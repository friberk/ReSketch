from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resketch.config import AppConfig, load_config  # noqa: E402
from resketch.models import (  # noqa: E402
    CandidateRequest,
    CandidateSet,
    Examples,
    MatchMode,
    ProviderKind,
    SynthesisResult,
    SynthesisSpec,
)
from resketch.regex_engine import RegexValidationError, regex_matches  # noqa: E402
from resketch.retrieval import FixtureCandidateProvider, LLMCandidateProvider  # noqa: E402
from resketch.retrieval.base import CandidateProvider  # noqa: E402
from resketch.sketch import parse_sketch  # noqa: E402
from resketch.synthesis import Synthesizer  # noqa: E402


@dataclass(frozen=True)
class DemoScenario:
    name: str
    title: str
    sketch: str
    positive: tuple[str, ...]
    negative: tuple[str, ...]
    narration: str
    talking_points: tuple[str, ...]
    config_overrides: dict[str, object] | None = None


SCENARIOS: dict[str, DemoScenario] = {
    "api_gateway_audit": DemoScenario(
        name="api_gateway_audit",
        title="API Gateway Audit Log",
        sketch=(
            r"^(?:INFO|WARN|ERROR) \[{hole: iso_timestamp}\] "
            r"svc=(?:payments|checkout) req={hole: request_code} "
            r"src={hole: ipv4_address} "
            r"batch=\[(?:{hole: order_id};){2}{hole: order_id}\] "
            r'"{hole: http_method} (?:/api/v1/orders|/checkout/pay) HTTP/[12]\.[01]" '
            r"status=(?:200|503)$"
        ),
        positive=(
            'INFO [2026-04-27T14:05:33Z] svc=payments req=A12 '
            'src=192.168.0.1 batch=[ORD-1001;ORD-1002;ORD-1003] '
            '"GET /api/v1/orders HTTP/1.1" status=200',
            'WARN [2026-04-27T14:06:01-0500] svc=checkout req=AB12 '
            'src=10.0.0.7 batch=[ORD-2001;ORD-2002;ORD-2003] '
            '"POST /checkout/pay HTTP/2.0" status=503',
        ),
        negative=(
            'INFO [2026-04-27T14:05:33Z] svc=payments req=ZZ99 '
            'src=192.168.0.1 batch=[ORD-1001;ORD-1002;ORD-1003] '
            '"GET /api/v1/orders HTTP/1.1" status=200',
            'INFO [2026-04-27T14:05:33Z] svc=payments req=A12 '
            'src=999.999.999.999 batch=[ORD-1001;ORD-1002;ORD-1003] '
            '"GET /api/v1/orders HTTP/1.1" status=200',
            'INFO [2026-04-27T14:05:33Z] svc=payments req=A12 '
            'src=192.168.0.1 batch=[ORD-1001;BAD-1002;ORD-1003] '
            '"GET /api/v1/orders HTTP/1.1" status=200',
            'INFO [2026-04-27T14:05:33Z] svc=payments req=A12 '
            'src=192.168.0.1 batch=[ORD-1001;ORD-1002;ORD-1003] '
            '"BLAH /api/v1/orders HTTP/1.1" status=200',
            'INFO [2026-04-27T14:05:33Z] svc=payments req=A12 '
            'src=192.168.0.1 batch=[ORD-1001;ORD-1002;ORD-1003] '
            '"GET /api/v1/orders HTTP/3.0" status=200',
            'INFO [2026-04-27T14:05:33Z] svc=payments req=A12 '
            'src=192.168.0.1 batch=[ORD-1001;ORD-1002;ORD-1003] '
            '"GET /api/v1/orders HTTP/1.1" status=700',
        ),
        narration=(
            "A single log-parser sketch mixes concrete regex structure with semantic "
            "holes. The batch segment nests a typed order-id hole under a repeated "
            "sub-sketch, so the trace shows both compositional regex context and "
            "repeated-hole evidence."
        ),
        talking_points=(
            "Existing regex parts constrain log level, service, HTTP version, and status.",
            "The request code is generalized from concrete retrieved components.",
            "The nested batch sketch records repeated order-id witnesses.",
            "Tuple-negative probes favor stricter IPv4 and method candidates.",
        ),
        config_overrides={
            "decomposition.max_hole_capture_length": 32,
            "synthesis.beam_size": 96,
            "synthesis.max_hole_options": 80,
            "synthesis.global_pruning.max_partial_beams": 96,
            "synthesis.strategy_order": [
                "direct_reuse",
                "component_generalization",
                "fragment_recombination",
            ],
            "synthesis.search.max_generated_candidates": 1800,
            "synthesis.search.max_nodes_per_size": 8,
            "synthesis.search.max_seconds_per_hole": 5.0,
        },
    ),
    "payment_receipt": DemoScenario(
        name="payment_receipt",
        title="Payment Receipt Extraction",
        sketch=(
            "CARD:{hole: credit_card} EXP:{hole: expiration_date} "
            "CVV:{hole: cvv} AMT:{hole: currency_amount}"
        ),
        positive=(
            "CARD:4111 1111 1111 1111 EXP:04/26 CVV:123 AMT:$19.95",
            "CARD:5555-4444-3333-2222 EXP:11/27 CVV:1234 AMT:$250.00",
        ),
        negative=(
            "CARD:4111 1111 1111 1111 EXP:13/26 CVV:123 AMT:$19.95",
            "CARD:5555-4444-3333-2222 EXP:11/27 CVV:12 AMT:$250.00",
        ),
        narration=(
            "A multi-hole business pattern. Literal labels decompose examples into "
            "local subproblems while final validation still happens globally."
        ),
        talking_points=(
            "Direct reuse from retrieved semantic components.",
            "Hard local examples inferred from the labeled skeleton.",
            "Tuple constraints from negative receipt strings.",
        ),
    ),
    "server_log": DemoScenario(
        name="server_log",
        title="Server Log Line",
        sketch=(
            "{hole: iso_date} {hole: ipv4_address} "
            "\"{hole: http_method} {hole: url_path}\""
        ),
        positive=(
            '2026-04-27 192.168.0.1 "GET /api/v1/orders"',
            '2026-04-27 10.0.0.7 "POST /checkout/pay"',
        ),
        negative=(
            '2026-04-27 999.999.999.999 "GET /api/v1/orders"',
            '2026-04-27 192.168.0.1 "TRACE /api/v1/orders"',
        ),
        narration=(
            "A dense log parser with several typed holes and tight literal context."
        ),
        talking_points=(
            "Several independent retrieval calls become one global regex.",
            "Automata-backed pruning checks partial assignments against positives.",
            "Broad fallback candidates lose to more precise retrieved components.",
        ),
    ),
    "product_code": DemoScenario(
        name="product_code",
        title="Generalizing Product Codes",
        sketch="{hole: product_code}",
        positive=("A12", "AB12"),
        negative=("A13", "ACD12", "B12"),
        narration=(
            "The fixture gives only concrete observed codes. The engine must "
            "anti-unify them into a structural pattern."
        ),
        talking_points=(
            "Direct reuse fails because no single concrete code covers both positives.",
            "Component generalization infers an optional middle literal.",
            "The selected provenance exposes the anti-unification step.",
        ),
    ),
    "ambiguous_split": DemoScenario(
        name="ambiguous_split",
        title="Ambiguous Adjacent Holes",
        sketch="{hole: alpha}{hole: integer}",
        positive=("AB123", "CD456"),
        negative=("123AB", "ABXYZ", "AB_123"),
        narration=(
            "Adjacent holes are ambiguous because many splits are possible. "
            "ReSketch records soft evidence instead of pretending the split is unique."
        ),
        talking_points=(
            "Ambiguity groups show multiple capture assignments.",
            "Soft positives rank candidates without unsound pruning.",
            "Global examples still decide the final composed regex.",
        ),
    ),
    "nested_repeat": DemoScenario(
        name="nested_repeat",
        title="Nested Repeated Hole",
        sketch="IDS:(?:{hole: integer}-){2}{hole: integer}",
        positive=("IDS:12-34-56", "IDS:7-890-12"),
        negative=("IDS:12-34", "IDS:12-AA-56", "IDS:12-34-56-78"),
        narration=(
            "A typed hole appears under a regex repeat. Decomposition must infer "
            "per-occurrence witnesses, then assembly applies the surrounding repeat."
        ),
        talking_points=(
            "The sketch structure contributes the repeated separator pattern.",
            "The repeated hole gathers witnesses from each repeated occurrence.",
            "Repeated-hole constraints record all observed occurrences.",
        ),
    ),
    "tuple_negative": DemoScenario(
        name="tuple_negative",
        title="Tuple-Negative Constraint",
        sketch="{hole: year}-{hole: integer}",
        positive=("2026-42",),
        negative=("3026-42",),
        narration=(
            "A broad year candidate and an integer candidate can each look locally "
            "reasonable, but together they explain a negative example."
        ),
        talking_points=(
            "Negative decomposition records a joint forbidden tuple.",
            "Beam expansion prunes globally bad assignments.",
            "A more semantic year component survives the tuple constraint.",
        ),
    ),
}


def main() -> None:
    args = _parse_args()
    console = Console()
    pause = _pause_fn(console, enabled=not args.no_pause)
    scenario_names = list(SCENARIOS) if args.auto else [args.scenario]

    _render_header(console, args.provider, scenario_names)
    for index, scenario_name in enumerate(scenario_names, start=1):
        scenario = SCENARIOS[scenario_name]
        _render_scenario_intro(console, scenario, index, len(scenario_names))
        pause()
        try:
            result = _run_synthesis(args, scenario, console)
        except Exception as exc:
            if args.provider == "llm" and not args.no_fixture_fallback:
                _render_fallback_notice(console, args, exc)
                fallback_args = copy.copy(args)
                fallback_args.provider = "fixture"
                try:
                    result = _run_synthesis(fallback_args, scenario, console)
                except Exception as fallback_exc:
                    _render_failure(console, fallback_args, fallback_exc)
                    raise SystemExit(2) from fallback_exc
            else:
                _render_failure(console, args, exc)
                raise SystemExit(2) from exc

        _render_decomposition(console, result)
        pause()
        _render_retrieval(console, result)
        pause()
        _render_search(console, result)
        pause()
        _render_assembly(console, result)
        pause()
        _render_final(console, result)
        if args.json_trace:
            console.print_json(data=result.model_dump(mode="json"))
        pause()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive ReSketch synthesizer walkthrough for presentations.",
    )
    parser.add_argument(
        "--provider",
        choices=["llm", "fixture"],
        default="llm",
        help="Candidate provider. Defaults to live LLM retrieval.",
    )
    parser.add_argument(
        "--fixture-path",
        default="demo/candidates.yaml",
        help="Fixture YAML used when --provider fixture is selected.",
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="payment_receipt",
        help="Scenario to run when --auto is not set.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run every scenario in a fixed order.",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Do not wait for Enter between sections.",
    )
    parser.add_argument(
        "--json-trace",
        action="store_true",
        help="Print the raw JSON result after the narrated trace.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Override the configured LLM model for --provider llm.",
    )
    parser.add_argument(
        "--llm-timeout",
        default=120.0,
        type=float,
        help="Per-attempt LLM retrieval timeout in seconds. Defaults to 120 for demo runs.",
    )
    parser.add_argument(
        "--no-fixture-fallback",
        action="store_true",
        help="Fail instead of falling back to fixtures when --provider llm is unavailable.",
    )
    return parser.parse_args()


def _run_synthesis(
    args: argparse.Namespace,
    scenario: DemoScenario,
    console: Console | None = None,
) -> SynthesisResult:
    if args.provider == "llm" and not os.environ.get("OPENAI_API_KEY"):
        msg = "OPENAI_API_KEY is required for --provider llm"
        raise RuntimeError(msg)

    provider_kind = ProviderKind.LLM if args.provider == "llm" else ProviderKind.FIXTURE
    overrides = {
        "retrieval.provider": provider_kind.value,
        "retrieval.fixture_path": (
            args.fixture_path if provider_kind is ProviderKind.FIXTURE else None
        ),
        "retrieval.cache_path": ".resketch_cache/demo_llm_candidates.json",
        "cegis.interactive": False,
        "llm.timeout_seconds": args.llm_timeout,
        "synthesis.beam_size": 24,
        "synthesis.max_hole_options": 40,
        "synthesis.global_pruning.max_partial_beams": 24,
        "synthesis.search.max_generated_candidates": 1500,
        "synthesis.search.max_seconds_per_hole": 4.0,
    }
    if args.llm_model:
        overrides["llm.model"] = args.llm_model
    if scenario.config_overrides:
        overrides.update(scenario.config_overrides)
    config = load_config(overrides=overrides)
    provider = _build_provider(config, provider_kind)
    if console is not None:
        provider = _ProgressCandidateProvider(provider, console, provider_kind)
    synthesizer = Synthesizer(config, provider)
    spec = SynthesisSpec(
        global_examples=Examples(
            positive=list(scenario.positive),
            negative=list(scenario.negative),
        )
    )
    return synthesizer.synthesize(scenario.sketch, spec)


def _build_provider(config: AppConfig, provider_kind: ProviderKind) -> CandidateProvider:
    if provider_kind is ProviderKind.FIXTURE:
        return FixtureCandidateProvider(config)
    return LLMCandidateProvider(config)


class _ProgressCandidateProvider:
    def __init__(
        self,
        provider: CandidateProvider,
        console: Console,
        provider_kind: ProviderKind,
    ) -> None:
        self._provider = provider
        self._console = console
        self._provider_kind = provider_kind

    def retrieve(self, request: CandidateRequest) -> CandidateSet:
        label = f"{request.hole.identifier}:{request.hole.semantic_type}"
        started = time.perf_counter()
        self._console.print(Text(f"retrieving candidates for {label}...", style="dim"))
        candidate_set = self._provider.retrieve(request)
        elapsed = time.perf_counter() - started
        source = self._provider_kind.value
        if candidate_set.trace.get("from_cache"):
            source = f"{source} cache"
        self._console.print(
            Text(
                f"retrieved {len(candidate_set.components)} candidates for {label} "
                f"in {elapsed:.2f}s",
                style="dim",
            )
        )
        return candidate_set


def _pause_fn(console: Console, *, enabled: bool) -> Callable[[], None]:
    def pause() -> None:
        if enabled:
            console.input("\n[bold cyan]Press Enter to continue...[/bold cyan]")

    return pause


def _render_header(console: Console, provider: str, scenario_names: list[str]) -> None:
    text = Text()
    text.append("ReSketch interactive synthesizer demo\n", style="bold")
    # text.append(f"Provider: {provider}\n")
    text.append(f"Scenarios: {', '.join(scenario_names)}")
    console.print(Panel(text, title="Demo"))


def _render_scenario_intro(
    console: Console,
    scenario: DemoScenario,
    index: int,
    total: int,
) -> None:
    parsed = parse_sketch(scenario.sketch)
    console.rule(f"[bold]Scenario {index}/{total}: {scenario.title}")
    console.print(Panel(scenario.narration, title="Task description"))
    console.print(Syntax(scenario.sketch, "text", theme="ansi_dark", word_wrap=True))

    examples = Table(title="Global Examples")
    examples.add_column("Kind")
    examples.add_column("String")
    for value in scenario.positive:
        examples.add_row("positive", _literal(value))
    for value in scenario.negative:
        examples.add_row("negative", _literal(value))
    console.print(examples)

    holes = Table(title="Typed Holes")
    holes.add_column("Hole")
    holes.add_column("Semantic type")
    holes.add_column("Offsets")
    for hole in parsed.holes:
        holes.add_row(
            _literal(hole.identifier),
            _literal(hole.semantic_type),
            _literal(f"{hole.start}-{hole.end}"),
        )
    console.print(holes)

    # console.print(Panel(points, title="Talking Points"))


def _render_retrieval(console: Console, result: SynthesisResult) -> None:
    console.rule("[bold]2. Retrieval")
    for candidate_set in result.trace.candidate_sets:
        table = Table(title=f"{candidate_set.hole.identifier}: {candidate_set.hole.semantic_type}")
        #table.add_column("source")
        #table.add_column("confidence")
        table.add_column("regex")
        table.add_column("description")
        for component in candidate_set.components:
            table.add_row(
                #_literal(component.source_id or "-"),
                #_fmt_float(component.confidence),
                _literal(component.regex),
                _literal(component.description),
            )
        if not candidate_set.components:
            # table.add_row("-", "-", "-", "No candidates returned")
            table.add_row("-", "No candidates returned")
        console.print(table)


def _render_decomposition(console: Console, result: SynthesisResult) -> None:
    console.rule("[bold]1. Decomposition and Constraints")
    decomposition = result.trace.decomposition
    if decomposition is None:
        console.print("No decomposition was needed.")
        return

    stats = decomposition.stats
    summary = Table(title="Decomposition Stats")
    summary.add_column("Metric")
    summary.add_column("Value")
    rows = {
        "matched positives": f"{stats.matched_positive_count}/{stats.global_positive_count}",
        "ambiguous examples": str(stats.ambiguous_example_count),
        "ambiguity groups": str(stats.ambiguity_group_count),
        "hard evidence": str(stats.hard_evidence_count),
        "soft evidence": str(stats.soft_evidence_count),
        "tuple negatives": str(stats.tuple_negative_constraint_count),
        "repeated-hole constraints": str(stats.repeated_hole_constraint_count),
    }
    for metric, value in rows.items():
        summary.add_row(metric, value)
    console.print(summary)

    evidence = Table(title="Hole-Local Evidence")
    evidence.add_column("Hole")
    evidence.add_column("Hard +")
    evidence.add_column("Hard -")
    evidence.add_column("Soft +")
    evidence.add_column("Diagnostics")
    for hole_id, hole_examples in decomposition.by_hole.items():
        evidence.add_row(
            _literal(hole_id),
            _literal(_join(hole_examples.hard.positive)),
            _literal(_join(hole_examples.hard.negative)),
            _literal(_join(hole_examples.soft_positive)),
            _literal(_join(hole_examples.diagnostics)),
        )
    console.print(evidence)

    if decomposition.ambiguity_groups:
        ambiguity = Table(title="Ambiguity Groups")
        ambiguity.add_column("Source")
        ambiguity.add_column("Choices")
        ambiguity.add_column("Reason")
        for group in decomposition.ambiguity_groups[:5]:
            ambiguity.add_row(
                _literal(group.source_example),
                _literal(str(group.choices[:3])),
                _literal(group.reason),
            )
        console.print(ambiguity)

    if decomposition.tuple_negative_constraints:
        constraints = Table(title="Tuple-Negative Constraints")
        constraints.add_column("Source negative")
        constraints.add_column("Forbidden hole values")
        constraints.add_column("Reason")
        for constraint in decomposition.tuple_negative_constraints[:5]:
            constraints.add_row(
                _literal(constraint.source_example),
                _literal(str(constraint.hole_values)),
                _literal(constraint.reason),
            )
        console.print(constraints)


def _render_search(console: Console, result: SynthesisResult) -> None:
    console.rule("[bold]3. Search: Mix, Match, Generalize, Refine")
    stats_table = Table(title="Per-Hole Search Frontier")
    stats_table.add_column("Hole")
    stats_table.add_column("Frontier")
    stats_table.add_column("Generated")
    stats_table.add_column("Evaluated")
    stats_table.add_column("Duplicates")
    stats_table.add_column("Local fails")
    stats_table.add_column("Strategies")
    for hole_id, stats in result.trace.search_stats.items():
        stats_table.add_row(
            _literal(hole_id),
            str(stats.frontier_size),
            str(stats.generated_candidates),
            str(stats.evaluated_candidates),
            str(stats.pruned_duplicates),
            str(stats.pruned_local_example_failures),
            _literal(_strategy_counts(stats.strategy_counts)),
        )
    console.print(stats_table)

    for hole_id, features in result.trace.candidate_features.items():
        table = Table(title=f"Top Candidate Features for {hole_id}")
        table.add_column("regex")
        table.add_column("strategy")
        # table.add_column("confidence")
        table.add_column("semantic")
        table.add_column("source")
        table.add_column("syntax cost")
        table.add_column("score")
        for feature in features[:6]:
            breakdown = feature.score_breakdown
            table.add_row(
                _literal(feature.regex),
                _literal(feature.strategy.value),
                # _fmt_float(feature.confidence),
                _fmt_float(breakdown.semantic_fit_score),
                _fmt_float(breakdown.source_confidence_score),
                _fmt_float(breakdown.syntactic_complexity_cost),
                _fmt_float(breakdown.total_score),
            )
        console.print(table)


def _render_assembly(console: Console, result: SynthesisResult) -> None:
    console.rule("[bold]4. Global Assembly and Pruning")
    trace = result.trace
    table = Table(title="Beam Assembly")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("hole order", " -> ".join(trace.hole_order) or "-")
    table.add_row("candidates explored", str(trace.candidates_explored))
    table.add_row("partial positive prunes", str(trace.pruned_partial_beams))
    table.add_row("tuple-negative prunes", str(trace.pruned_tuple_negative_constraints))
    # table.add_row(
    #    "automata positive prunes",
    #    str(trace.automata_stats.pruned_positive_feasibility),
    #)
    #table.add_row(
    #    "automata negative prunes",
    #    str(trace.automata_stats.pruned_negative_feasibility),
    #)
    #table.add_row("automata fail-open count", str(trace.automata_stats.fail_open_count))
    console.print(table)

    provenance = Table(title="Selected Derivation")
    provenance.add_column("Hole")
    provenance.add_column("Operation")
    provenance.add_column("Inputs")
    provenance.add_column("Output")
    for hole_id, steps in trace.selected_provenance_steps.items():
        for step in steps:
            provenance.add_row(
                _literal(hole_id),
                _literal(f"{step.strategy.value}:{step.operation}"),
                _literal(_join(step.inputs)),
                _literal(step.output or "-"),
            )
    if not trace.selected_provenance_steps:
        provenance.add_row("-", "-", "-", "No complete assignment selected")
    console.print(provenance)


def _render_final(console: Console, result: SynthesisResult) -> None:
    console.rule("[bold]5. Final Regex")
    status = "SUCCESS" if result.success else "FAILED"
    style = "bold green" if result.success else "bold red"
    console.print(Panel(_literal(result.regex or "<no regex>"), title=status, border_style=style))

    assignment = Table(title="Hole Assignments")
    assignment.add_column("Hole")
    assignment.add_column("Regex")
    for hole_id, regex in result.trace.hole_assignments.items():
        assignment.add_row(_literal(hole_id), _literal(regex))
    if not result.trace.hole_assignments:
        assignment.add_row("-", "<none>")
    console.print(assignment)

    if not result.success:
        _render_failure_analysis(console, result)

    if result.evaluation is not None:
        evaluation = Table(title="Evaluation")
        evaluation.add_column("Metric")
        evaluation.add_column("Value")
        evaluation.add_row(
            "positives matched",
            f"{result.evaluation.positive_matched}/{result.evaluation.positive_total}",
        )
        evaluation.add_row(
            "negatives rejected",
            f"{result.evaluation.negative_rejected}/{result.evaluation.negative_total}",
        )
        evaluation.add_row("success", str(result.evaluation.success))
        for failure in result.evaluation.failures:
            evaluation.add_row("failure", _literal(failure))
        console.print(evaluation)

    # if scope is not None:
    #     completeness = Table(title="Completeness Scope")
    #     completeness.add_column("Field")
    #     completeness.add_column("Value")
    #     completeness.add_row("status", scope.status.value)
    #     completeness.add_row("complete within bounds", str(scope.complete_within_bounds))
    #     completeness.add_row("max AST size", str(scope.max_ast_size))
    #     completeness.add_row("limits hit", str(scope.limits_hit))
    #     completeness.add_row("timed out", str(scope.timed_out))
    #     completeness.add_row("reasons", _literal(_join(scope.incomplete_reasons)))
    #     console.print(completeness)


def _render_failure_analysis(console: Console, result: SynthesisResult) -> None:
    trace = result.trace
    table = Table(title="Failure Analysis")
    table.add_column("Signal")
    table.add_column("Value")
    table.add_row(
        "selected assignment",
        "none survived assembly" if result.regex is None else "best assignment failed validation",
    )
    table.add_row("partial-positive prunes", str(trace.pruned_partial_beams))
    table.add_row("tuple-negative prunes", str(trace.pruned_tuple_negative_constraints))
    table.add_row(
        "global-negative prunes",
        str(
            sum(
                stats.pruned_global_negative_feasibility
                for stats in trace.search_stats.values()
            )
        ),
    )
    table.add_row("diagnostics", _literal(_join(trace.diagnostics)))
    console.print(table)

    decomposition = trace.decomposition
    if decomposition is None or not decomposition.tuple_negative_constraints:
        return

    pressure = Table(title="Tuple-Negative Pressure")
    pressure.add_column("Negative")
    pressure.add_column("Candidate escape routes")
    pressure.add_column("Interpretation")
    for constraint in decomposition.tuple_negative_constraints[:5]:
        escape_routes = _tuple_escape_routes(result, constraint.hole_values)
        pressure.add_row(
            _literal(constraint.source_example),
            _literal(_join(escape_routes)),
            (
                "some local candidates can reject this negative"
                if escape_routes
                else "all retained local candidates match this negative tuple"
            ),
        )
    console.print(pressure)


def _tuple_escape_routes(
    result: SynthesisResult,
    hole_values: dict[str, list[str]],
) -> list[str]:
    routes: list[str] = []
    for hole_id, values in hole_values.items():
        for feature in result.trace.candidate_features.get(hole_id, []):
            if any(not _matches_full(feature.regex, value) for value in values):
                routes.append(f"{hole_id}: {feature.regex}")
                break
    return routes


def _matches_full(regex: str, value: str) -> bool:
    try:
        return regex_matches(regex, value, MatchMode.FULLMATCH)
    except RegexValidationError:
        return True


def _render_failure(console: Console, args: argparse.Namespace, exc: Exception) -> None:
    console.print(Panel(_literal(str(exc)), title="Demo failed", border_style="red"))
    if args.provider == "llm":
        command = (
            "poetry run python demo/interactive_demo.py "
            f"--provider fixture --scenario {args.scenario}"
        )
        console.print("Use the deterministic fallback:")
        console.print(Syntax(command, "bash", theme="ansi_dark", word_wrap=True))


def _render_fallback_notice(
    console: Console,
    args: argparse.Namespace,
    exc: Exception,
) -> None:
    message = Text()
    message.append("LLM retrieval failed, so this run is switching to fixtures.\n")
    message.append(str(exc))
    console.print(Panel(message, title="Using fixture fallback", border_style="yellow"))
    command = (
        "poetry run python demo/interactive_demo.py "
        f"--provider llm --scenario {args.scenario} --no-fixture-fallback"
    )
    console.print("To require live LLM retrieval, rerun with:")
    console.print(Syntax(command, "bash", theme="ansi_dark", word_wrap=True))


def _join(values: list[str] | tuple[str, ...]) -> str:
    if not values:
        return "-"
    return ", ".join(values)


def _literal(value: object) -> Text:
    return Text(str(value), overflow="fold")


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _strategy_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{key.rsplit('.', 1)[-1]}={value}" for key, value in counts.items())


if __name__ == "__main__":
    main()
