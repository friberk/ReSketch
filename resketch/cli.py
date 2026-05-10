from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from resketch.cegis import CegisRunner
from resketch.config import AppConfig, load_config
from resketch.evaluation import run_benchmark
from resketch.models import DecompositionMode, Examples, MatchMode, ProviderKind, SynthesisSpec
from resketch.regex_engine import evaluate_examples
from resketch.retrieval import (
    DBCandidateProvider,
    FixtureCandidateProvider,
    LLMCandidateProvider,
)
from resketch.retrieval.base import CandidateProvider
from resketch.sketch import Sketch, parse_sketch
from resketch.synthesis import Synthesizer

app = typer.Typer(help="ReSketch semantic regex synthesis CLI.")
config_app = typer.Typer(help="Configuration helpers.")
app.add_typer(config_app, name="config")
console = Console()


def _load_with_overrides(
    config_path: Path | None,
    provider: ProviderKind | None,
    model: str | None,
    timeout_seconds: float | None,
    temperature: float | None,
    max_retries: int | None,
    beam_size: int | None,
    max_rounds: int | None,
    match_mode: MatchMode | None,
    decomposition_mode: DecompositionMode | None,
    automata_pruning: bool | None,
    interactive: bool | None,
    fixture_path: Path | None,
    db_path: Path | None,
) -> AppConfig:
    overrides = {
        "retrieval.provider": provider.value if provider else None,
        "llm.model": model,
        "llm.timeout_seconds": timeout_seconds,
        "llm.temperature": temperature,
        "llm.max_retries": max_retries,
        "synthesis.beam_size": beam_size,
        "cegis.max_rounds": max_rounds,
        "matching.mode": match_mode.value if match_mode else None,
        "decomposition.mode": decomposition_mode.value if decomposition_mode else None,
        "synthesis.automata.enabled": automata_pruning,
        "cegis.interactive": interactive,
        "retrieval.fixture_path": str(fixture_path) if fixture_path else None,
        "retrieval.db_path": str(db_path) if db_path else None,
    }
    return load_config(path=config_path, overrides=overrides)


def _build_provider(config: AppConfig) -> CandidateProvider:
    if config.retrieval.provider is ProviderKind.FIXTURE:
        return FixtureCandidateProvider(config)
    if config.retrieval.provider is ProviderKind.DB:
        return DBCandidateProvider(config)
    return LLMCandidateProvider(config)


def _parse_hole_example_entries(
    sketch_source: str,
    hole_pos: list[str] | None,
    hole_neg: list[str] | None,
) -> dict[str, Examples]:
    sketch = parse_sketch(sketch_source)
    by_hole: dict[str, Examples] = {}
    for entry in hole_pos or []:
        hole_id, value = _parse_hole_example_entry(sketch, entry)
        by_hole.setdefault(hole_id, Examples()).positive.append(value)
    for entry in hole_neg or []:
        hole_id, value = _parse_hole_example_entry(sketch, entry)
        by_hole.setdefault(hole_id, Examples()).negative.append(value)
    return by_hole


def _parse_hole_example_entry(sketch: Sketch, entry: str) -> tuple[str, str]:
    if "=" not in entry:
        msg = f"Expected hole example in the form h0=value or semantic_type=value: {entry!r}"
        raise typer.BadParameter(msg)
    raw_hole, value = entry.split("=", 1)
    hole_id = _resolve_hole_reference(sketch, raw_hole.strip())
    return hole_id, value


def _resolve_hole_reference(sketch: Sketch, reference: str) -> str:
    hole_ids = {hole.identifier for hole in sketch.holes}
    if reference in hole_ids:
        return reference

    matches = [
        hole.identifier
        for hole in sketch.holes
        if hole.semantic_type == reference
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        msg = f"Semantic type {reference!r} appears more than once; use a hole id like h0."
        raise typer.BadParameter(msg)

    msg = f"Unknown hole reference {reference!r}; run inspect-sketch to see valid hole ids."
    raise typer.BadParameter(msg)


@app.command()
def synthesize(
    sketch: Annotated[str, typer.Option(help="Semantic sketch containing typed holes.")],
    pos: Annotated[list[str] | None, typer.Option("--pos", help="Positive example.")] = None,
    neg: Annotated[list[str] | None, typer.Option("--neg", help="Negative example.")] = None,
    hole_pos: Annotated[
        list[str] | None,
        typer.Option("--hole-pos", help="Explicit hole positive example, e.g. h0=123."),
    ] = None,
    hole_neg: Annotated[
        list[str] | None,
        typer.Option("--hole-neg", help="Explicit hole negative example, e.g. h1=12."),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Optional YAML config overlay."),
    ] = None,
    provider: Annotated[
        ProviderKind | None,
        typer.Option("--provider", help="Candidate provider."),
    ] = None,
    model: Annotated[str | None, typer.Option("--model", help="LiteLLM model name.")] = None,
    timeout_seconds: Annotated[
        float | None,
        typer.Option("--timeout-seconds", help="LLM timeout override."),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option("--temperature", help="LLM temperature override."),
    ] = None,
    max_retries: Annotated[
        int | None,
        typer.Option("--max-retries", help="LLM retry override."),
    ] = None,
    beam_size: Annotated[
        int | None,
        typer.Option("--beam-size", help="Synthesis beam size."),
    ] = None,
    max_rounds: Annotated[
        int | None,
        typer.Option("--max-rounds", help="CEGIS round limit."),
    ] = None,
    match_mode: Annotated[
        MatchMode | None,
        typer.Option("--match-mode", help="Regex matching mode."),
    ] = None,
    decomposition_mode: Annotated[
        DecompositionMode | None,
        typer.Option("--decomposition-mode", help="Hole-example decomposition mode."),
    ] = None,
    automata_pruning: Annotated[
        bool | None,
        typer.Option(
            "--automata-pruning/--no-automata-pruning",
            help="Enable or disable automata-backed partial pruning.",
        ),
    ] = None,
    interactive: Annotated[
        bool | None,
        typer.Option("--interactive/--no-interactive", help="Enable or disable CEGIS prompts."),
    ] = None,
    fixture_path: Annotated[
        Path | None,
        typer.Option("--fixture-path", help="Fixture provider YAML."),
    ] = None,
    db_path: Annotated[
        Path | None,
        typer.Option("--db-path", help="SQLite candidate database for provider=db."),
    ] = None,
) -> None:
    loaded = _load_with_overrides(
        config,
        provider,
        model,
        timeout_seconds,
        temperature,
        max_retries,
        beam_size,
        max_rounds,
        match_mode,
        decomposition_mode,
        automata_pruning,
        interactive,
        fixture_path,
        db_path,
    )
    candidate_provider = _build_provider(loaded)
    synthesizer = Synthesizer(loaded, candidate_provider)
    runner = CegisRunner(loaded, synthesizer, output_fn=console.print)
    spec = SynthesisSpec(
        global_examples=Examples(positive=pos or [], negative=neg or []),
        explicit_hole_examples=_parse_hole_example_entries(sketch, hole_pos, hole_neg),
    )
    result = runner.run(
        sketch,
        spec,
        interactive=loaded.cegis.interactive,
    )
    console.print_json(data=result.model_dump(mode="json"))


@app.command()
def validate(
    regex: Annotated[str, typer.Option(help="Python regex to validate.")],
    pos: Annotated[list[str] | None, typer.Option("--pos", help="Positive example.")] = None,
    neg: Annotated[list[str] | None, typer.Option("--neg", help="Negative example.")] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Optional YAML config overlay."),
    ] = None,
    match_mode: Annotated[
        MatchMode | None,
        typer.Option("--match-mode", help="Regex matching mode."),
    ] = None,
) -> None:
    loaded = _load_with_overrides(
        config,
        provider=None,
        model=None,
        timeout_seconds=None,
        temperature=None,
        max_retries=None,
        beam_size=None,
        max_rounds=None,
        match_mode=match_mode,
        decomposition_mode=None,
        automata_pruning=None,
        interactive=None,
        fixture_path=None,
        db_path=None,
    )
    evaluation = evaluate_examples(
        regex,
        Examples(positive=pos or [], negative=neg or []),
        loaded.matching.mode,
    )

    table = Table(title="Validation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("positive", f"{evaluation.positive_matched}/{evaluation.positive_total}")
    table.add_row("negative", f"{evaluation.negative_rejected}/{evaluation.negative_total}")
    table.add_row("success", str(evaluation.success))
    console.print(table)
    if evaluation.failures:
        console.print_json(data={"failures": evaluation.failures})


@app.command("inspect-sketch")
def inspect_sketch(
    sketch: Annotated[str, typer.Option(help="Semantic sketch containing typed holes.")],
) -> None:
    parsed = parse_sketch(sketch)
    table = Table(title="Sketch Holes")
    table.add_column("Hole")
    table.add_column("Semantic Type")
    table.add_column("Start")
    table.add_column("End")
    for hole in parsed.holes:
        table.add_row(hole.identifier, hole.semantic_type, str(hole.start), str(hole.end))
    console.print(table)


@config_app.command("show")
def config_show(
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Optional YAML config overlay."),
    ] = None,
) -> None:
    loaded = load_config(path=config)
    console.print_json(data=loaded.model_dump(mode="json"))


@app.command("eval")
def eval_command(
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Optional YAML config overlay."),
    ] = None,
    provider: Annotated[
        ProviderKind | None,
        typer.Option("--provider", help="Candidate provider."),
    ] = None,
    decomposition_mode: Annotated[
        DecompositionMode | None,
        typer.Option("--decomposition-mode", help="Hole-example decomposition mode."),
    ] = None,
    automata_pruning: Annotated[
        bool | None,
        typer.Option(
            "--automata-pruning/--no-automata-pruning",
            help="Enable or disable automata-backed partial pruning.",
        ),
    ] = None,
    benchmark_path: Annotated[
        Path | None,
        typer.Option("--benchmarks", help="Benchmark YAML path."),
    ] = None,
    fixture_path: Annotated[
        Path | None,
        typer.Option("--fixture-path", help="Fixture provider YAML."),
    ] = None,
    db_path: Annotated[
        Path | None,
        typer.Option("--db-path", help="SQLite candidate database for provider=db."),
    ] = None,
) -> None:
    loaded = _load_with_overrides(
        config,
        provider,
        model=None,
        timeout_seconds=None,
        temperature=None,
        max_retries=None,
        beam_size=None,
        max_rounds=None,
        match_mode=None,
        decomposition_mode=decomposition_mode,
        automata_pruning=automata_pruning,
        interactive=False,
        fixture_path=fixture_path,
        db_path=db_path,
    )
    candidate_provider = _build_provider(loaded)
    synthesizer = Synthesizer(loaded, candidate_provider)
    result = run_benchmark(loaded, synthesizer, path=benchmark_path)
    console.print(json.dumps(result, indent=2))


def main() -> None:
    app()
