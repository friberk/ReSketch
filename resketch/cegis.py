from __future__ import annotations

from collections.abc import Callable

from resketch.config import AppConfig
from resketch.models import Examples, SynthesisResult, SynthesisSpec
from resketch.synthesis import Synthesizer

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


class CegisRunner:
    def __init__(
        self,
        config: AppConfig,
        synthesizer: Synthesizer,
        input_fn: InputFn = input,
        output_fn: OutputFn = print,
    ) -> None:
        self._config = config
        self._synthesizer = synthesizer
        self._input = input_fn
        self._output = output_fn

    def run(
        self,
        sketch_source: str,
        spec_or_examples: SynthesisSpec | Examples,
        interactive: bool | None = None,
    ) -> SynthesisResult:
        should_interact = self._config.cegis.interactive if interactive is None else interactive
        current_spec = _normalize_spec(spec_or_examples)
        last_result: SynthesisResult | None = None

        for _ in range(self._config.cegis.max_rounds):
            last_result = self._synthesizer.synthesize(sketch_source, current_spec)
            if not should_interact:
                return last_result

            self._display_result(last_result)
            decision = self._input(self._config.cegis.prompt_accept).strip().lower()
            if decision in self._config.cegis.accept_tokens:
                return last_result
            if decision in self._config.cegis.quit_tokens:
                return last_result
            if decision not in self._config.cegis.reject_tokens:
                self._output("Unrecognized decision; treating it as rejection.")

            added = self._collect_counterexamples(current_spec.global_examples)
            if not added:
                return last_result

        if last_result is None:
            return self._synthesizer.synthesize(sketch_source, current_spec)
        return last_result

    def _display_result(self, result: SynthesisResult) -> None:
        self._output(f"Candidate regex: {result.regex}")
        self._output(f"Success on current examples: {result.success}")
        if result.trace.hole_witnesses:
            rendered = ", ".join(
                f"{hole_id}={values}"
                for hole_id, values in sorted(result.trace.hole_witnesses.items())
            )
            self._output(f"Hole witnesses: {rendered}")
        if result.evaluation and result.evaluation.failures:
            for failure in result.evaluation.failures:
                self._output(f"- {failure}")
        if result.trace.diagnostics:
            for diagnostic in result.trace.diagnostics[:3]:
                self._output(f"- diagnostic: {diagnostic}")

    def _collect_counterexamples(self, examples: Examples) -> bool:
        added = False
        while True:
            value = self._input(self._config.cegis.prompt_counterexample)
            if not value:
                return added

            kind = self._input(self._config.cegis.prompt_counterexample_kind).strip().lower()
            if kind in self._config.cegis.positive_tokens:
                examples.positive.append(value)
                added = True
            elif kind in self._config.cegis.negative_tokens:
                examples.negative.append(value)
                added = True
            else:
                self._output("Unrecognized example kind; counterexample ignored.")


def _normalize_spec(spec_or_examples: SynthesisSpec | Examples) -> SynthesisSpec:
    if isinstance(spec_or_examples, SynthesisSpec):
        return spec_or_examples.model_copy(deep=True)
    return SynthesisSpec(global_examples=spec_or_examples.model_copy(deep=True))
