from __future__ import annotations

import string
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from resketch.config import AppConfig
from resketch.models import Examples, MatchMode
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
    parse_regex_to_ast,
)
from resketch.regex_engine import RegexValidationError
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
)


class AutomataUnsupportedError(RuntimeError):
    """Raised when a regex/sketch fragment cannot be compiled soundly."""


@dataclass(frozen=True)
class AutomataFeasibilityResult:
    feasible: bool
    supported: bool
    reasons: tuple[str, ...] = ()
    state_count: int = 0
    alphabet_size: int = 0


def check_partial_positive_feasibility(
    config: AppConfig,
    sketch: Sketch,
    assignments: dict[str, str],
    examples: Examples,
) -> AutomataFeasibilityResult:
    if config.matching.mode is not MatchMode.FULLMATCH:
        return _unsupported("automata pruning only supports fullmatch mode", config)

    alphabet_size = 0
    try:
        alphabet = _alphabet_for(config, examples)
        alphabet_size = len(alphabet)
        compiler = _AutomataCompiler(config=config, alphabet=alphabet)
        nfa = compiler.compile_sketch_node(sketch.root, assignments)
        state_count = _state_count(nfa)
        if state_count > config.synthesis.automata.max_states:
            return _unsupported(
                f"automaton state bound exceeded: {state_count}",
                config,
                alphabet_size=len(alphabet),
                state_count=state_count,
            )
        feasible = all(bool(nfa.accepts_input(example)) for example in examples.positive)
        return AutomataFeasibilityResult(
            feasible=feasible,
            supported=True,
            state_count=state_count,
            alphabet_size=len(alphabet),
        )
    except AutomataUnsupportedError as exc:
        return _unsupported(str(exc), config, alphabet_size=alphabet_size)


def check_complete_negative_rejection(
    config: AppConfig,
    sketch: Sketch,
    assignments: dict[str, str],
    examples: Examples,
) -> AutomataFeasibilityResult:
    if config.matching.mode is not MatchMode.FULLMATCH:
        return _unsupported("automata pruning only supports fullmatch mode", config)
    if not examples.negative:
        return AutomataFeasibilityResult(feasible=True, supported=True)

    alphabet_size = 0
    try:
        alphabet = _alphabet_for(config, examples)
        alphabet_size = len(alphabet)
        compiler = _AutomataCompiler(config=config, alphabet=alphabet)
        nfa = compiler.compile_sketch_node(sketch.root, assignments)
        state_count = _state_count(nfa)
        if state_count > config.synthesis.automata.max_states:
            return _unsupported(
                f"automaton state bound exceeded: {state_count}",
                config,
                alphabet_size=len(alphabet),
                state_count=state_count,
            )
        rejects_all = all(not bool(nfa.accepts_input(example)) for example in examples.negative)
        return AutomataFeasibilityResult(
            feasible=rejects_all,
            supported=True,
            state_count=state_count,
            alphabet_size=len(alphabet),
        )
    except AutomataUnsupportedError as exc:
        return _unsupported(str(exc), config, alphabet_size=alphabet_size)


def _unsupported(
    reason: str,
    config: AppConfig,
    *,
    alphabet_size: int = 0,
    state_count: int = 0,
) -> AutomataFeasibilityResult:
    if not config.synthesis.automata.fail_open_on_unsupported:
        return AutomataFeasibilityResult(
            feasible=False,
            supported=False,
            reasons=(reason,),
            state_count=state_count,
            alphabet_size=alphabet_size,
        )
    return AutomataFeasibilityResult(
        feasible=True,
        supported=False,
        reasons=(reason,),
        state_count=state_count,
        alphabet_size=alphabet_size,
    )


class _AutomataCompiler:
    def __init__(self, config: AppConfig, alphabet: frozenset[str]) -> None:
        self._config = config
        self._alphabet = alphabet
        self._counter = 0
        try:
            from automata.fa.nfa import NFA
        except ImportError as exc:
            msg = "automata-lib is not installed"
            raise AutomataUnsupportedError(msg) from exc
        self._nfa_type: Any = NFA

    def compile_sketch_node(
        self,
        node: SketchNode,
        assignments: dict[str, str],
    ) -> Any:
        if isinstance(node, SketchLiteral):
            return self._literal_string(node.value)
        if isinstance(node, SketchDot):
            return self._char_set(self._dot_chars())
        if isinstance(node, SketchCategory):
            return self._char_set(self._category_chars(node.pattern))
        if isinstance(node, SketchCharClass):
            return self._char_set(self._char_class_chars(node.body, node.negated))
        if isinstance(node, SketchAnchor):
            return self._anchor(node.pattern)
        if isinstance(node, SketchHole):
            assigned = assignments.get(node.hole.identifier)
            if assigned is None:
                return self._bounded_wildcard(
                    self._config.synthesis.automata.max_unknown_hole_length
                )
            try:
                parsed = parse_regex_to_ast(
                    assigned,
                    allow_raw_regex=self._config.synthesis.search.allow_raw_regex,
                )
            except RegexValidationError as exc:
                raise AutomataUnsupportedError(str(exc)) from exc
            return self.compile_regex_node(parsed)
        if isinstance(node, SketchConcat):
            return self._concat(
                [self.compile_sketch_node(part, assignments) for part in node.parts]
            )
        if isinstance(node, SketchAlt):
            return self._union(
                [self.compile_sketch_node(part, assignments) for part in node.options]
            )
        if isinstance(node, SketchRepeat):
            return self._repeat_sketch(node, assignments)
        raise AutomataUnsupportedError(f"unsupported sketch node: {type(node).__name__}")

    def compile_regex_node(self, node: RegexNode) -> Any:
        if isinstance(node, Literal):
            return self._literal_string(node.value)
        if isinstance(node, Dot):
            return self._char_set(self._dot_chars())
        if isinstance(node, Category):
            return self._char_set(self._category_chars(node.pattern))
        if isinstance(node, CharClass):
            return self._char_set(self._char_class_chars(node.body, node.negated))
        if isinstance(node, Anchor):
            return self._anchor(node.pattern)
        if isinstance(node, Concat):
            return self._concat([self.compile_regex_node(part) for part in node.parts])
        if isinstance(node, Alt):
            return self._union([self.compile_regex_node(option) for option in node.options])
        if isinstance(node, Repeat):
            return self._repeat_regex(node)
        if isinstance(node, RawRegex):
            msg = "raw regex nodes are outside the automata-supported subset"
            raise AutomataUnsupportedError(msg)
        raise AutomataUnsupportedError(f"unsupported regex node: {type(node).__name__}")

    def _repeat_sketch(self, node: SketchRepeat, assignments: dict[str, str]) -> Any:
        def build_child() -> Any:
            return self.compile_sketch_node(node.child, assignments)

        return self._repeat(build_child, node.min_repeat, node.max_repeat)

    def _repeat_regex(self, node: Repeat) -> Any:
        def build_child() -> Any:
            return self.compile_regex_node(node.child)

        return self._repeat(build_child, node.min_repeat, node.max_repeat)

    def _repeat(
        self,
        build_child: Callable[[], Any],
        min_repeat: int,
        max_repeat: int | None,
    ) -> Any:
        if max_repeat is not None and max_repeat < min_repeat:
            raise AutomataUnsupportedError("invalid repeat bounds")
        if max_repeat is None:
            prefix = self._exact_repeat(build_child, min_repeat)
            star = build_child().kleene_star()
            return self._concat([prefix, star])
        return self._union(
            [self._exact_repeat(build_child, count) for count in range(min_repeat, max_repeat + 1)]
        )

    def _exact_repeat(self, build_child: Callable[[], Any], count: int) -> Any:
        if count == 0:
            return self._empty_string()
        return self._concat([build_child() for _ in range(count)])

    def _literal_string(self, value: str) -> Any:
        if not value:
            return self._empty_string()
        return self._concat([self._char_set(frozenset({char})) for char in value])

    def _empty_string(self) -> Any:
        state = self._new_state()
        return self._make_nfa(
            states={state},
            transitions={state: {}},
            initial_state=state,
            final_states={state},
        )

    def _bounded_wildcard(self, max_length: int) -> Any:
        states = [self._new_state() for _ in range(max_length + 1)]
        transitions: dict[str, dict[str, set[str]]] = {}
        for index, state in enumerate(states):
            if index == max_length:
                transitions[state] = {}
            else:
                transitions[state] = {
                    char: {states[index + 1]}
                    for char in self._alphabet
                }
        return self._make_nfa(
            states=set(states),
            transitions=transitions,
            initial_state=states[0],
            final_states=set(states),
        )

    def _char_set(self, chars: frozenset[str]) -> Any:
        start = self._new_state()
        end = self._new_state()
        transitions = {
            start: {
                char: {end}
                for char in chars
                if char in self._alphabet
            },
            end: {},
        }
        return self._make_nfa(
            states={start, end},
            transitions=transitions,
            initial_state=start,
            final_states={end},
        )

    def _anchor(self, pattern: str) -> Any:
        if pattern in {"^", "$", r"\A", r"\Z"}:
            return self._empty_string()
        raise AutomataUnsupportedError(f"unsupported anchor: {pattern}")

    def _concat(self, nfas: list[Any]) -> Any:
        if not nfas:
            return self._empty_string()
        result = nfas[0]
        for nfa in nfas[1:]:
            result = result.concatenate(nfa)
        return result

    def _union(self, nfas: list[Any]) -> Any:
        if not nfas:
            return self._char_set(frozenset())
        result = nfas[0]
        for nfa in nfas[1:]:
            result = result.union(nfa)
        return result

    def _make_nfa(
        self,
        *,
        states: set[str],
        transitions: dict[str, dict[str, set[str]]],
        initial_state: str,
        final_states: set[str],
    ) -> Any:
        return self._nfa_type(
            states=states,
            input_symbols=set(self._alphabet),
            transitions=transitions,
            initial_state=initial_state,
            final_states=final_states,
        )

    def _new_state(self) -> str:
        state = f"q{self._counter}"
        self._counter += 1
        return state

    def _dot_chars(self) -> frozenset[str]:
        return frozenset(char for char in self._alphabet if char != "\n")

    def _category_chars(self, pattern: str) -> frozenset[str]:
        if pattern == r"\d":
            return frozenset(char for char in self._alphabet if char.isdigit())
        if pattern == r"\D":
            return frozenset(char for char in self._alphabet if not char.isdigit())
        if pattern == r"\s":
            return frozenset(char for char in self._alphabet if char.isspace())
        if pattern == r"\S":
            return frozenset(char for char in self._alphabet if not char.isspace())
        if pattern == r"\w":
            return frozenset(
                char for char in self._alphabet if char.isalnum() or char == "_"
            )
        if pattern == r"\W":
            return frozenset(
                char for char in self._alphabet if not (char.isalnum() or char == "_")
            )
        raise AutomataUnsupportedError(f"unsupported category: {pattern}")

    def _char_class_chars(self, body: str, negated: bool) -> frozenset[str]:
        chars: set[str] = set()
        index = 0
        while index < len(body):
            char = body[index]
            if char == "\\" and index + 1 < len(body):
                escaped = body[index + 1]
                category = self._category_escape(escaped)
                if category is not None:
                    chars.update(category)
                else:
                    chars.add(escaped)
                index += 2
                continue
            if index + 2 < len(body) and body[index + 1] == "-":
                end = body[index + 2]
                chars.update(
                    chr(codepoint)
                    for codepoint in range(ord(char), ord(end) + 1)
                )
                index += 3
                continue
            chars.add(char)
            index += 1

        bounded = frozenset(char for char in chars if char in self._alphabet)
        if negated:
            return frozenset(char for char in self._alphabet if char not in bounded)
        return bounded

    def _category_escape(self, escaped: str) -> frozenset[str] | None:
        mapping = {
            "d": r"\d",
            "D": r"\D",
            "s": r"\s",
            "S": r"\S",
            "w": r"\w",
            "W": r"\W",
        }
        pattern = mapping.get(escaped)
        if pattern is None:
            return None
        return self._category_chars(pattern)


def _alphabet_for(config: AppConfig, examples: Examples) -> frozenset[str]:
    if config.synthesis.automata.alphabet_policy != "ascii_examples":
        raise AutomataUnsupportedError(
            f"unsupported automata alphabet policy: {config.synthesis.automata.alphabet_policy}"
        )
    chars = set(string.ascii_letters + string.digits + string.punctuation + " \t\n\r\f\v")
    for example in [*examples.positive, *examples.negative]:
        chars.update(example)
    return frozenset(chars)


def _state_count(nfa: Any) -> int:
    states = getattr(nfa, "states", None)
    if states is None:
        return 0
    return len(states)
