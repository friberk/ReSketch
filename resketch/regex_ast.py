from __future__ import annotations

import re
import sre_constants
import sre_parse
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, SupportsIndex, TypeAlias, cast

from resketch.regex_engine import RegexValidationError, compile_regex


@dataclass(frozen=True)
class Literal:
    value: str


@dataclass(frozen=True)
class Dot:
    pass


@dataclass(frozen=True)
class Category:
    pattern: str


@dataclass(frozen=True)
class CharClass:
    body: str
    negated: bool = False


@dataclass(frozen=True)
class Anchor:
    pattern: str


@dataclass(frozen=True)
class Concat:
    parts: tuple[RegexNode, ...]


@dataclass(frozen=True)
class Alt:
    options: tuple[RegexNode, ...]


@dataclass(frozen=True)
class Repeat:
    child: RegexNode
    min_repeat: int
    max_repeat: int | None


@dataclass(frozen=True)
class RawRegex:
    pattern: str


RegexNode: TypeAlias = (
    Literal | Dot | Category | CharClass | Anchor | Concat | Alt | Repeat | RawRegex
)
Token: TypeAlias = tuple[Any, Any]


def parse_regex_to_ast(pattern: str, *, allow_raw_regex: bool) -> RegexNode:
    try:
        tokens = _parse_tokens(pattern)
        node = _tokens_to_node(tokens)
        if node is None:
            raise RegexValidationError(f"Unsupported regex syntax: {pattern!r}")
        node = normalize_node(node)
        compile_regex(render_regex(node))
        return node
    except RegexValidationError:
        if allow_raw_regex:
            compile_regex(pattern)
            return RawRegex(pattern)
        raise
    except re.error as exc:
        msg = f"Invalid Python regex {pattern!r}: {exc}"
        raise RegexValidationError(msg) from exc


def render_regex(node: RegexNode) -> str:
    if isinstance(node, Literal):
        return re.escape(node.value)
    if isinstance(node, Dot):
        return "."
    if isinstance(node, Category):
        return node.pattern
    if isinstance(node, CharClass):
        return f"[{'^' if node.negated else ''}{node.body}]"
    if isinstance(node, Anchor):
        return node.pattern
    if isinstance(node, RawRegex):
        return node.pattern
    if isinstance(node, Concat):
        return "".join(_render_for_concat(part) for part in node.parts)
    if isinstance(node, Alt):
        return "(?:" + "|".join(render_regex(option) for option in node.options) + ")"
    if isinstance(node, Repeat):
        quantifier = _quantifier_to_regex(node.min_repeat, node.max_repeat)
        return f"{_render_repeat_child(node.child)}{quantifier}"


def node_size(node: RegexNode) -> int:
    if isinstance(node, Concat):
        return 1 + sum(node_size(part) for part in node.parts)
    if isinstance(node, Alt):
        return 1 + sum(node_size(option) for option in node.options)
    if isinstance(node, Repeat):
        return 1 + node_size(node.child)
    return 1


def extract_subnodes(node: RegexNode) -> list[RegexNode]:
    seen: set[str] = set()
    ordered: list[RegexNode] = []

    def visit(current: RegexNode) -> None:
        rendered = render_regex(current)
        if rendered not in seen:
            seen.add(rendered)
            ordered.append(current)
        if isinstance(current, Concat):
            for part in current.parts:
                visit(part)
        elif isinstance(current, Alt):
            for option in current.options:
                visit(option)
        elif isinstance(current, Repeat):
            visit(current.child)

    visit(node)
    return ordered


def normalize_node(node: RegexNode) -> RegexNode:
    if isinstance(node, Concat):
        parts: list[RegexNode] = []
        literal_buffer: list[str] = []
        for part in node.parts:
            normalized = normalize_node(part)
            nested_parts = normalized.parts if isinstance(normalized, Concat) else (normalized,)
            for nested in nested_parts:
                if isinstance(nested, Literal):
                    literal_buffer.append(nested.value)
                    continue
                if literal_buffer:
                    value = "".join(literal_buffer)
                    if value:
                        parts.append(Literal(value))
                    literal_buffer = []
                if not (isinstance(nested, Literal) and nested.value == ""):
                    parts.append(nested)
        if literal_buffer:
            value = "".join(literal_buffer)
            if value:
                parts.append(Literal(value))
        if not parts:
            return Literal("")
        if len(parts) == 1:
            return parts[0]
        return Concat(tuple(parts))
    if isinstance(node, Alt):
        options: dict[str, RegexNode] = {}
        for option in node.options:
            normalized = normalize_node(option)
            if isinstance(normalized, Alt):
                for nested in normalized.options:
                    options[render_regex(nested)] = nested
            else:
                options[render_regex(normalized)] = normalized
        if not options:
            return Literal("")
        if len(options) == 1:
            return next(iter(options.values()))
        return Alt(tuple(options[key] for key in sorted(options)))
    if isinstance(node, Repeat):
        child = normalize_node(node.child)
        if node.min_repeat == 1 and node.max_repeat == 1:
            return child
        if node.min_repeat == 0 and node.max_repeat == 0:
            return Literal("")
        return Repeat(child=child, min_repeat=node.min_repeat, max_repeat=node.max_repeat)
    return node


def make_concat(left: RegexNode, right: RegexNode) -> RegexNode:
    parts: list[RegexNode] = []
    if isinstance(left, Concat):
        parts.extend(left.parts)
    else:
        parts.append(left)
    if isinstance(right, Concat):
        parts.extend(right.parts)
    else:
        parts.append(right)
    return normalize_node(Concat(tuple(parts)))


def make_alt(left: RegexNode, right: RegexNode) -> RegexNode:
    options: list[RegexNode] = []
    if isinstance(left, Alt):
        options.extend(left.options)
    else:
        options.append(left)
    if isinstance(right, Alt):
        options.extend(right.options)
    else:
        options.append(right)
    deduped = {render_regex(option): option for option in options}
    return normalize_node(Alt(tuple(deduped[key] for key in sorted(deduped))))


def is_broad_wildcard(node: RegexNode) -> bool:
    if isinstance(node, Dot):
        return True
    return (
        isinstance(node, Repeat)
        and isinstance(node.child, Dot)
        and node.min_repeat == 0
        and node.max_repeat is None
    )


def _render_for_concat(node: RegexNode) -> str:
    if isinstance(node, Alt):
        return render_regex(node)
    return render_regex(node)


def _render_repeat_child(node: RegexNode) -> str:
    if isinstance(node, Literal | Dot | Category | CharClass | Anchor | RawRegex):
        return render_regex(node)
    return f"(?:{render_regex(node)})"


def _parse_tokens(pattern: str) -> list[Token]:
    parsed = sre_parse.parse(pattern)
    return cast(list[Token], list(cast(Any, parsed)))


def _tokens_to_node(tokens: Iterable[Token]) -> RegexNode | None:
    nodes = [_token_to_node(op, arg) for op, arg in tokens]
    if any(node is None for node in nodes):
        return None
    typed_nodes = tuple(node for node in nodes if node is not None)
    if not typed_nodes:
        return Literal("")
    if len(typed_nodes) == 1:
        return typed_nodes[0]
    return Concat(typed_nodes)


def _token_to_node(op: Any, arg: Any) -> RegexNode | None:
    if op is sre_parse.LITERAL:
        return Literal(chr(cast(SupportsIndex, arg)))
    if op is sre_parse.ANY:
        return Dot()
    if op is sre_parse.CATEGORY:
        return _category_to_node(arg)
    if op is sre_parse.IN:
        return _character_class_to_node(arg)
    if op is sre_parse.MAX_REPEAT or op is sre_parse.MIN_REPEAT:
        min_repeat, max_repeat, repeated = arg
        child = _tokens_to_node(cast(Iterable[Token], repeated))
        if child is None:
            return None
        return Repeat(
            child=child,
            min_repeat=int(min_repeat),
            max_repeat=None if max_repeat == sre_constants.MAXREPEAT else int(max_repeat),
        )
    if op is sre_parse.SUBPATTERN:
        _, _, _, nested = arg
        return _tokens_to_node(cast(Iterable[Token], nested))
    if op is sre_parse.BRANCH:
        _, branches = arg
        options = [_tokens_to_node(cast(Iterable[Token], branch)) for branch in branches]
        if any(option is None for option in options):
            return None
        return Alt(tuple(option for option in options if option is not None))
    if op is sre_parse.AT:
        return _anchor_to_node(arg)
    return None


def _category_to_node(category: Any) -> Category | None:
    mapping: dict[Any, str] = {
        sre_parse.CATEGORY_DIGIT: r"\d",
        sre_parse.CATEGORY_NOT_DIGIT: r"\D",
        sre_parse.CATEGORY_SPACE: r"\s",
        sre_parse.CATEGORY_NOT_SPACE: r"\S",
        sre_parse.CATEGORY_WORD: r"\w",
        sre_parse.CATEGORY_NOT_WORD: r"\W",
    }
    pattern = mapping.get(category)
    return Category(pattern) if pattern is not None else None


def _anchor_to_node(anchor: Any) -> Anchor | None:
    mapping: dict[Any, str] = {
        sre_parse.AT_BEGINNING: "^",
        sre_parse.AT_BEGINNING_STRING: r"\A",
        sre_parse.AT_END: "$",
        sre_parse.AT_END_STRING: r"\Z",
    }
    pattern = mapping.get(anchor)
    return Anchor(pattern) if pattern is not None else None


def _character_class_to_node(items: Iterable[Token]) -> RegexNode | None:
    item_list = list(items)
    if len(item_list) == 1 and item_list[0][0] is sre_parse.CATEGORY:
        return _category_to_node(item_list[0][1])

    body_parts: list[str] = []
    negated = False
    for op, arg in item_list:
        if op is sre_parse.NEGATE:
            negated = True
        elif op is sre_parse.LITERAL:
            body_parts.append(_escape_class_char(chr(cast(SupportsIndex, arg))))
        elif op is sre_parse.RANGE:
            start, end = arg
            body_parts.append(
                f"{_escape_class_char(chr(cast(SupportsIndex, start)))}-"
                f"{_escape_class_char(chr(cast(SupportsIndex, end)))}"
            )
        elif op is sre_parse.CATEGORY:
            category = _category_to_node(arg)
            if category is None:
                return None
            body_parts.append(category.pattern)
        else:
            return None
    return CharClass(body="".join(body_parts), negated=negated)


def _escape_class_char(value: str) -> str:
    if value in {"\\", "]", "^", "-"}:
        return "\\" + value
    return value


def _quantifier_to_regex(min_repeat: int, max_repeat: int | None) -> str:
    if min_repeat == 0 and max_repeat is None:
        return "*"
    if min_repeat == 1 and max_repeat is None:
        return "+"
    if min_repeat == 0 and max_repeat == 1:
        return "?"
    if max_repeat is None:
        return f"{{{min_repeat},}}"
    if min_repeat == max_repeat:
        return f"{{{min_repeat}}}"
    return f"{{{min_repeat},{max_repeat}}}"
