from __future__ import annotations

import re
import sre_constants
import sre_parse
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, SupportsIndex, TypeAlias, cast

from resketch.models import Hole

PRIVATE_USE_START = 0xE000
PRIVATE_USE_END = 0xF8FF


class SketchParseError(ValueError):
    """Raised when a semantic sketch cannot be parsed."""


@dataclass(frozen=True)
class SketchLiteral:
    value: str


@dataclass(frozen=True)
class SketchDot:
    pass


@dataclass(frozen=True)
class SketchCategory:
    pattern: str


@dataclass(frozen=True)
class SketchCharClass:
    body: str
    negated: bool = False


@dataclass(frozen=True)
class SketchAnchor:
    pattern: str


@dataclass(frozen=True)
class SketchConcat:
    parts: tuple[SketchNode, ...]


@dataclass(frozen=True)
class SketchAlt:
    options: tuple[SketchNode, ...]


@dataclass(frozen=True)
class SketchRepeat:
    child: SketchNode
    min_repeat: int
    max_repeat: int | None


@dataclass(frozen=True)
class SketchHole:
    hole: Hole


SketchNode: TypeAlias = (
    SketchLiteral
    | SketchDot
    | SketchCategory
    | SketchCharClass
    | SketchAnchor
    | SketchConcat
    | SketchAlt
    | SketchRepeat
    | SketchHole
)
Token: TypeAlias = tuple[Any, Any]


@dataclass(frozen=True)
class Sketch:
    source: str
    root: SketchNode
    holes: tuple[Hole, ...]

    def render(self, assignments: dict[str, str]) -> str:
        return render_sketch_node(self.root, assignments)

    def render_partial(self, assignments: dict[str, str], unknown_pattern: str) -> str:
        return render_sketch_node_partial(self.root, assignments, unknown_pattern)


def parse_sketch(source: str) -> Sketch:
    _reject_internal_placeholders(source)
    transformed, holes, placeholder_to_hole = _replace_holes_with_placeholders(source)
    try:
        root = _tokens_to_node(_parse_tokens(transformed), placeholder_to_hole)
    except re.error as exc:
        msg = f"Invalid regular sketch {source!r}: {exc}"
        raise SketchParseError(msg) from exc
    if root is None:
        msg = f"Unsupported regular sketch syntax: {source!r}"
        raise SketchParseError(msg)
    return Sketch(source=source, root=root, holes=tuple(holes))


def render_sketch_node(node: SketchNode, assignments: dict[str, str]) -> str:
    if isinstance(node, SketchLiteral):
        return re.escape(node.value)
    if isinstance(node, SketchDot):
        return "."
    if isinstance(node, SketchCategory):
        return node.pattern
    if isinstance(node, SketchCharClass):
        return f"[{'^' if node.negated else ''}{node.body}]"
    if isinstance(node, SketchAnchor):
        return node.pattern
    if isinstance(node, SketchHole):
        if node.hole.identifier not in assignments:
            msg = f"Missing assignment for hole {node.hole.identifier}"
            raise KeyError(msg)
        return assignments[node.hole.identifier]
    if isinstance(node, SketchConcat):
        return "".join(render_sketch_node(part, assignments) for part in node.parts)
    if isinstance(node, SketchAlt):
        rendered_options = "|".join(
            render_sketch_node(option, assignments)
            for option in node.options
        )
        return f"(?:{rendered_options})"
    if isinstance(node, SketchRepeat):
        return (
            f"{_render_repeat_child(node.child, assignments)}"
            f"{_quantifier_to_regex(node.min_repeat, node.max_repeat)}"
        )


def render_sketch_node_partial(
    node: SketchNode,
    assignments: dict[str, str],
    unknown_pattern: str,
) -> str:
    if isinstance(node, SketchLiteral):
        return re.escape(node.value)
    if isinstance(node, SketchDot):
        return "."
    if isinstance(node, SketchCategory):
        return node.pattern
    if isinstance(node, SketchCharClass):
        return f"[{'^' if node.negated else ''}{node.body}]"
    if isinstance(node, SketchAnchor):
        return node.pattern
    if isinstance(node, SketchHole):
        return assignments.get(node.hole.identifier, unknown_pattern)
    if isinstance(node, SketchConcat):
        return "".join(
            render_sketch_node_partial(part, assignments, unknown_pattern)
            for part in node.parts
        )
    if isinstance(node, SketchAlt):
        rendered_options = "|".join(
            render_sketch_node_partial(option, assignments, unknown_pattern)
            for option in node.options
        )
        return f"(?:{rendered_options})"
    if isinstance(node, SketchRepeat):
        return (
            f"{_render_repeat_child_partial(node.child, assignments, unknown_pattern)}"
            f"{_quantifier_to_regex(node.min_repeat, node.max_repeat)}"
        )


def _replace_holes_with_placeholders(
    source: str,
) -> tuple[str, list[Hole], dict[str, Hole]]:
    holes: list[Hole] = []
    placeholder_to_hole: dict[str, Hole] = {}
    pieces: list[str] = []
    cursor = 0

    while cursor < len(source):
        marker = _hole_marker_at(source, cursor)
        if marker is None:
            pieces.append(source[cursor])
            cursor += 1
            continue

        if _is_inside_character_class(source, cursor):
            msg = f"Typed holes are not supported inside character classes at offset {cursor}"
            raise SketchParseError(msg)

        type_start = cursor + len(marker)
        type_end = source.find("}", type_start)
        if type_end < 0:
            msg = f"Unclosed typed hole at offset {cursor}"
            raise SketchParseError(msg)

        semantic_type = source[type_start:type_end].strip()
        if not semantic_type:
            msg = f"Empty semantic type at offset {cursor}"
            raise SketchParseError(msg)

        placeholder = _placeholder_for_index(len(holes))
        hole = Hole(
            identifier=f"h{len(holes)}",
            semantic_type=semantic_type,
            start=cursor,
            end=type_end + 1,
        )
        holes.append(hole)
        placeholder_to_hole[placeholder] = hole
        pieces.append(placeholder)
        cursor = type_end + 1

    return "".join(pieces), holes, placeholder_to_hole


def _tokens_to_node(
    tokens: Iterable[Token],
    placeholder_to_hole: dict[str, Hole],
) -> SketchNode | None:
    nodes = [_token_to_node(op, arg, placeholder_to_hole) for op, arg in tokens]
    if any(node is None for node in nodes):
        return None
    typed_nodes = tuple(node for node in nodes if node is not None)
    if not typed_nodes:
        return SketchLiteral("")
    if len(typed_nodes) == 1:
        return typed_nodes[0]
    return SketchConcat(typed_nodes)


def _token_to_node(
    op: Any,
    arg: Any,
    placeholder_to_hole: dict[str, Hole],
) -> SketchNode | None:
    if op is sre_parse.LITERAL:
        value = chr(cast(SupportsIndex, arg))
        hole = placeholder_to_hole.get(value)
        if hole is not None:
            return SketchHole(hole)
        return SketchLiteral(value)
    if op is sre_parse.ANY:
        return SketchDot()
    if op is sre_parse.CATEGORY:
        return _category_to_node(arg)
    if op is sre_parse.IN:
        return _character_class_to_node(arg, placeholder_to_hole)
    if op is sre_parse.MAX_REPEAT or op is sre_parse.MIN_REPEAT:
        min_repeat, max_repeat, repeated = arg
        child = _tokens_to_node(cast(Iterable[Token], repeated), placeholder_to_hole)
        if child is None:
            return None
        return SketchRepeat(
            child=child,
            min_repeat=int(min_repeat),
            max_repeat=None if max_repeat == sre_constants.MAXREPEAT else int(max_repeat),
        )
    if op is sre_parse.SUBPATTERN:
        _, _, _, nested = arg
        return _tokens_to_node(cast(Iterable[Token], nested), placeholder_to_hole)
    if op is sre_parse.BRANCH:
        _, branches = arg
        options = [
            _tokens_to_node(cast(Iterable[Token], branch), placeholder_to_hole)
            for branch in branches
        ]
        if any(option is None for option in options):
            return None
        return SketchAlt(tuple(option for option in options if option is not None))
    if op is sre_parse.AT:
        return _anchor_to_node(arg)
    return None


def _category_to_node(category: Any) -> SketchCategory | None:
    mapping: dict[Any, str] = {
        sre_parse.CATEGORY_DIGIT: r"\d",
        sre_parse.CATEGORY_NOT_DIGIT: r"\D",
        sre_parse.CATEGORY_SPACE: r"\s",
        sre_parse.CATEGORY_NOT_SPACE: r"\S",
        sre_parse.CATEGORY_WORD: r"\w",
        sre_parse.CATEGORY_NOT_WORD: r"\W",
    }
    pattern = mapping.get(category)
    return SketchCategory(pattern) if pattern is not None else None


def _anchor_to_node(anchor: Any) -> SketchAnchor | None:
    mapping: dict[Any, str] = {
        sre_parse.AT_BEGINNING: "^",
        sre_parse.AT_BEGINNING_STRING: r"\A",
        sre_parse.AT_END: "$",
        sre_parse.AT_END_STRING: r"\Z",
    }
    pattern = mapping.get(anchor)
    return SketchAnchor(pattern) if pattern is not None else None


def _character_class_to_node(
    items: Iterable[Token],
    placeholder_to_hole: dict[str, Hole],
) -> SketchNode | None:
    item_list = list(items)
    if any(
        op is sre_parse.LITERAL and chr(cast(SupportsIndex, arg)) in placeholder_to_hole
        for op, arg in item_list
    ):
        return _optimized_alternation_to_node(item_list, placeholder_to_hole)

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
    return SketchCharClass(body="".join(body_parts), negated=negated)


def _optimized_alternation_to_node(
    items: Iterable[Token],
    placeholder_to_hole: dict[str, Hole],
) -> SketchNode | None:
    options: list[SketchNode] = []
    for op, arg in items:
        if op is sre_parse.LITERAL:
            value = chr(cast(SupportsIndex, arg))
            hole = placeholder_to_hole.get(value)
            options.append(SketchHole(hole) if hole is not None else SketchLiteral(value))
        elif op is sre_parse.CATEGORY:
            category = _category_to_node(arg)
            if category is None:
                return None
            options.append(category)
        elif op is sre_parse.RANGE:
            start, end = arg
            options.append(
                SketchCharClass(
                    body=(
                        f"{_escape_class_char(chr(cast(SupportsIndex, start)))}-"
                        f"{_escape_class_char(chr(cast(SupportsIndex, end)))}"
                    )
                )
            )
        else:
            return None
    return SketchAlt(tuple(options))


def _render_repeat_child(node: SketchNode, assignments: dict[str, str]) -> str:
    if isinstance(
        node,
        SketchLiteral | SketchDot | SketchCategory | SketchCharClass | SketchAnchor,
    ):
        return render_sketch_node(node, assignments)
    return f"(?:{render_sketch_node(node, assignments)})"


def _render_repeat_child_partial(
    node: SketchNode,
    assignments: dict[str, str],
    unknown_pattern: str,
) -> str:
    if isinstance(
        node,
        SketchLiteral | SketchDot | SketchCategory | SketchCharClass | SketchAnchor,
    ):
        return render_sketch_node_partial(node, assignments, unknown_pattern)
    return f"(?:{render_sketch_node_partial(node, assignments, unknown_pattern)})"


def _parse_tokens(pattern: str) -> list[Token]:
    parsed = sre_parse.parse(pattern)
    return cast(list[Token], list(cast(Any, parsed)))


def _hole_marker_at(source: str, offset: int) -> str | None:
    for marker in ("{□:", "{hole:", "{?:"):
        if source.startswith(marker, offset):
            return marker
    return None


def _is_inside_character_class(source: str, offset: int) -> bool:
    escaped = False
    in_class = False
    for char in source[:offset]:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "[":
            in_class = True
        elif char == "]":
            in_class = False
    return in_class


def _reject_internal_placeholders(source: str) -> None:
    for char in source:
        if PRIVATE_USE_START <= ord(char) <= PRIVATE_USE_END:
            msg = "Sketch source cannot contain private-use placeholder characters"
            raise SketchParseError(msg)


def _placeholder_for_index(index: int) -> str:
    codepoint = PRIVATE_USE_START + index
    if codepoint > PRIVATE_USE_END:
        msg = "Too many sketch holes for private-use placeholder encoding"
        raise SketchParseError(msg)
    return chr(codepoint)


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
