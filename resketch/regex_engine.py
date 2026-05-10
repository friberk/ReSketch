from __future__ import annotations

import re
import sre_constants
import sre_parse
from collections.abc import Iterable
from typing import Any, SupportsIndex, cast

from resketch.models import CandidateEvaluation, Examples, MatchMode

Token = tuple[Any, Any]


class RegexValidationError(ValueError):
    """Raised when a regex is invalid under Python regex semantics."""


def compile_regex(pattern: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as exc:
        msg = f"Invalid Python regex {pattern!r}: {exc}"
        raise RegexValidationError(msg) from exc


def is_valid_regex(pattern: str) -> bool:
    try:
        compile_regex(pattern)
    except RegexValidationError:
        return False
    return True


def regex_matches(pattern: str, value: str, mode: MatchMode) -> bool:
    compiled = compile_regex(pattern)
    if mode is MatchMode.FULLMATCH:
        return compiled.fullmatch(value) is not None
    if mode is MatchMode.MATCH:
        return compiled.match(value) is not None
    return compiled.search(value) is not None


def evaluate_examples(pattern: str, examples: Examples, mode: MatchMode) -> CandidateEvaluation:
    failures: list[str] = []
    positive_matched = 0
    negative_rejected = 0

    for example in examples.positive:
        if regex_matches(pattern, example, mode):
            positive_matched += 1
        else:
            failures.append(f"missed positive example: {example!r}")

    for example in examples.negative:
        if regex_matches(pattern, example, mode):
            failures.append(f"matched negative example: {example!r}")
        else:
            negative_rejected += 1

    return CandidateEvaluation(
        positive_total=len(examples.positive),
        positive_matched=positive_matched,
        negative_total=len(examples.negative),
        negative_rejected=negative_rejected,
        failures=failures,
    )


def extract_fragments(pattern: str, max_candidate_length: int) -> list[str]:
    parsed = _parse_tokens(pattern)
    fragments = {pattern}
    fragments.update(_extract_from_tokens(parsed))
    return sorted(
        (
            fragment
            for fragment in fragments
            if _acceptable_fragment(fragment, max_candidate_length)
        ),
        key=lambda item: (len(item), item),
    )


def refined_repeat_variants(
    pattern: str,
    lower_delta: int,
    upper_delta: int,
    max_variants: int,
    max_candidate_length: int,
) -> list[str]:
    parsed = _parse_tokens(pattern)
    variants: list[str] = []

    for index, token in enumerate(parsed):
        op, arg = token
        if op is not sre_parse.MAX_REPEAT:
            continue

        min_repeat, max_repeat, repeated = arg
        candidate_bounds = _nearby_repeat_bounds(
            min_repeat=min_repeat,
            max_repeat=max_repeat,
            lower_delta=lower_delta,
            upper_delta=upper_delta,
        )
        for new_min, new_max in candidate_bounds:
            changed = list(parsed)
            changed[index] = (op, (new_min, new_max, repeated))
            rendered = _tokens_to_regex(changed)
            if rendered is None or rendered == pattern:
                continue
            if not _acceptable_fragment(rendered, max_candidate_length):
                continue
            if rendered not in variants and is_valid_regex(rendered):
                variants.append(rendered)
            if len(variants) >= max_variants:
                return variants

    return variants


def _parse_tokens(pattern: str) -> list[Token]:
    parsed = sre_parse.parse(pattern)
    return cast(list[Token], list(cast(Any, parsed)))


def _extract_from_tokens(tokens: Iterable[Token]) -> set[str]:
    fragments: set[str] = set()
    for op, arg in tokens:
        rendered = _token_to_regex(op, arg)
        if rendered is not None:
            fragments.add(rendered)
        if op is sre_parse.MAX_REPEAT:
            _, _, repeated = arg
            fragments.update(_extract_from_tokens(repeated))
        elif op is sre_parse.SUBPATTERN:
            _, _, _, nested = arg
            fragments.update(_extract_from_tokens(nested))
        elif op is sre_parse.BRANCH:
            _, branches = arg
            for branch in branches:
                fragments.update(_extract_from_tokens(branch))
    return fragments


def _tokens_to_regex(tokens: Iterable[Token]) -> str | None:
    rendered = [_token_to_regex(op, arg) for op, arg in tokens]
    if any(fragment is None for fragment in rendered):
        return None
    return "".join(fragment for fragment in rendered if fragment is not None)


def _token_to_regex(op: Any, arg: Any) -> str | None:
    if op is sre_parse.LITERAL:
        return re.escape(chr(cast(SupportsIndex, arg)))
    if op is sre_parse.ANY:
        return "."
    if op is sre_parse.CATEGORY:
        return _category_to_regex(arg)
    if op is sre_parse.IN:
        return _character_class_to_regex(arg)
    if op is sre_parse.MAX_REPEAT:
        min_repeat, max_repeat, repeated = arg
        inner = _tokens_to_regex(repeated)
        if inner is None:
            return None
        return f"(?:{inner}){_quantifier_to_regex(min_repeat, max_repeat)}"
    if op is sre_parse.SUBPATTERN:
        _, _, _, nested = arg
        inner = _tokens_to_regex(nested)
        if inner is None:
            return None
        return f"(?:{inner})"
    if op is sre_parse.BRANCH:
        _, branches = arg
        rendered_branches = [_tokens_to_regex(branch) for branch in branches]
        if any(branch is None for branch in rendered_branches):
            return None
        return "(?:" + "|".join(branch for branch in rendered_branches if branch is not None) + ")"
    if op is sre_parse.AT:
        return _anchor_to_regex(arg)
    return None


def _category_to_regex(category: Any) -> str | None:
    mapping: dict[Any, str] = {
        sre_parse.CATEGORY_DIGIT: r"\d",
        sre_parse.CATEGORY_NOT_DIGIT: r"\D",
        sre_parse.CATEGORY_SPACE: r"\s",
        sre_parse.CATEGORY_NOT_SPACE: r"\S",
        sre_parse.CATEGORY_WORD: r"\w",
        sre_parse.CATEGORY_NOT_WORD: r"\W",
    }
    return mapping.get(category)


def _anchor_to_regex(anchor: Any) -> str | None:
    mapping: dict[Any, str] = {
        sre_parse.AT_BEGINNING: "^",
        sre_parse.AT_BEGINNING_STRING: r"\A",
        sre_parse.AT_END: "$",
        sre_parse.AT_END_STRING: r"\Z",
    }
    return mapping.get(anchor)


def _character_class_to_regex(items: Iterable[Token]) -> str | None:
    item_list = list(items)
    if len(item_list) == 1 and item_list[0][0] is sre_parse.CATEGORY:
        return _category_to_regex(item_list[0][1])

    parts: list[str] = []
    negate = False
    for op, arg in item_list:
        if op is sre_parse.NEGATE:
            negate = True
        elif op is sre_parse.LITERAL:
            parts.append(_escape_class_char(chr(cast(SupportsIndex, arg))))
        elif op is sre_parse.RANGE:
            start, end = arg
            start_char = _escape_class_char(chr(cast(SupportsIndex, start)))
            end_char = _escape_class_char(chr(cast(SupportsIndex, end)))
            parts.append(f"{start_char}-{end_char}")
        elif op is sre_parse.CATEGORY:
            category = _category_to_regex(arg)
            if category is None:
                return None
            parts.append(category)
        else:
            return None
    prefix = "^" if negate else ""
    return f"[{prefix}{''.join(parts)}]"


def _escape_class_char(value: str) -> str:
    if value in {"\\", "]", "^", "-"}:
        return "\\" + value
    return value


def _quantifier_to_regex(min_repeat: int, max_repeat: int) -> str:
    if min_repeat == 0 and max_repeat == sre_constants.MAXREPEAT:
        return "*"
    if min_repeat == 1 and max_repeat == sre_constants.MAXREPEAT:
        return "+"
    if min_repeat == 0 and max_repeat == 1:
        return "?"
    if max_repeat == sre_constants.MAXREPEAT:
        return f"{{{min_repeat},}}"
    if min_repeat == max_repeat:
        return f"{{{min_repeat}}}"
    return f"{{{min_repeat},{max_repeat}}}"


def _nearby_repeat_bounds(
    min_repeat: int,
    max_repeat: int,
    lower_delta: int,
    upper_delta: int,
) -> list[tuple[int, int]]:
    lower_floor = max(0, min_repeat - lower_delta)
    finite_max = min_repeat + upper_delta if max_repeat == sre_constants.MAXREPEAT else max_repeat
    upper_ceiling = finite_max + upper_delta

    bounds: list[tuple[int, int]] = []
    for new_min in range(lower_floor, min_repeat + lower_delta + 1):
        for new_max in range(max(new_min, finite_max - upper_delta), upper_ceiling + 1):
            bounds.append((new_min, new_max))
    return bounds


def _acceptable_fragment(fragment: str, max_candidate_length: int) -> bool:
    return bool(fragment) and len(fragment) <= max_candidate_length and is_valid_regex(fragment)
