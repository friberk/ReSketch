from resketch.models import MatchMode
from resketch.regex_ast import (
    Alt,
    Category,
    CharClass,
    Concat,
    Repeat,
    extract_subnodes,
    node_size,
    parse_regex_to_ast,
    render_regex,
)
from resketch.regex_engine import regex_matches


def test_parse_render_supported_regex_ast() -> None:
    node = parse_regex_to_ast(r"[A-Z]\d{2}", allow_raw_regex=True)

    assert isinstance(node, Concat)
    assert render_regex(node) == r"[A-Z]\d{2}"
    assert regex_matches(render_regex(node), "A12", MatchMode.FULLMATCH)


def test_parse_render_alternation_and_subnodes() -> None:
    node = parse_regex_to_ast(r"(?:ab|cd)", allow_raw_regex=True)
    subnodes = extract_subnodes(node)

    assert isinstance(node, Alt)
    assert render_regex(node) == r"(?:ab|cd)"
    assert node_size(node) > 1
    assert any(render_regex(subnode) == "ab" for subnode in subnodes)


def test_parse_render_repeat_category_and_charclass() -> None:
    digit = parse_regex_to_ast(r"\d{3,4}", allow_raw_regex=True)
    letters = parse_regex_to_ast(r"[A-Za-z]", allow_raw_regex=True)

    assert isinstance(digit, Repeat)
    assert isinstance(digit.child, Category)
    assert isinstance(letters, CharClass)
    assert render_regex(digit) == r"\d{3,4}"


def test_parse_normalizes_nested_alternation() -> None:
    node = parse_regex_to_ast(r"(?:ab|(?:cd|ab))", allow_raw_regex=True)

    assert isinstance(node, Alt)
    assert render_regex(node) == r"(?:ab|cd)"
