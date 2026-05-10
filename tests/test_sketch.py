import pytest

from resketch.sketch import SketchAlt, SketchParseError, SketchRepeat, parse_sketch


def test_parse_sketch_preserves_regex_quantifiers() -> None:
    sketch = parse_sketch(r"{□: credit_card}\s+CVV:\d{3}")

    assert len(sketch.holes) == 1
    assert sketch.holes[0].semantic_type == "credit_card"
    assert sketch.render({"h0": r"\d{16}"}) == r"\d{16}\s+CVV:\d{3}"


def test_parse_sketch_rejects_unclosed_hole() -> None:
    with pytest.raises(SketchParseError):
        parse_sketch("{□: email_address")


def test_parse_sketch_supports_nested_repeat_context() -> None:
    sketch = parse_sketch(r"{□: digit}{3}")

    assert len(sketch.holes) == 1
    assert isinstance(sketch.root, SketchRepeat)
    assert sketch.render({"h0": r"\d"}) == r"(?:\d){3}"


def test_parse_sketch_supports_optimized_alternation_context() -> None:
    sketch = parse_sketch(r"(?:{□: country}|{□: state})")

    assert len(sketch.holes) == 2
    assert isinstance(sketch.root, SketchAlt)
    assert sketch.render({"h0": "US", "h1": "CA"}) == r"(?:US|CA)"


def test_parse_sketch_supports_optional_nested_context() -> None:
    sketch = parse_sketch(r"{□: year}-(?:{□: month}-{□: day})?")

    assert [hole.semantic_type for hole in sketch.holes] == ["year", "month", "day"]
    assert sketch.render({"h0": r"\d{4}", "h1": r"\d{2}", "h2": r"\d{2}"}) == (
        r"\d{4}\-(?:\d{2}\-\d{2})?"
    )


def test_parse_sketch_rejects_unsupported_nonregular_constructs() -> None:
    with pytest.raises(SketchParseError):
        parse_sketch(r"(?={□: digit})\d")

    with pytest.raises(SketchParseError):
        parse_sketch(r"(.)\1")


def test_parse_sketch_rejects_holes_inside_character_classes() -> None:
    with pytest.raises(SketchParseError):
        parse_sketch(r"[{□: digit}]")
