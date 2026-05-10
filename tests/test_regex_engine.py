from resketch.models import Examples, MatchMode
from resketch.regex_engine import evaluate_examples, extract_fragments, refined_repeat_variants


def test_evaluate_examples_uses_python_fullmatch() -> None:
    evaluation = evaluate_examples(
        r"\d{3}",
        Examples(positive=["123"], negative=["1234", "abc"]),
        MatchMode.FULLMATCH,
    )

    assert evaluation.success


def test_extract_fragments_uses_sre_parse() -> None:
    fragments = extract_fragments(r"[A-Z]\d{2}", max_candidate_length=100)

    assert r"[A-Z]" in fragments
    assert r"(?:\d){2}" in fragments


def test_refined_repeat_variants() -> None:
    variants = refined_repeat_variants(
        r"\d{3}",
        lower_delta=1,
        upper_delta=1,
        max_variants=10,
        max_candidate_length=100,
    )

    assert any(variant != r"\d{3}" for variant in variants)
