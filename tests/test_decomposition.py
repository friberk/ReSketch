from resketch.config import load_config
from resketch.decomposition import decompose_examples
from resketch.models import DecompositionMode, EvidenceKind, Examples
from resketch.sketch import parse_sketch


def test_decomposition_infers_single_separated_hole(load_fixture_config) -> None:
    config = load_fixture_config(provider="fixture")
    sketch = parse_sketch("CVV:{□: cvv}")

    result = decompose_examples(
        config,
        sketch,
        Examples(positive=["CVV:123"], negative=["CVV:12"]),
    )

    assert result.by_hole["h0"].hard.positive == ["123"]
    assert result.by_hole["h0"].hard.negative == ["12"]
    assert result.by_hole["h0"].evidence[0].kind is EvidenceKind.HARD_POSITIVE
    assert result.stats.hard_evidence_count == 1
    assert result.stats.inferred_negative_count == 1
    assert result.stats.decomposition_success_rate == 1.0


def test_decomposition_infers_multiple_separated_holes(load_fixture_config) -> None:
    config = load_fixture_config(provider="fixture")
    sketch = parse_sketch(r"{□: year}-{□: integer}")

    result = decompose_examples(config, sketch, Examples(positive=["2026-42"]))

    assert result.by_hole["h0"].hard.positive == ["2026"]
    assert result.by_hole["h1"].hard.positive == ["42"]


def test_decomposition_keeps_adjacent_holes_soft(load_fixture_config) -> None:
    config = load_fixture_config(provider="fixture")
    sketch = parse_sketch("{□: first}{□: second}")

    result = decompose_examples(config, sketch, Examples(positive=["abc"]))

    assert result.by_hole["h0"].hard.positive == []
    assert result.by_hole["h1"].hard.positive == []
    assert result.by_hole["h0"].soft_positive == ["a", "ab"]
    assert result.by_hole["h1"].soft_positive == ["bc", "c"]
    assert result.ambiguity_groups
    assert result.constraint_set.ambiguity_groups == result.ambiguity_groups
    assert result.stats.ambiguous_example_count == 1
    assert result.stats.ambiguity_group_count == 1
    assert result.stats.soft_evidence_count == 4


def test_decomposition_merges_explicit_hole_examples(load_fixture_config) -> None:
    config = load_fixture_config(provider="fixture")
    sketch = parse_sketch("{□: integer}")

    result = decompose_examples(
        config,
        sketch,
        Examples(positive=["42"]),
        explicit_hole_examples={"h0": Examples(positive=["7"], negative=["3.14"])},
    )

    assert result.by_hole["h0"].hard.positive == ["42", "7"]
    assert result.by_hole["h0"].hard.negative == ["3.14"]
    assert result.stats.explicit_evidence_count == 2


def test_hard_only_mode_excludes_soft_evidence() -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "decomposition.mode": DecompositionMode.HARD_ONLY.value,
        }
    )
    sketch = parse_sketch("{□: first}{□: second}")

    result = decompose_examples(config, sketch, Examples(positive=["abc"]))

    assert result.by_hole["h0"].hard.positive == []
    assert result.by_hole["h0"].soft_positive == []
    assert result.stats.ambiguous_example_count == 1
    assert result.stats.soft_evidence_count == 0


def test_explicit_only_mode_ignores_inferred_evidence() -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "decomposition.mode": DecompositionMode.EXPLICIT_ONLY.value,
        }
    )
    sketch = parse_sketch("{□: integer}")

    result = decompose_examples(
        config,
        sketch,
        Examples(positive=["42"]),
        explicit_hole_examples={"h0": Examples(positive=["7"])},
    )

    assert result.by_hole["h0"].hard.positive == ["7"]
    assert [evidence.kind for evidence in result.by_hole["h0"].evidence] == [
        EvidenceKind.EXPLICIT_POSITIVE
    ]
    assert result.stats.matched_positive_count == 0


def test_off_mode_ignores_explicit_and_inferred_evidence() -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "decomposition.mode": DecompositionMode.OFF.value,
        }
    )
    sketch = parse_sketch("{□: integer}")

    result = decompose_examples(
        config,
        sketch,
        Examples(positive=["42"]),
        explicit_hole_examples={"h0": Examples(positive=["7"])},
    )

    assert result.by_hole["h0"].hard.positive == []
    assert result.by_hole["h0"].evidence == []
    assert result.diagnostics == ["decomposition disabled"]


def test_unmatched_positive_is_tracked(load_fixture_config) -> None:
    config = load_fixture_config(provider="fixture")
    sketch = parse_sketch("ID:{□: integer}")

    result = decompose_examples(config, sketch, Examples(positive=["no id here"]))

    assert result.unmatched_positive == ["no id here"]
    assert result.stats.unmatched_positive_count == 1


def test_repeated_hole_yields_multiple_hard_witnesses(load_fixture_config) -> None:
    config = load_fixture_config(provider="fixture")
    sketch = parse_sketch(r"{□: digit}{3}")

    result = decompose_examples(config, sketch, Examples(positive=["123"]))

    assert result.by_hole["h0"].hard.positive == ["1", "2", "3"]
    assert result.repeated_hole_constraints
    assert result.repeated_hole_constraints[0].hole_id == "h0"
    assert result.repeated_hole_constraints[0].occurrence_count == 3
    assert result.stats.hard_evidence_count == 3
    assert result.stats.repeated_hole_constraint_count == 1


def test_alternative_branch_holes_are_soft_when_branch_is_ambiguous(load_fixture_config) -> None:
    config = load_fixture_config(provider="fixture")
    sketch = parse_sketch(r"(?:{□: country}|{□: state})")

    result = decompose_examples(config, sketch, Examples(positive=["US"]))

    assert result.by_hole["h0"].hard.positive == []
    assert result.by_hole["h1"].hard.positive == []
    assert result.by_hole["h0"].soft_positive == ["US"]
    assert result.by_hole["h1"].soft_positive == ["US"]
    assert result.stats.ambiguous_example_count == 1


def test_optional_nested_holes_are_hard_when_context_is_unique(load_fixture_config) -> None:
    config = load_fixture_config(provider="fixture")
    sketch = parse_sketch(r"{□: year}-(?:{□: month}-{□: day})?")

    result = decompose_examples(config, sketch, Examples(positive=["2026-04-27"]))

    assert result.by_hole["h0"].hard.positive == ["2026"]
    assert result.by_hole["h1"].hard.positive == ["04"]
    assert result.by_hole["h2"].hard.positive == ["27"]


def test_decomposition_reports_bounded_assignment_truncation() -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "decomposition.max_assignments_per_example": 1,
        }
    )
    sketch = parse_sketch(r"{□: first}{□: second}")

    result = decompose_examples(config, sketch, Examples(positive=["abc"]))

    assert result.stats.truncated_assignment_count > 0
    assert any("truncated" in diagnostic for diagnostic in result.diagnostics)


def test_decomposition_records_tuple_negative_constraint(load_fixture_config) -> None:
    config = load_fixture_config(provider="fixture")
    sketch = parse_sketch(r"{□: year}-{□: integer}")

    result = decompose_examples(
        config,
        sketch,
        Examples(positive=["2026-42"], negative=["2027-99"]),
    )

    assert result.tuple_negative_constraints
    assert result.tuple_negative_constraints[0].hole_values == {
        "h0": ["2027"],
        "h1": ["99"],
    }
    assert result.stats.tuple_negative_constraint_count == 1
