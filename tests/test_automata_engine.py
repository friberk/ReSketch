import pytest

from resketch.automata_engine import (
    check_complete_negative_rejection,
    check_partial_positive_feasibility,
)
from resketch.config import load_config
from resketch.models import Examples
from resketch.sketch import parse_sketch

pytest.importorskip("automata")


def test_automata_partial_positive_feasibility_accepts_possible_assignment() -> None:
    config = load_config(overrides={"retrieval.provider": "fixture"})
    sketch = parse_sketch(r"{□: integer}-{□: digit}")

    result = check_partial_positive_feasibility(
        config,
        sketch,
        {"h0": r"\d+"},
        Examples(positive=["12-3"]),
    )

    assert result.supported
    assert result.feasible
    assert result.alphabet_size > 0
    assert result.state_count > 0


def test_automata_partial_positive_feasibility_rejects_impossible_assignment() -> None:
    config = load_config(overrides={"retrieval.provider": "fixture"})
    sketch = parse_sketch(r"{□: integer}-{□: digit}")

    result = check_partial_positive_feasibility(
        config,
        sketch,
        {"h0": r"[A-Z]+"},
        Examples(positive=["12-3"]),
    )

    assert result.supported
    assert not result.feasible


def test_automata_complete_negative_rejection_prunes_bad_assignment() -> None:
    config = load_config(overrides={"retrieval.provider": "fixture"})
    sketch = parse_sketch(r"{□: integer}")

    result = check_complete_negative_rejection(
        config,
        sketch,
        {"h0": r"\d+"},
        Examples(negative=["123"]),
    )

    assert result.supported
    assert not result.feasible


def test_automata_pruning_fails_open_for_search_mode() -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "matching.mode": "search",
        }
    )
    sketch = parse_sketch(r"{□: integer}")

    result = check_partial_positive_feasibility(
        config,
        sketch,
        {"h0": r"\d+"},
        Examples(positive=["id=12"]),
    )

    assert not result.supported
    assert result.feasible
    assert result.reasons
