import pytest
from pydantic import ValidationError

from resketch.config import load_config
from resketch.models import MatchMode, ProviderKind


def test_load_config_with_cli_style_overrides() -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "matching.mode": "search",
            "llm.model": "openai/example",
        }
    )

    assert config.retrieval.provider is ProviderKind.FIXTURE
    assert config.matching.mode is MatchMode.SEARCH
    assert config.llm.model == "openai/example"
    assert config.retrieval.db_path == "fixtures/candidates.sqlite"
    assert config.synthesis.automata.enabled
    assert config.synthesis.automata.alphabet_policy == "ascii_examples"


def test_grouped_scoring_config_loads_defaults() -> None:
    config = load_config()

    assert config.synthesis.scoring.semantic_fit.positive_match == 10.0
    assert config.synthesis.scoring.source_confidence.weight == 4.0
    assert config.synthesis.scoring.syntactic_complexity_cost.length_penalty == 0.01
    assert config.synthesis.scoring.strategy_penalty["direct_reuse"] == 0.0


def test_old_scoring_config_paths_are_not_accepted() -> None:
    with pytest.raises(ValidationError):
        load_config(
            overrides={
                "synthesis.score_weights.tuple_negative_probe_reject": 10.0,
            }
        )
