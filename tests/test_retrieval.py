import sqlite3
from pathlib import Path

import pytest

from resketch.models import CandidateRequest, Examples, HoleExampleSet
from resketch.retrieval.base import ProviderError
from resketch.retrieval.db import DBCandidateProvider
from resketch.retrieval.fixture import FixtureCandidateProvider
from resketch.retrieval.llm import LLMCandidateProvider, _extract_json_object
from resketch.sketch import parse_sketch


def test_extract_json_object_allows_wrapped_content() -> None:
    content = 'Here is JSON: {"candidates": []}'

    assert _extract_json_object(content) == '{"candidates": []}'


def test_llm_provider_parses_mocked_response(load_fixture_config, monkeypatch) -> None:
    config = load_fixture_config(provider="llm")
    provider = LLMCandidateProvider(config)
    sketch = parse_sketch("{□: email_address}")

    def fake_completion(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"candidates":[{"regex":"\\\\d+","type":"integer",'
                            '"description":"digits","confidence":0.9}]}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setitem(
        __import__("sys").modules,
        "litellm",
        type("LiteLLM", (), {"completion": fake_completion}),
    )

    candidate_set = provider.retrieve(
        CandidateRequest(
            hole=sketch.holes[0],
            sketch=sketch.source,
            global_examples=Examples(),
            hole_examples=HoleExampleSet(),
            max_candidates=3,
        )
    )

    assert candidate_set.components[0].regex == r"\d+"


def test_llm_prompt_preserves_open_vocabulary_type(load_fixture_config) -> None:
    config = load_fixture_config(provider="llm")
    provider = LLMCandidateProvider(config)
    sketch = parse_sketch("{□: dinosaur}")

    prompt = provider._format_user_prompt(
        CandidateRequest(
            hole=sketch.holes[0],
            sketch=sketch.source,
            global_examples=Examples(positive=["triceratops"]),
            hole_examples=HoleExampleSet(),
            max_candidates=3,
        )
    )

    assert "Semantic type: dinosaur" in prompt
    assert "Taxonomy context" not in prompt


def test_fixture_provider_uses_exact_open_vocabulary_type(load_fixture_config) -> None:
    config = load_fixture_config(provider="fixture")
    provider = FixtureCandidateProvider(config)
    sketch = parse_sketch("{□: numeric}")

    candidate_set = provider.retrieve(
        CandidateRequest(
            hole=sketch.holes[0],
            sketch=sketch.source,
            global_examples=Examples(),
            hole_examples=HoleExampleSet(),
            max_candidates=20,
        )
    )

    assert candidate_set.components == []


def test_db_provider_retrieves_seeded_candidates(load_fixture_config) -> None:
    config = load_fixture_config(provider="db")
    provider = DBCandidateProvider(config)
    sketch = parse_sketch("{□: integer}")

    candidate_set = provider.retrieve(
        CandidateRequest(
            hole=sketch.holes[0],
            sketch=sketch.source,
            global_examples=Examples(),
            hole_examples=HoleExampleSet(),
            max_candidates=20,
        )
    )

    assert candidate_set.components[0].regex == r"[+-]?\d+"
    assert candidate_set.components[0].source_id == "fixture-integer"
    assert candidate_set.trace["schema_version"] == "1"


def test_db_provider_uses_exact_open_vocabulary_type(load_fixture_config) -> None:
    config = load_fixture_config(provider="db")
    provider = DBCandidateProvider(config)
    sketch = parse_sketch("{□: numeric}")

    candidate_set = provider.retrieve(
        CandidateRequest(
            hole=sketch.holes[0],
            sketch=sketch.source,
            global_examples=Examples(),
            hole_examples=HoleExampleSet(),
            max_candidates=20,
        )
    )

    assert candidate_set.components == []


def test_db_provider_honors_max_candidates_and_confidence_order(load_fixture_config) -> None:
    config = load_fixture_config(provider="db")
    provider = DBCandidateProvider(config)
    sketch = parse_sketch("{□: credit_card}")

    candidate_set = provider.retrieve(
        CandidateRequest(
            hole=sketch.holes[0],
            sketch=sketch.source,
            global_examples=Examples(),
            hole_examples=HoleExampleSet(),
            max_candidates=1,
        )
    )

    assert [component.source_id for component in candidate_set.components] == [
        "fixture-credit-card-flex"
    ]


def test_db_provider_filters_invalid_regexes_and_duplicates(
    load_fixture_config,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "candidates.sqlite"
    _write_test_db(db_path)
    config = load_fixture_config(provider="db")
    provider = DBCandidateProvider(config, db_path=db_path)
    sketch = parse_sketch("{□: token}")

    candidate_set = provider.retrieve(
        CandidateRequest(
            hole=sketch.holes[0],
            sketch=sketch.source,
            global_examples=Examples(),
            hole_examples=HoleExampleSet(),
            max_candidates=10,
        )
    )

    assert [component.source_id for component in candidate_set.components] == ["valid-token"]


def test_db_provider_rejects_missing_database(load_fixture_config, tmp_path: Path) -> None:
    config = load_fixture_config(provider="db")

    with pytest.raises(ProviderError, match="does not exist"):
        DBCandidateProvider(config, db_path=tmp_path / "missing.sqlite")


def _write_test_db(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE metadata (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );

            CREATE TABLE regex_components (
              id INTEGER PRIMARY KEY,
              semantic_type TEXT NOT NULL,
              regex TEXT NOT NULL,
              description TEXT NOT NULL,
              confidence REAL,
              source_id TEXT,
              positive_examples_json TEXT NOT NULL DEFAULT '[]',
              negative_examples_json TEXT NOT NULL DEFAULT '[]',
              metadata_json TEXT NOT NULL DEFAULT '{}',
              CHECK (confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0),
              UNIQUE (semantic_type, regex, source_id)
            );
            """
        )
        connection.execute(
            "INSERT INTO metadata (key, value) VALUES ('schema_version', '1')"
        )
        connection.executemany(
            """
            INSERT INTO regex_components (
              semantic_type,
              regex,
              description,
              confidence,
              source_id,
              positive_examples_json,
              negative_examples_json,
              metadata_json
            )
            VALUES (?, ?, ?, ?, ?, '[]', '[]', '{}')
            """,
            [
                ("token", "[", "invalid", 0.99, "invalid-token"),
                ("token", r"\w+", "valid", 0.9, "valid-token"),
                ("token", r"\w+", "duplicate", 0.8, "duplicate-token"),
            ],
        )
