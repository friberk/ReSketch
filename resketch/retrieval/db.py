from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, cast

from resketch.config import AppConfig, resolve_project_path
from resketch.models import CandidateRequest, CandidateSet, Examples, ProviderKind, RegexComponent
from resketch.regex_engine import is_valid_regex
from resketch.retrieval.base import ProviderError


class DBCandidateProvider:
    """SQLite-backed retrieval strategy for annotated corpus candidates."""

    def __init__(self, config: AppConfig, db_path: str | Path | None = None) -> None:
        self._config = config
        configured_path = db_path or config.retrieval.db_path
        self._db_path = resolve_project_path(configured_path)
        self._schema_version = self._validate_database(self._db_path)

    def retrieve(self, request: CandidateRequest) -> CandidateSet:
        components: list[RegexComponent] = []
        seen_regexes: set[str] = set()

        with sqlite3.connect(self._db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                  id,
                  semantic_type,
                  regex,
                  description,
                  confidence,
                  source_id,
                  positive_examples_json,
                  negative_examples_json,
                  metadata_json
                FROM regex_components
                WHERE semantic_type = ?
                ORDER BY confidence IS NULL ASC, confidence DESC, id ASC
                """,
                (request.hole.semantic_type,),
            )

            for row in rows:
                regex = _row_str(row, "regex")
                if not is_valid_regex(regex):
                    continue
                if self._config.retrieval.deduplicate and regex in seen_regexes:
                    continue
                seen_regexes.add(regex)
                components.append(_component_from_row(row))
                if len(components) >= request.max_candidates:
                    break

        return CandidateSet(
            provider=ProviderKind.DB,
            hole=request.hole,
            components=components,
            trace={
                "db_path": str(self._db_path),
                "schema_version": self._schema_version,
            },
        )

    @staticmethod
    def _validate_database(path: Path) -> str:
        if not path.exists():
            msg = f"Candidate database does not exist: {path}"
            raise ProviderError(msg)

        try:
            with sqlite3.connect(path) as connection:
                row = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'schema_version'"
                ).fetchone()
        except sqlite3.Error as exc:
            msg = f"Candidate database is not a valid ReSketch SQLite corpus: {path}"
            raise ProviderError(msg) from exc

        if row is None or row[0] != "1":
            msg = (
                "Unsupported candidate database schema version "
                f"{row[0] if row else None!r}; expected '1'."
            )
            raise ProviderError(msg)
        return cast(str, row[0])


def _component_from_row(row: sqlite3.Row) -> RegexComponent:
    return RegexComponent(
        regex=_row_str(row, "regex"),
        type=_row_str(row, "semantic_type"),
        description=_row_str(row, "description"),
        confidence=_row_optional_float(row, "confidence"),
        source_id=_row_optional_str(row, "source_id"),
        examples=Examples(
            positive=_json_list(_row_str(row, "positive_examples_json"), "positive_examples_json"),
            negative=_json_list(_row_str(row, "negative_examples_json"), "negative_examples_json"),
        ),
        metadata=_json_object(_row_str(row, "metadata_json"), "metadata_json"),
    )


def _row_str(row: sqlite3.Row, key: str) -> str:
    value = row[key]
    if not isinstance(value, str):
        msg = f"Candidate database column {key!r} must be text"
        raise ProviderError(msg)
    return value


def _row_optional_str(row: sqlite3.Row, key: str) -> str | None:
    value = row[key]
    if value is None or isinstance(value, str):
        return value
    msg = f"Candidate database column {key!r} must be text or null"
    raise ProviderError(msg)


def _row_optional_float(row: sqlite3.Row, key: str) -> float | None:
    value = row[key]
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    msg = f"Candidate database column {key!r} must be numeric or null"
    raise ProviderError(msg)


def _json_list(raw: str, column: str) -> list[str]:
    value = _json_load(raw, column)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        msg = f"Candidate database column {column!r} must contain a JSON string array"
        raise ProviderError(msg)
    return value


def _json_object(raw: str, column: str) -> dict[str, Any]:
    value = _json_load(raw, column)
    if not isinstance(value, dict):
        msg = f"Candidate database column {column!r} must contain a JSON object"
        raise ProviderError(msg)
    return cast(dict[str, Any], value)


def _json_load(raw: str, column: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"Candidate database column {column!r} contains invalid JSON"
        raise ProviderError(msg) from exc
