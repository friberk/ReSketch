from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, cast

import yaml

SCHEMA = """
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

CREATE INDEX idx_regex_components_type_rank
  ON regex_components (semantic_type, confidence DESC, id);
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the seeded ReSketch SQLite corpus.")
    parser.add_argument(
        "--source",
        default="fixtures/candidates.yaml",
        type=Path,
        help="YAML fixture to seed from.",
    )
    parser.add_argument(
        "--output",
        default="fixtures/candidates.sqlite",
        type=Path,
        help="SQLite database to write.",
    )
    args = parser.parse_args()

    data = _load_yaml(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    with sqlite3.connect(args.output) as connection:
        connection.executescript(SCHEMA)
        connection.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ("schema_version", "1"),
        )
        connection.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ("seed_source", str(args.source)),
        )
        for semantic_type, component in _iter_components(data):
            examples = _mapping(component.get("examples"))
            metadata = _mapping(component.get("metadata"))
            connection.execute(
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
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    semantic_type,
                    _required_str(component, "regex"),
                    str(component.get("description", "")),
                    _optional_float(component.get("confidence")),
                    _optional_str(component.get("source_id")),
                    json.dumps(_string_list(examples.get("positive")), sort_keys=True),
                    json.dumps(_string_list(examples.get("negative")), sort_keys=True),
                    json.dumps(metadata, sort_keys=True),
                ),
            )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = f"Expected a mapping in {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], data)


def _iter_components(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    raw_types = data.get("types")
    if not isinstance(raw_types, dict):
        return []

    components: list[tuple[str, dict[str, Any]]] = []
    for semantic_type, raw_components in raw_types.items():
        if not isinstance(raw_components, list):
            continue
        for component in raw_components:
            if not isinstance(component, dict):
                continue
            normalized = dict(component)
            normalized.setdefault("type", str(semantic_type))
            components.append((str(normalized["type"]), normalized))
    return components


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        msg = f"Expected string field {key!r}"
        raise ValueError(msg)
    return value


def _optional_str(value: object) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return float(str(value))


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    return {}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


if __name__ == "__main__":
    main()
