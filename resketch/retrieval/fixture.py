from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from resketch.config import AppConfig, resolve_project_path
from resketch.models import CandidateRequest, CandidateSet, ProviderKind, RegexComponent
from resketch.regex_engine import is_valid_regex


class FixtureCandidateProvider:
    """Deterministic provider for tests, demos, and offline evaluation."""

    def __init__(self, config: AppConfig, fixture_path: str | Path | None = None) -> None:
        self._config = config
        configured_path = fixture_path or config.retrieval.fixture_path
        self._fixture_path = resolve_project_path(configured_path)
        self._by_type = self._load(self._fixture_path)

    def retrieve(self, request: CandidateRequest) -> CandidateSet:
        raw_components = self._by_type.get(request.hole.semantic_type, [])
        components: list[RegexComponent] = []
        seen_regexes: set[str] = set()
        for component in raw_components:
            if len(components) >= request.max_candidates:
                break
            if component.regex in seen_regexes or not is_valid_regex(component.regex):
                continue
            seen_regexes.add(component.regex)
            components.append(component)
        return CandidateSet(
            provider=ProviderKind.FIXTURE,
            hole=request.hole,
            components=components,
            trace={"fixture_path": str(self._fixture_path)},
        )

    @staticmethod
    def _load(path: Path) -> dict[str, list[RegexComponent]]:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            return {}
        raw_types = data.get("types", {})
        if not isinstance(raw_types, dict):
            return {}

        by_type: dict[str, list[RegexComponent]] = {}
        for semantic_type, raw_components in raw_types.items():
            if not isinstance(raw_components, list):
                continue
            by_type[str(semantic_type)] = [
                RegexComponent.model_validate(_normalize_component(item, semantic_type))
                for item in raw_components
                if isinstance(item, dict)
            ]
        return by_type


def _normalize_component(item: dict[str, Any], semantic_type: object) -> dict[str, Any]:
    normalized = dict(item)
    normalized.setdefault("type", str(semantic_type))
    return normalized
