from __future__ import annotations

import hashlib
import json
import os
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from resketch.config import AppConfig, resolve_project_path
from resketch.models import CandidateRequest, CandidateSet, ProviderKind, RegexComponent
from resketch.regex_engine import is_valid_regex
from resketch.retrieval.base import ProviderError


class LLMCandidatePayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidates: list[RegexComponent] = Field(default_factory=list)


class LLMCandidateProvider:
    """LiteLLM-backed retrieval strategy that mimics database candidate lookup."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._cache_path = resolve_project_path(config.retrieval.cache_path)

    def retrieve(self, request: CandidateRequest) -> CandidateSet:
        cache_key = self._cache_key(request)
        cached = self._read_cache().get(cache_key) if self._config.retrieval.cache_enabled else None
        if cached is not None:
            return self._candidate_set_from_payload(request, cached, from_cache=True)

        payload = self._call_llm(request)
        if self._config.retrieval.cache_enabled:
            cache = self._read_cache()
            cache[cache_key] = payload
            self._write_cache(cache)
        return self._candidate_set_from_payload(request, payload, from_cache=False)

    def _call_llm(self, request: CandidateRequest) -> dict[str, Any]:
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")
        try:
            import litellm
        except ImportError as exc:
            msg = "LiteLLM is required for provider=llm. Install dependencies with Poetry."
            raise ProviderError(msg) from exc
        litellm.suppress_debug_info = True

        messages = [
            {"role": "system", "content": self._config.llm.system_prompt},
            {"role": "user", "content": self._format_user_prompt(request)},
        ]

        last_error: Exception | None = None
        for _ in range(self._config.llm.max_retries + 1):
            try:
                response = litellm.completion(
                    model=self._config.llm.model,
                    messages=messages,
                    temperature=self._config.llm.temperature,
                    timeout=self._config.llm.timeout_seconds,
                    response_format=self._config.llm.response_format,
                )
                content = _response_content(response)
                payload = json.loads(_extract_json_object(content))
                if not isinstance(payload, dict):
                    msg = "LLM response JSON was not an object"
                    raise ProviderError(msg)
                return cast(dict[str, Any], payload)
            except Exception as exc:  # LiteLLM providers raise provider-specific exceptions.
                last_error = exc

        msg = (
            "LLM candidate retrieval failed "
            f"for {request.hole.identifier}:{request.hole.semantic_type} "
            f"with model {self._config.llm.model!r} after configured retries: {last_error}"
        )
        raise ProviderError(msg)

    def _format_user_prompt(self, request: CandidateRequest) -> str:
        return self._config.llm.user_prompt_template.format(
            hole_type=request.hole.semantic_type,
            sketch=request.sketch,
            global_positive_examples=json.dumps(request.global_examples.positive),
            global_negative_examples=json.dumps(request.global_examples.negative),
            hole_positive_examples=json.dumps(request.hole_examples.hard.positive),
            hole_negative_examples=json.dumps(request.hole_examples.hard.negative),
            hole_soft_positive_examples=json.dumps(request.hole_examples.soft_positive),
            hole_diagnostics=json.dumps(request.hole_examples.diagnostics),
            max_candidates=request.max_candidates,
        )

    def _candidate_set_from_payload(
        self,
        request: CandidateRequest,
        payload: dict[str, Any],
        *,
        from_cache: bool,
    ) -> CandidateSet:
        try:
            parsed = LLMCandidatePayload.model_validate(payload)
        except ValidationError as exc:
            msg = f"LLM response did not match candidate schema: {exc}"
            raise ProviderError(msg) from exc

        components = []
        seen: set[str] = set()
        for component in parsed.candidates:
            if not is_valid_regex(component.regex):
                continue
            if self._config.retrieval.deduplicate and component.regex in seen:
                continue
            seen.add(component.regex)
            components.append(component)
            if len(components) >= request.max_candidates:
                break

        return CandidateSet(
            provider=ProviderKind.LLM,
            hole=request.hole,
            components=components,
            trace={
                "from_cache": from_cache,
                "model": self._config.llm.model,
            },
        )

    def _cache_key(self, request: CandidateRequest) -> str:
        data = {
            "request": request.model_dump(mode="json"),
            "model": self._config.llm.model,
            "temperature": self._config.llm.temperature,
            "max_candidates": self._config.llm.max_candidates,
        }
        encoded = json.dumps(data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _read_cache(self) -> dict[str, dict[str, Any]]:
        if not self._cache_path.exists():
            return {}
        with self._cache_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return {}
        return data

    def _write_cache(self, cache: dict[str, dict[str, Any]]) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2, sort_keys=True)


def _response_content(response: Any) -> str:
    if isinstance(response, dict):
        content = response["choices"][0]["message"]["content"]
    else:
        content = response.choices[0].message.content
    if not isinstance(content, str):
        msg = "LLM response content was not text"
        raise ProviderError(msg)
    return content


def _extract_json_object(content: str) -> str:
    start = content.find("{")
    if start < 0:
        msg = "LLM response did not contain a JSON object"
        raise ProviderError(msg)

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start : index + 1]

    msg = "LLM response contained an incomplete JSON object"
    raise ProviderError(msg)
