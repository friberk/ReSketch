import pytest

from resketch.config import load_config
from resketch.models import (
    CandidateRequest,
    CandidateSet,
    CompletenessStatus,
    Examples,
    ProviderKind,
    RegexComponent,
    SynthesisSpec,
)
from resketch.retrieval.fixture import FixtureCandidateProvider
from resketch.synthesis import Synthesizer


class _AnchoredCandidateProvider:
    def retrieve(self, request: CandidateRequest) -> CandidateSet:
        return CandidateSet(
            provider=ProviderKind.FIXTURE,
            hole=request.hole,
            components=[
                RegexComponent(
                    regex=r"^\d{3}$",
                    type=request.hole.semantic_type,
                    description="Anchored three-digit component.",
                    confidence=1.0,
                    source_id="anchored-cvv",
                )
            ],
        )


class _ServerLogCandidateProvider:
    def retrieve(self, request: CandidateRequest) -> CandidateSet:
        components_by_type = {
            "iso_date": [
                RegexComponent(
                    regex=r"^\d{4}-\d{2}-\d{2}$",
                    type="iso_date",
                    description="Anchored ISO-like date.",
                    confidence=0.7,
                    source_id="date",
                )
            ],
            "ipv4_address": [
                RegexComponent(
                    regex=r"^(?:\d{1,3}\.){3}\d{1,3}$",
                    type="ipv4_address",
                    description="Broad dotted quad.",
                    confidence=0.8,
                    source_id="broad-ipv4",
                ),
                RegexComponent(
                    regex=(
                        r"^(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}"
                        r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)$"
                    ),
                    type="ipv4_address",
                    description="Strict IPv4.",
                    confidence=0.8,
                    source_id="strict-ipv4",
                ),
            ],
            "http_method": [
                RegexComponent(
                    regex=r"[A-Z]{3,7}",
                    type="http_method",
                    description="Broad uppercase method.",
                    confidence=0.8,
                    source_id="broad-method",
                ),
                RegexComponent(
                    regex=r"^(?:GET|POST|PUT|DELETE|PATCH)$",
                    type="http_method",
                    description="Whitelisted method.",
                    confidence=0.8,
                    source_id="strict-method",
                ),
            ],
            "url_path": [
                RegexComponent(
                    regex=r"^/[A-Za-z0-9/._-]+$",
                    type="url_path",
                    description="URL path.",
                    confidence=0.8,
                    source_id="path",
                )
            ],
        }
        return CandidateSet(
            provider=ProviderKind.FIXTURE,
            hole=request.hole,
            components=components_by_type[request.hole.semantic_type],
        )


def test_synthesizer_direct_reuse_email(load_fixture_config) -> None:
    config = load_fixture_config(provider="fixture")
    synthesizer = Synthesizer(config, FixtureCandidateProvider(config))

    result = synthesizer.synthesize(
        "{□: email_address}",
        Examples(positive=["john.doe@example.com"], negative=["not an email"]),
    )

    assert result.success
    assert result.regex is not None
    assert "h0" in result.trace.search_stats
    assert result.trace.selected_option_provenance["h0"]["source_id"] == "fixture-email-address"


def test_synthesizer_handles_literal_parts(load_fixture_config) -> None:
    config = load_fixture_config(provider="fixture")
    synthesizer = Synthesizer(config, FixtureCandidateProvider(config))

    result = synthesizer.synthesize(
        "CVV:{□: cvv}",
        Examples(positive=["CVV:123"], negative=["CVV:12"]),
    )

    assert result.success
    assert result.regex == r"CVV:\d{3,4}"


def test_embedded_hole_strips_outer_anchors() -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "cegis.interactive": False,
            "synthesis.strategy_order": ["direct_reuse"],
        }
    )
    synthesizer = Synthesizer(config, _AnchoredCandidateProvider())

    result = synthesizer.synthesize(
        "CVV:{□: cvv}",
        Examples(positive=["CVV:123"], negative=["CVV:12"]),
    )

    assert result.success
    assert result.regex == r"CVV:\d{3}"
    assert result.trace.hole_assignments["h0"] == r"\d{3}"
    assert (
        result.trace.selected_provenance_steps["h0"][-1].operation
        == "strip_embedding_anchors"
    )


def test_server_log_synthesis_keeps_probe_distinguishable_candidates() -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "cegis.interactive": False,
            "synthesis.strategy_order": ["direct_reuse"],
            "synthesis.beam_size": 64,
            "synthesis.max_hole_options": 64,
            "synthesis.global_pruning.max_partial_beams": 64,
        }
    )
    synthesizer = Synthesizer(config, _ServerLogCandidateProvider())

    result = synthesizer.synthesize(
        '{hole: iso_date} {hole: ipv4_address} "{hole: http_method} {hole: url_path}"',
        Examples(
            positive=[
                '2026-04-27 192.168.0.1 "GET /api/v1/orders"',
                '2026-04-27 10.0.0.7 "POST /checkout/pay"',
            ],
            negative=[
                '2026-04-27 999.999.999.999 "GET /api/v1/orders"',
                '2026-04-27 192.168.0.1 "TRACE /api/v1/orders"',
            ],
        ),
    )

    assert result.success
    assert result.regex is not None
    assert "999.999.999.999" not in result.trace.hole_witnesses["h1"]
    assert result.trace.hole_assignments["h1"].startswith(
        r"(?:(?:1\d{2}|25[0-5]|2[0-4]\d|[1-9]?\d)\.){3}"
    )
    assert result.trace.hole_assignments["h2"] == r"(?:DELETE|GET|PATCH|POST|PUT)"
    assert any(
        feature.constraint_probe_rejections > 0
        for feature in result.trace.candidate_features["h1"]
    )


def test_synthesizer_records_decomposition_and_witnesses(load_fixture_config) -> None:
    config = load_fixture_config(provider="fixture")
    synthesizer = Synthesizer(config, FixtureCandidateProvider(config))

    result = synthesizer.synthesize(
        "{□: year}-{□: integer}",
        Examples(positive=["2026-42"], negative=["3026-42"]),
    )

    assert result.success
    assert result.trace.decomposition is not None
    assert result.trace.decomposition.by_hole["h0"].hard.positive == ["2026"]
    assert result.trace.decomposition.by_hole["h1"].hard.positive == ["42"]
    assert result.trace.hole_witnesses == {"h0": ["2026"], "h1": ["42"]}


def test_explicit_hole_examples_prune_candidates(load_fixture_config) -> None:
    config = load_fixture_config(provider="fixture")
    synthesizer = Synthesizer(config, FixtureCandidateProvider(config))

    result = synthesizer.synthesize(
        "{□: cvv}",
        SynthesisSpec(
            global_examples=Examples(positive=["1234"], negative=["12"]),
            explicit_hole_examples={"h0": Examples(positive=["1234"], negative=["12345"])},
        ),
    )

    assert result.success
    assert result.regex == r"\d{3,4}"


def test_synthesizer_handles_nested_repeat_hole(load_fixture_config) -> None:
    config = load_fixture_config(provider="fixture")
    synthesizer = Synthesizer(config, FixtureCandidateProvider(config))

    result = synthesizer.synthesize(
        r"{□: digit}{3}",
        Examples(positive=["123"], negative=["12", "abc"]),
    )

    assert result.success
    assert result.regex == r"(?:\d){3}"
    assert result.trace.hole_witnesses == {"h0": ["1", "2", "3"]}


def test_synthesizer_records_provenance_and_completeness_scope(load_fixture_config) -> None:
    config = load_fixture_config(provider="fixture")
    synthesizer = Synthesizer(config, FixtureCandidateProvider(config))

    result = synthesizer.synthesize(
        "{□: integer}",
        Examples(positive=["42"], negative=["abc"]),
    )

    assert result.success
    assert result.trace.completeness_scope is not None
    assert result.trace.completeness_scope.regex_subset == "regular"
    assert result.trace.automata_stats.enabled
    assert result.trace.candidate_features["h0"]
    feature = result.trace.candidate_features["h0"][0]
    dumped_feature = feature.model_dump()
    breakdown = feature.score_breakdown
    assert "local_score" not in dumped_feature
    assert "cost" not in dumped_feature
    assert "search_score" not in dumped_feature
    assert "score_breakdown" in dumped_feature
    assert breakdown.syntactic_complexity_cost == pytest.approx(
        breakdown.ast_complexity_cost + breakdown.length_complexity_cost
    )
    assert breakdown.total_score == pytest.approx(
        breakdown.semantic_fit_score
        + breakdown.source_confidence_score
        - breakdown.syntactic_complexity_cost
        - breakdown.strategy_penalty
    )
    assert result.trace.selected_provenance_steps["h0"]


def test_global_partial_pruning_rejects_impossible_prefix(load_fixture_config) -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "cegis.interactive": False,
            "decomposition.mode": "off",
            "synthesis.strategy_order": ["direct_reuse"],
        }
    )
    synthesizer = Synthesizer(config, FixtureCandidateProvider(config))

    result = synthesizer.synthesize(
        "{□: integer}{□: digit}",
        Examples(positive=["A1"]),
    )

    assert not result.success
    assert result.trace.pruned_partial_beams > 0
    assert result.trace.search_stats["h0"].pruned_global_positive_feasibility > 0


def test_tuple_negative_constraints_prune_joint_assignment(load_fixture_config) -> None:
    config = load_fixture_config(provider="fixture")
    synthesizer = Synthesizer(config, FixtureCandidateProvider(config))

    result = synthesizer.synthesize(
        "{□: year}-{□: integer}",
        Examples(positive=["2026-42"], negative=["2026-999"]),
    )

    assert result.trace.decomposition is not None
    assert result.trace.decomposition.tuple_negative_constraints
    assert result.trace.pruned_tuple_negative_constraints > 0


def test_global_negative_pruning_rejects_complete_assignment(load_fixture_config) -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "cegis.interactive": False,
            "decomposition.mode": "off",
            "synthesis.strategy_order": ["direct_reuse"],
        }
    )
    synthesizer = Synthesizer(config, FixtureCandidateProvider(config))

    result = synthesizer.synthesize(
        "{□: integer}",
        Examples(positive=["42"], negative=["123"]),
    )

    assert not result.success
    assert result.trace.search_stats["h0"].pruned_global_negative_feasibility > 0
    assert result.trace.automata_stats.pruned_negative_feasibility > 0
    assert any(
        "No complete assignment survived global assembly" in diagnostic
        for diagnostic in result.trace.diagnostics
    )


def test_completeness_status_records_non_fullmatch_automata_scope(load_fixture_config) -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "cegis.interactive": False,
            "matching.mode": "search",
        }
    )
    synthesizer = Synthesizer(config, FixtureCandidateProvider(config))

    result = synthesizer.synthesize(
        "{□: integer}",
        Examples(positive=["id=42"], negative=["abc"]),
    )

    assert result.trace.completeness_scope is not None
    assert (
        result.trace.completeness_scope.status
        is CompletenessStatus.INCOMPLETE_DUE_TO_NON_FULLMATCH
    )


def test_outcome_logging_writes_jsonl(load_fixture_config, tmp_path) -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "cegis.interactive": False,
            "synthesis.outcome_logging.enabled": True,
            "synthesis.outcome_logging.path": str(tmp_path / "outcomes.jsonl"),
        }
    )
    synthesizer = Synthesizer(config, FixtureCandidateProvider(config))

    result = synthesizer.synthesize("{□: integer}", Examples(positive=["42"]))

    assert result.success
    assert (tmp_path / "outcomes.jsonl").read_text(encoding="utf-8")
