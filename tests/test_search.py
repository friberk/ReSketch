from resketch.config import load_config
from resketch.models import Examples, HoleExampleSet, RegexComponent
from resketch.search import RegexSearchEngine


def test_behavioral_dedup_keeps_best_representative() -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "synthesis.strategy_order": ["direct_reuse"],
        }
    )
    engine = RegexSearchEngine(config)

    result = engine.search(
        [
            RegexComponent(
                regex=r"[0-9]+",
                type="integer",
                description="digits char class",
                confidence=0.5,
            ),
            RegexComponent(
                regex=r"\d+",
                type="integer",
                description="digits category",
                confidence=0.9,
            ),
        ],
        HoleExampleSet(hard=Examples(positive=["123"], negative=["abc"])),
    )

    assert result.stats.pruned_duplicates >= 1
    assert result.options[0].regex == r"\d+"


def test_behavioral_dedup_distinguishes_tuple_negative_probes() -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "synthesis.strategy_order": ["direct_reuse"],
            "synthesis.scoring.semantic_fit.tuple_negative_probe_reject": 10.0,
        }
    )
    engine = RegexSearchEngine(config)

    result = engine.search(
        [
            RegexComponent(
                regex=r"\d{4}",
                type="year",
                description="any four digits",
                confidence=0.7,
            ),
            RegexComponent(
                regex=r"20\d{2}",
                type="year",
                description="2000s year",
                confidence=0.7,
            ),
        ],
        HoleExampleSet(hard=Examples(positive=["2026"])),
        negative_probes=["3026"],
    )

    regexes = {option.regex for option in result.options}
    assert regexes >= {r"\d{4}", r"20\d{2}"}
    assert result.options[0].regex == r"20\d{2}"
    assert result.options[0].constraint_probe_rejections == 1


def test_ast_recombination_solves_when_direct_reuse_fails() -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "synthesis.strategy_order": ["fragment_recombination"],
            "synthesis.search.max_ast_size": 3,
        }
    )
    engine = RegexSearchEngine(config)

    result = engine.search(
        [
            RegexComponent(regex=r"[A-Z]", type="code", description="letter"),
            RegexComponent(regex=r"\d", type="code", description="digit"),
        ],
        HoleExampleSet(hard=Examples(positive=["A1"], negative=["AA", "11"])),
    )

    assert any(option.regex == r"[A-Z]\d" for option in result.options)


def test_parameterized_refinement_adjusts_repeat_bounds() -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "synthesis.strategy_order": ["parameterized_refinement"],
            "synthesis.max_hole_options": 100,
            "synthesis.search.behavioral_dedup": False,
        }
    )
    engine = RegexSearchEngine(config)

    result = engine.search(
        [
            RegexComponent(regex=r"\d{3}", type="cvv", description="three digits"),
        ],
        HoleExampleSet(hard=Examples(positive=["1234"], negative=["12345"])),
    )

    assert any(option.regex == r"\d{3,4}" for option in result.options)


def test_search_respects_generated_candidate_limit() -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "synthesis.search.max_generated_candidates": 1,
        }
    )
    engine = RegexSearchEngine(config)

    result = engine.search(
        [
            RegexComponent(regex=r"\d+", type="integer", description="digits"),
            RegexComponent(regex=r"[A-Z]+", type="integer", description="letters"),
        ],
        HoleExampleSet(hard=Examples(positive=["123"])),
    )

    assert result.stats.generated_candidates <= 1


def test_cost_ordered_search_evaluates_low_cost_seed_first() -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "synthesis.strategy_order": ["direct_reuse"],
            "synthesis.search.max_generated_candidates": 1,
        }
    )
    engine = RegexSearchEngine(config)

    result = engine.search(
        [
            RegexComponent(regex=r"\d{2}", type="code", description="two digits"),
            RegexComponent(regex=r"\d", type="code", description="one digit"),
        ],
        HoleExampleSet(hard=Examples(positive=["1"])),
    )

    assert result.options[0].regex == r"\d"
    assert result.stats.frontier_size == 2
    assert result.stats.canonicalized_candidates == 1


def test_search_prefers_high_confidence_semantic_candidate_over_broad_pattern() -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "synthesis.strategy_order": ["direct_reuse"],
        }
    )
    engine = RegexSearchEngine(config)

    result = engine.search(
        [
            RegexComponent(
                regex=r"20\d{2}",
                type="year",
                description="2000s year",
                confidence=1.0,
            ),
            RegexComponent(
                regex=r"\d{4}",
                type="year",
                description="any four digits",
                confidence=0.3,
            ),
        ],
        HoleExampleSet(hard=Examples(positive=["2026"])),
    )

    assert result.options[0].regex == r"20\d{2}"


def test_search_prunes_unsupported_nonregular_candidate_when_raw_regex_disabled() -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "synthesis.strategy_order": ["direct_reuse"],
            "synthesis.search.allow_raw_regex": False,
        }
    )
    engine = RegexSearchEngine(config)

    result = engine.search(
        [
            RegexComponent(
                regex=r"(?=a)a",
                type="letter",
                description="lookahead candidate",
                confidence=1.0,
            ),
        ],
        HoleExampleSet(hard=Examples(positive=["a"])),
    )

    assert result.options == []
    assert result.stats.pruned_invalid_regexes == 1


def test_component_generalization_anti_unifies_literals() -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "synthesis.strategy_order": ["component_generalization"],
            "synthesis.search.behavioral_dedup": False,
        }
    )
    engine = RegexSearchEngine(config)

    result = engine.search(
        [
            RegexComponent(regex="A1", type="code", description="first code"),
            RegexComponent(regex="B2", type="code", description="second code"),
        ],
        HoleExampleSet(hard=Examples(positive=["A1", "B2"])),
    )

    assert any(option.regex == r"[A-Z]\d" for option in result.options)
    assert result.options[0].provenance_steps[0].operation == "anti_unify"


def test_component_generalization_widens_literal_lengths() -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "synthesis.strategy_order": ["component_generalization"],
            "synthesis.search.behavioral_dedup": False,
        }
    )
    engine = RegexSearchEngine(config)

    result = engine.search(
        [
            RegexComponent(regex="123", type="digits", description="three digits"),
            RegexComponent(regex="1234", type="digits", description="four digits"),
        ],
        HoleExampleSet(hard=Examples(positive=["123", "1234"], negative=["12"])),
    )

    assert any(option.regex == r"\d{3,4}" for option in result.options)


def test_component_generalization_infers_optional_literal_segment() -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "synthesis.strategy_order": ["component_generalization"],
            "synthesis.search.behavioral_dedup": False,
        }
    )
    engine = RegexSearchEngine(config)

    result = engine.search(
        [
            RegexComponent(regex="A12", type="code", description="short code"),
            RegexComponent(regex="AB12", type="code", description="long code"),
        ],
        HoleExampleSet(hard=Examples(positive=["A12", "AB12"])),
    )

    assert any(option.regex == "AB?12" for option in result.options)


def test_separator_refinement_introduces_optional_separator() -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "synthesis.strategy_order": ["parameterized_refinement"],
            "synthesis.search.behavioral_dedup": False,
        }
    )
    engine = RegexSearchEngine(config)

    result = engine.search(
        [
            RegexComponent(regex=r"\d", type="digit", description="digit"),
        ],
        HoleExampleSet(hard=Examples(positive=["1", "1-"], negative=["a"])),
    )

    assert any(option.regex == r"\d\-?" for option in result.options)


def test_anchor_refinement_does_not_duplicate_existing_anchors() -> None:
    config = load_config(
        overrides={
            "retrieval.provider": "fixture",
            "synthesis.strategy_order": ["parameterized_refinement"],
            "synthesis.search.behavioral_dedup": False,
        }
    )
    engine = RegexSearchEngine(config)

    result = engine.search(
        [
            RegexComponent(regex=r"^\d{4}$", type="year", description="anchored year"),
        ],
        HoleExampleSet(hard=Examples(positive=["2026"], negative=["abc"])),
    )

    assert all("^^" not in option.regex for option in result.options)
    assert all("$$" not in option.regex for option in result.options)
