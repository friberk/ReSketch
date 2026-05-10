from resketch.evaluation import run_benchmark
from resketch.retrieval.fixture import FixtureCandidateProvider
from resketch.synthesis import Synthesizer


def test_benchmark_reports_decomposition_metrics(load_fixture_config) -> None:
    config = load_fixture_config(provider="fixture")
    result = run_benchmark(config, Synthesizer(config, FixtureCandidateProvider(config)))

    assert result["decomposition"]["global_positive_count"] > 0
    assert result["decomposition"]["decomposition_success_rate"] == 1.0
    assert result["tasks"][0]["decomposition"]["hard_evidence_count"] > 0
