from resketch.cegis import CegisRunner
from resketch.models import Examples
from resketch.retrieval.fixture import FixtureCandidateProvider
from resketch.synthesis import Synthesizer


def test_cegis_accepts_candidate(load_fixture_config) -> None:
    config = load_fixture_config(provider="fixture")
    synthesizer = Synthesizer(config, FixtureCandidateProvider(config))
    prompts = iter(["y"])
    output: list[str] = []
    runner = CegisRunner(
        config,
        synthesizer,
        input_fn=lambda _: next(prompts),
        output_fn=output.append,
    )

    result = runner.run(
        "{□: integer}",
        Examples(positive=["42"], negative=["abc"]),
        interactive=True,
    )

    assert result.success
    assert output
