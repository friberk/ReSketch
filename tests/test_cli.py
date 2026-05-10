from typer.testing import CliRunner

from resketch.cli import app


def test_cli_synthesize_fixture_noninteractive() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "synthesize",
            "--provider",
            "fixture",
            "--no-interactive",
            "--sketch",
            "{□: integer}",
            "--pos",
            "42",
            "--neg",
            "abc",
        ],
    )

    assert result.exit_code == 0
    assert '"success": true' in result.output


def test_cli_synthesize_db_noninteractive() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "synthesize",
            "--provider",
            "db",
            "--db-path",
            "fixtures/candidates.sqlite",
            "--no-interactive",
            "--sketch",
            "{□: integer}",
            "--pos",
            "123",
            "--neg",
            "abc",
        ],
    )

    assert result.exit_code == 0
    assert '"success": true' in result.output


def test_cli_accepts_explicit_hole_examples() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "synthesize",
            "--provider",
            "fixture",
            "--no-interactive",
            "--sketch",
            "{□: cvv}",
            "--pos",
            "1234",
            "--neg",
            "12",
            "--hole-pos",
            "h0=1234",
            "--hole-neg",
            "h0=12345",
        ],
    )

    assert result.exit_code == 0
    assert '"success": true' in result.output


def test_cli_inspect_sketch() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["inspect-sketch", "--sketch", "{□: year}-{□: integer}"],
    )

    assert result.exit_code == 0
    assert "h0" in result.output
    assert "h1" in result.output


def test_cli_accepts_decomposition_mode() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "synthesize",
            "--provider",
            "fixture",
            "--decomposition-mode",
            "hard-only",
            "--no-interactive",
            "--sketch",
            "{□: integer}",
            "--pos",
            "42",
            "--neg",
            "abc",
        ],
    )

    assert result.exit_code == 0
    assert '"success": true' in result.output


def test_cli_accepts_automata_pruning_toggle() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "synthesize",
            "--provider",
            "fixture",
            "--no-automata-pruning",
            "--no-interactive",
            "--sketch",
            "{□: integer}",
            "--pos",
            "42",
            "--neg",
            "abc",
        ],
    )

    assert result.exit_code == 0
    assert '"enabled": false' in result.output
