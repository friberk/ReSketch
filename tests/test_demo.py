import os
import subprocess
import sys


def test_interactive_demo_fixture_smoke() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "demo/interactive_demo.py",
            "--provider",
            "fixture",
            "--scenario",
            "product_code",
            "--no-pause",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Generalizing Product Codes" in result.stdout
    assert "SUCCESS" in result.stdout
    assert "anti_unify" in result.stdout


def test_interactive_demo_llm_missing_key_falls_back_to_fixture() -> None:
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "demo/interactive_demo.py",
            "--provider",
            "llm",
            "--scenario",
            "product_code",
            "--no-pause",
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0
    assert "Using fixture fallback" in result.stdout
    assert "Scenarios: product_code" in result.stdout
    assert "SUCCESS" in result.stdout


def test_interactive_demo_nested_repeat_smoke() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "demo/interactive_demo.py",
            "--provider",
            "fixture",
            "--scenario",
            "nested_repeat",
            "--no-pause",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Nested Repeated Hole" in result.stdout
    assert "repeated-hole constraints" in result.stdout
    assert "SUCCESS" in result.stdout


def test_interactive_demo_api_gateway_audit_smoke() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "demo/interactive_demo.py",
            "--provider",
            "fixture",
            "--scenario",
            "api_gateway_audit",
            "--no-pause",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "API Gateway Audit Log" in result.stdout
    assert "SUCCESS" in result.stdout
    assert "anti_unify" in result.stdout
    assert "repeated-hole constraints" in result.stdout
    assert "Tuple-Negative Constraints" in result.stdout
    assert "AB?12" in result.stdout
    assert "ORD" in result.stdout
