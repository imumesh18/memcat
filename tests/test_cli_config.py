"""``memcat`` CLI: where ``--config`` goes, and clear errors for bad input."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from memcat.cli import main
from tests.fakes import FakeHTTPTTS


class NeedsKey(FakeHTTPTTS):
    def __init__(self, *, api_key: str, **kwargs: Any):
        super().__init__(**kwargs)


# Store commands need a shared store; tests use S3 under moto (see the s3 fixture).
STORE = """
[store]
type = "s3"
bucket = "memcat-cli"
region_name = "us-east-1"
"""

GOOD = (
    """
[service]
class = "tests.fakes.FakeHTTPTTS"
sample_rate = 16000
"""
    + STORE
)

# A provider that is not installed and a key that is not set: store-only commands
# must still work, because they never build the service.
NO_PROVIDER = (
    """
[service]
class = "not_installed.provider.TTSService"
sample_rate = 16000

[service.args]
api_key = "${MEMCAT_TEST_UNSET_KEY}"
"""
    + STORE
)


@pytest.fixture(autouse=True)
def s3(monkeypatch):
    pytest.importorskip("moto")
    import boto3
    from moto import mock_aws

    for k, v in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
    }.items():
        monkeypatch.setenv(k, v)
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="memcat-cli")
        yield


def _write(tmp: Path, text: str, name: str = "memcat.toml") -> Path:
    p = tmp / name
    p.write_text(text)
    return p


def _fails(capsys, *argv: str) -> str:
    assert main(list(argv)) == 1
    err = capsys.readouterr().err.strip()
    assert err.startswith("memcat") and "Traceback" not in err and len(err.splitlines()) == 1
    return err


@pytest.mark.parametrize("before", [True, False])
def test_config_before_or_after_the_command(tmp_path, capsys, before) -> None:
    cfg = str(_write(tmp_path, GOOD))
    argv = ["--config", cfg, "stats"] if before else ["stats", "--config", cfg]
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out)["entries"] == 0


def test_config_after_the_command_wins(tmp_path, capsys) -> None:
    good = str(_write(tmp_path, GOOD))
    assert main(["--config", str(tmp_path / "missing.toml"), "stats", "--config", good]) == 0
    capsys.readouterr()


def test_store_commands_do_not_build_the_service(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.delenv("MEMCAT_TEST_UNSET_KEY", raising=False)
    cfg = str(_write(tmp_path, NO_PROVIDER))
    for argv in (
        ["stats"],
        ["purge", "--all", "--yes"],
        ["purge", "--text", "hello there"],  # finds every voice's copy by text alone
        ["export", "--to", str(tmp_path / "x")],
    ):
        assert main(["--config", cfg, *argv]) == 0, argv
    assert main(["--config", cfg, "import", "--from", str(tmp_path / "x")]) == 0


def test_missing_config(tmp_path, capsys) -> None:
    err = _fails(capsys, "--config", str(tmp_path / "nope.toml"), "stats")
    assert "nope.toml not found" in err


def test_bad_toml(tmp_path, capsys) -> None:
    err = _fails(capsys, "--config", str(_write(tmp_path, "[service\nclass=")), "stats")
    assert "not valid TOML" in err


def test_missing_service_class(tmp_path, capsys) -> None:
    cfg = str(_write(tmp_path, "[service]\nsample_rate = 16000\n"))
    err = _fails(capsys, "--config", cfg, "warm", "--phrase", "hi")
    assert "[service] class is required" in err


def test_missing_service_class_for_serve(tmp_path, capsys) -> None:
    cfg = str(_write(tmp_path, "[serve]\nport = 0\n"))
    assert "[service] class is required" in _fails(capsys, "--config", cfg, "serve")


def test_unimportable_service_class(tmp_path, capsys) -> None:
    cfg = str(_write(tmp_path, '[service]\nclass = "tests.fakes.Nope"\nsample_rate = 16000\n'))
    assert "has no attribute Nope" in _fails(capsys, "--config", cfg, "warm", "--phrase", "hi")


def test_bad_service_args(tmp_path, capsys) -> None:
    cfg = str(_write(tmp_path, GOOD.replace("fakes.FakeHTTPTTS", "test_cli_config.NeedsKey")))
    assert "cannot build NeedsKey" in _fails(capsys, "--config", cfg, "warm", "--phrase", "hi")


def test_bad_jsonl_line(tmp_path, capsys) -> None:
    cfg = str(_write(tmp_path, GOOD))
    calls = _write(tmp_path, '{"agent_turns": ["Hello there."]}\n{oops\n', "calls.jsonl")
    err = _fails(capsys, "--config", cfg, "warm", "--from", str(calls), "--dry-run")
    assert "calls.jsonl:2: not valid JSON" in err


def test_jsonl_line_without_turns_list(tmp_path, capsys) -> None:
    cfg = str(_write(tmp_path, GOOD))
    calls = _write(tmp_path, '{"agent_turns": "Hello there."}\n', "calls.jsonl")
    err = _fails(capsys, "--config", cfg, "warm", "--from", str(calls), "--dry-run")
    assert "calls.jsonl:1: expected" in err


def test_missing_jsonl(tmp_path, capsys) -> None:
    cfg = str(_write(tmp_path, GOOD))
    err = _fails(capsys, "--config", cfg, "warm", "--from", str(tmp_path / "x.jsonl"))
    assert "x.jsonl not found" in err


def test_purge_needs_a_flag(tmp_path, capsys) -> None:
    cfg = str(_write(tmp_path, GOOD))
    assert "pass --text, --all or --older-than" in _fails(capsys, "--config", cfg, "purge")
