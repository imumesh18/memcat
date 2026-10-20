from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

from memcat.cli import load_config, main

REDIS = (os.environ.get("MEMCAT_REDIS_URLS", "").split(",") or [""])[0]
pytestmark = pytest.mark.skipif(not REDIS, reason="needs MEMCAT_REDIS_URLS")


def _toml(tmp: Path, cls: str, ns: str) -> Path:
    p = tmp / "memcat.toml"
    p.write_text(f"""
[service]
class = "{cls}"
sample_rate = 16000

[service.args]
voice_hint = "${{MEMCAT_TEST_VOICE:-alice}}"

[store]
type = "redis"
url = "{REDIS}"

[memcat]
namespace = "{ns}"
""")
    return p


def _run(capsys, *argv: str) -> dict[str, Any]:
    assert main(list(argv)) == 0
    return json.loads(capsys.readouterr().out)


def test_env_substitution(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MEMCAT_TEST_VOICE", "bob")
    conf = load_config(_toml(tmp_path, "x.Y", "n"))
    assert conf["service"]["args"]["voice_hint"] == "bob"


def test_warm_stats_export_import_purge(tmp_path, capsys, monkeypatch) -> None:
    from tests import fakes

    monkeypatch.setattr(fakes.FakeWSTTS, "__init__", _accept_hint(fakes.FakeWSTTS.__init__))
    ns = f"cli{uuid.uuid4().hex[:8]}"
    cfg = _toml(tmp_path, "tests.fakes.FakeWSTTS", ns)
    calls = tmp_path / "calls.jsonl"
    lines = []
    for i in range(5):
        turns = ["Thanks for calling, how can I help you today?", f"Your order {i}21 shipped."]
        if i != 2:  # in 4 of 5 calls
            turns.append("Is there anything else I can help with?")
        lines.append(json.dumps({"call_id": f"c{i}", "agent_turns": turns}))
    calls.write_text("\n".join(lines) + "\n")

    dry = _run(capsys, "--config", str(cfg), "warm", "--from", str(calls), "--dry-run")
    texts = [t for t, _ in dry["top"]]
    assert "how can I help you today?" in texts and all("21" not in t for t in texts)

    out = _run(capsys, "--config", str(cfg), "warm", "--from", str(calls), "--top", "10")
    assert out["stored"] == 3 and out["render_failures"] == 0
    st = _run(capsys, "--config", str(cfg), "stats")
    assert st["entries"] == 3 and st["origin"] == {"warmed": 3}

    exp = tmp_path / "export"
    assert _run(capsys, "--config", str(cfg), "export", "--to", str(exp))["exported"] == 3
    assert _run(capsys, "--config", str(cfg), "purge", "--all", "--yes")["deleted"] == 3
    assert _run(capsys, "--config", str(cfg), "stats")["entries"] == 0
    assert _run(capsys, "--config", str(cfg), "import", "--from", str(exp))["imported"] == 3
    assert _run(capsys, "--config", str(cfg), "purge", "--older-than", "1d")["deleted"] == 0
    assert (
        _run(
            capsys,
            "--config",
            str(cfg),
            "purge",
            "--text",
            "Thanks for calling, how can I help you today?",
        )["deleted"]
        == 2
    )
    assert _run(capsys, "--config", str(cfg), "purge", "--all", "--yes")["deleted"] == 1


def _accept_hint(init):
    def wrapped(self, *a, voice_hint=None, **kw):
        init(self, *a, **kw)

    return wrapped
