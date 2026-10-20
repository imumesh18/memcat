"""``memcat`` CLI: destructive and store commands fail safe and report what really happened."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from memcat.cli import main
from tests.test_cli_config import GOOD, _fails, _write, s3  # noqa: F401 - s3 is a fixture

pytestmark = pytest.mark.usefixtures("s3")

NO_STORE = """
[service]
class = "tests.fakes.FakeHTTPTTS"
sample_rate = 16000
"""


def _ok(capsys, *argv: str) -> dict[str, Any]:
    assert main(list(argv)) == 0, argv
    return json.loads(capsys.readouterr().out)


def _cfg(tmp_path, ns: str = "tenant-a", extra: str = "", name: str = "memcat.toml") -> str:
    return str(_write(tmp_path, GOOD + f'\n[memcat]\nnamespace = "{ns}"\n' + extra, name))


def _warm(capsys, cfg: str, *phrases: str) -> dict[str, Any]:
    argv = ["--config", cfg, "warm"]
    for ph in phrases:
        argv += ["--phrase", ph]
    return _ok(capsys, *argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["stats"],
        ["export", "--to", "x"],
        ["import", "--from", "x"],
        ["purge", "--all", "--yes"],
        ["purge", "--older-than", "1d"],
        ["warm", "--phrase", "Hello there."],
    ],
)
def test_cache_commands_refuse_a_memory_store(tmp_path, capsys, argv) -> None:
    cfg = str(_write(tmp_path, NO_STORE))
    err = _fails(capsys, "--config", cfg, *argv)
    assert "needs a shared store" in err and "[store]" in err


def test_warm_dry_run_needs_no_store(tmp_path, capsys) -> None:
    cfg = str(_write(tmp_path, NO_STORE))
    assert _ok(capsys, "--config", cfg, "warm", "--phrase", "Hi.", "--dry-run")["would_render"]


def test_purge_all_needs_confirmation(tmp_path, capsys, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    assert _warm(capsys, cfg, "Thanks for calling.")["stored"] == 1
    # Not a terminal (as under pytest, cron or CI): refuse without --yes.
    err = _fails(capsys, "--config", cfg, "purge", "--all")
    assert "pass --yes" in err and "tenant-a" in err
    assert _ok(capsys, "--config", cfg, "stats")["entries"] == 1

    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stdin", Tty("n\n"))
    assert main(["--config", cfg, "purge", "--all"]) == 1
    err = capsys.readouterr().err
    assert "Delete every entry in namespace 'tenant-a'?" in err and "cancelled" in err
    assert _ok(capsys, "--config", cfg, "stats")["entries"] == 1
    monkeypatch.setattr("sys.stdin", Tty("y\n"))
    out = _ok(capsys, "--config", cfg, "purge", "--all")
    assert out == {"deleted": 1}
    assert _ok(capsys, "--config", cfg, "stats")["entries"] == 0


def test_import_refuses_another_namespace(tmp_path, capsys) -> None:
    a = _cfg(tmp_path, "tenant-a", name="a.toml")
    b = _cfg(tmp_path, "tenant-b", name="b.toml")
    _warm(capsys, a, "Thanks for calling.")
    exp = tmp_path / "export"
    assert _ok(capsys, "--config", a, "export", "--to", str(exp))["exported"] == 1

    err = _fails(capsys, "--config", b, "import", "--from", str(exp))
    assert "namespace 'tenant-a', not 'tenant-b'" in err and "--force" in err
    assert _ok(capsys, "--config", b, "stats")["entries"] == 0
    assert _ok(capsys, "--config", a, "import", "--from", str(exp))["imported"] == 1
    assert _ok(capsys, "--config", b, "import", "--from", str(exp), "--force")["imported"] == 1

    (exp / "manifest.json").unlink()
    assert "no manifest" in _fails(capsys, "--config", b, "import", "--from", str(exp))
    missing = str(tmp_path / "nope")
    assert "not found" in _fails(capsys, "--config", b, "import", "--from", missing)


def test_stats_counts_only_entries_it_found(tmp_path, capsys, monkeypatch) -> None:
    from memcat.core.store import s3 as s3_mod

    cfg = _cfg(tmp_path)
    _warm(capsys, cfg, "Thanks for calling.", "Have a good day.")
    real_scan = s3_mod._BoundS3Store.scan

    async def scan(self):
        async for d in real_scan(self):
            yield d
        yield b"\x00" * 16  # listed, but gone by the time it is read

    monkeypatch.setattr(s3_mod._BoundS3Store, "scan", scan)
    assert _ok(capsys, "--config", cfg, "stats")["entries"] == 2


async def test_warm_reports_this_runs_failures(tmp_path) -> None:
    import argparse

    from memcat import cli

    extra = '\n[service.args]\nfail_on = "boom"\n'
    conf = cli.load_config(_write(tmp_path, GOOD.replace("[store]", extra + "\n[store]")))
    ctx = cli._Ctx(conf, "warm")

    def args(phrase: str) -> argparse.Namespace:
        return argparse.Namespace(
            phrase=[phrase], from_file=None, dry_run=False, timeout=30.0, top=10, min_calls=1
        )

    first = await cli._cmd_warm(ctx, args("A boom happened."))
    assert first["render_failures"] == 1 and first["stored"] == 0
    second = await cli._cmd_warm(ctx, args("All good here."))
    assert second["render_failures"] == 0 and second["stored"] == 1


def test_older_than_help_says_what_it_measures(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["purge", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "Redis: since last written or served" in out and "S3: since written" in out


@pytest.mark.parametrize(
    "flags",
    [
        ["--all", "--older-than", "30d"],
        ["--all", "--text", "Thanks for calling."],
        ["--text", "Thanks for calling.", "--older-than", "30d"],
    ],
)
def test_purge_takes_one_selection(tmp_path, capsys, flags) -> None:
    cfg = _cfg(tmp_path)
    _warm(capsys, cfg, "Thanks for calling.")
    with pytest.raises(SystemExit) as e:
        main(["--config", cfg, "purge", *flags, "--yes"])
    assert e.value.code == 2 and "not allowed with" in capsys.readouterr().err
    assert _ok(capsys, "--config", cfg, "stats")["entries"] == 1


def test_purge_older_than_needs_a_positive_age(tmp_path, capsys) -> None:
    cfg = _cfg(tmp_path)
    _warm(capsys, cfg, "Thanks for calling.")
    assert "positive age" in _fails(capsys, "--config", cfg, "purge", "--older-than", "0s")
    assert _ok(capsys, "--config", cfg, "stats")["entries"] == 1


def test_export_never_mixes_with_an_old_export(tmp_path, capsys) -> None:
    cfg = _cfg(tmp_path)
    _warm(capsys, cfg, "Thanks for calling.", "Have a good day.")
    old, new = tmp_path / "old", tmp_path / "new"
    assert _ok(capsys, "--config", cfg, "export", "--to", str(old))["exported"] == 2
    assert _ok(capsys, "--config", cfg, "purge", "--text", "Have a good day.") == {"deleted": 1}
    assert "already holds an export" in _fails(capsys, "--config", cfg, "export", "--to", str(old))

    assert _ok(capsys, "--config", cfg, "export", "--to", str(new))["exported"] == 1
    manifest = json.loads((new / "manifest.json").read_text())
    assert [f"{h}.mcat" for h in manifest["digests"]] == [p.name for p in new.glob("*.mcat")]
    for p in old.glob("*.mcat"):  # files copied in beside the export are not part of it
        (new / p.name).write_bytes(p.read_bytes())
    assert _ok(capsys, "--config", cfg, "purge", "--all", "--yes") == {"deleted": 1}
    assert _ok(capsys, "--config", cfg, "import", "--from", str(new)) == {
        "imported": 1,
        "skipped": 0,
    }
    assert _ok(capsys, "--config", cfg, "stats")["entries"] == 1


def test_bad_input_is_one_line_not_a_traceback(tmp_path, capsys) -> None:
    bad_ns = _cfg(tmp_path, "bad ns", name="ns.toml")
    assert "namespace" in _fails(capsys, "--config", bad_ns, "stats")
    redis = GOOD.split("[store]")[0] + '[store]\ntype = "redis"\nurl = "http://nope"\n'
    assert "redis://" in _fails(capsys, "--config", str(_write(tmp_path, redis, "r.toml")), "stats")

    cfg = _cfg(tmp_path)
    _warm(capsys, cfg, "Thanks for calling.")
    afile = tmp_path / "afile"
    afile.write_text("")
    assert "cannot export" in _fails(capsys, "--config", cfg, "export", "--to", str(afile))

    exp = tmp_path / "export"
    _ok(capsys, "--config", cfg, "export", "--to", str(exp))
    (exp / "manifest.json").unlink()
    (exp / f"{'ab' * 16}.mcat").mkdir()  # a directory with an entry's name
    out = _ok(capsys, "--config", cfg, "import", "--from", str(exp), "--force")
    assert out == {"imported": 1, "skipped": 1}


async def test_bulk_commands_read_a_few_entries_at_a_time_on_the_bulk_budget() -> None:
    from memcat import cli

    class Custom:  # a BoundStore of the user's own: get_many only
        def __init__(self) -> None:
            self.reads: list[int] = []

        async def scan(self):
            for i in range(100):
                yield bytes([i]) * 16

        async def get_many(self, digests):
            self.reads.append(len(digests))
            return [b"x"] * len(digests)

    class Bulk(Custom):
        async def get_many(self, digests):
            raise AssertionError("the hot-path read has a 50 ms budget")

        async def get_bulk(self, digests):
            return await super().get_many(digests)

    for store in (Custom(), Bulk()):
        assert len([d async for d, _ in cli._read_all(store)]) == 100
        assert max(store.reads) <= 32 and sum(store.reads) == 100


def test_warm_on_a_slow_store_uses_the_bulk_budget_and_reports_store_errors(
    tmp_path, capsys, monkeypatch
) -> None:
    from memcat.core.store import s3 as s3_mod
    from memcat.core.store.base import StoreError

    real_get_many = s3_mod._BoundS3Store.get_many

    async def too_slow(self, digests):
        raise StoreError("TimeoutError: over the 50 ms call-path budget")

    async def get_bulk(self, digests):
        return await real_get_many(self, digests)

    monkeypatch.setattr(s3_mod._BoundS3Store, "get_many", too_slow)
    monkeypatch.setattr(s3_mod._BoundS3Store, "get_bulk", get_bulk, raising=False)
    cfg = _cfg(tmp_path)
    out = _warm(capsys, cfg, "Thanks for calling.")
    assert out["stored"] == 1 and out["store_errors"] == 0

    async def lock_fails(self, digest, ttl_s):
        raise StoreError("TimeoutError")

    monkeypatch.setattr(s3_mod._BoundS3Store, "try_lock", lock_fails)
    err = _fails(capsys, "--config", cfg, "warm", "--phrase", "Have a good day.")
    assert "stored nothing" in err and "1 store errors" in err
