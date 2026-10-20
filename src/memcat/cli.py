"""``memcat`` command line: warm, stats, purge, export, import, serve (see docs/operations.md)."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import re
import sys
import time
import tomllib
from collections import Counter
from collections.abc import AsyncIterator, Callable, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any

from memcat.core.store.base import StoreError

__all__ = ["CLIError", "load_config", "main"]

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_SENT = re.compile(r"(?<=[.!?])\s+(?=\S)")
_ENTRY = ".mcat"  # export file suffix
_READ_BATCH = 32  # entries per bulk read: a few MB at most


class CLIError(Exception):
    """A configuration or input problem, reported as ``memcat: <message>`` and exit 1."""


def _subst(value: Any) -> Any:
    if isinstance(value, str):

        def rep(m: re.Match[str]) -> str:
            v = os.environ.get(m.group(1))
            if v is not None:
                return v
            default: str | None = m.group(2)
            if default is None:
                raise CLIError(f"memcat: environment variable {m.group(1)} is not set")
            return default

        return _ENV.sub(rep, value)
    if isinstance(value, dict):
        return {k: _subst(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_subst(v) for v in value]
    return value


def _read_config(path: Path) -> dict[str, Any]:
    """memcat.toml as written, before ${ENV} substitution."""
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise CLIError(f"memcat: {path} not found (pass --config PATH)") from None
    except tomllib.TOMLDecodeError as e:
        raise CLIError(f"memcat: {path} is not valid TOML: {e}") from None
    except OSError as e:
        raise CLIError(f"memcat: cannot read {path}: {e.strerror or e}") from None


def load_config(path: Path) -> dict[str, Any]:
    """memcat.toml with ${ENV} substituted everywhere."""
    return _subst(_read_config(path))


def _section(conf: dict[str, Any], name: str) -> dict[str, Any]:
    value = conf.get(name, {})
    if not isinstance(value, dict):
        raise CLIError(f"memcat: [{name}] must be a table")
    return _subst(value)


def _import(path: str) -> Any:
    mod, _, name = path.rpartition(".")
    if not mod:
        raise CLIError(f"memcat: '{path}' is not a dotted class path")
    try:
        return getattr(importlib.import_module(mod), name)
    except ImportError as e:
        raise CLIError(f"memcat: cannot import {mod}: {e}") from None
    except AttributeError:
        raise CLIError(f"memcat: {mod} has no attribute {name}") from None


def _store(conf: dict[str, Any]) -> Any:
    from memcat import MemoryStore, RedisStore, S3Store

    s: dict[str, Any] = _section(conf, "store") if "store" in conf else {"type": "memory"}
    kind = s.pop("type", "memory")
    if kind == "memory":
        return MemoryStore()
    if kind == "redis":
        return RedisStore(s.pop("url", "redis://localhost:6379/0"), **s)
    if kind == "s3":
        return S3Store(**s)
    raise CLIError(f"memcat: unknown store type '{kind}'")


def _memcat_config(conf: dict[str, Any]) -> Any:
    from memcat import MemcatConfig

    try:
        return MemcatConfig(store=_store(conf), **_section(conf, "memcat"))
    except (TypeError, ValueError) as e:
        raise CLIError(f"memcat: bad [memcat] or [store] option: {e}") from None


class _Service:
    """The ``[service]`` table: which Pipecat service to wrap and how to build it."""

    def __init__(self, conf: dict[str, Any]):
        svc = _section(conf, "service")
        if "class" not in svc:
            raise CLIError("memcat: [service] class is required")
        self.base: Any = _import(str(svc["class"]))
        try:
            self.sample_rate = int(svc.get("sample_rate", 0))
        except (TypeError, ValueError):
            raise CLIError("memcat: [service] sample_rate must be a number") from None
        args = svc.get("args", {})
        if not isinstance(args, dict):
            raise CLIError("memcat: [service.args] must be a table")
        self.args: dict[str, Any] = dict(args)

    def factory(self, cfg: Any) -> Callable[[int], Any]:
        """A function building the Memcat-wrapped service at a given output rate."""
        from memcat import cached

        wrapped = cached(self.base)

        def make(rate: int) -> Any:
            return wrapped(memcat=cfg, **{**self.args, "sample_rate": rate})

        return make

    def check(self, make: Callable[[int], Any], rate: int) -> Any:
        """Build one service now, so bad constructor arguments fail with a clear message."""
        try:
            return make(rate)
        except Exception as e:  # the service's own constructor: any error is a config error
            raise CLIError(
                f"memcat: cannot build {self.base.__name__}: {type(e).__name__}: {e}"
            ) from None


class _Ctx:
    """Everything a command needs, built from memcat.toml on first use."""

    def __init__(self, conf: dict[str, Any], cmd: str = ""):
        self._conf = conf
        self._cmd = cmd

    @cached_property
    def cfg(self) -> Any:
        return _memcat_config(self._conf)

    @cached_property
    def spec(self) -> _Service:
        spec = _Service(self._conf)
        if not spec.sample_rate:
            raise CLIError("memcat: [service] sample_rate is required (it is part of every key)")
        return spec

    @cached_property
    def service(self) -> Any:
        svc = self.spec.check(self.spec.factory(self.cfg), self.spec.sample_rate)
        # Keys include the output rate, which Pipecat only sets when a pipeline starts;
        # TTSService has no public setter, so set it from [service] sample_rate here.
        svc._sample_rate = self.spec.sample_rate
        return svc

    def runtime(self) -> Any:
        """The shared store's runtime. A memory store would be empty and gone when the
        command exits, so commands that read or write the cache refuse it."""
        from memcat import MemoryStore
        from memcat.core.runtime import get_runtime

        if isinstance(self.cfg.store, MemoryStore):
            raise CLIError(
                f"memcat {self._cmd}: needs a shared store, but memcat.toml has no [store] "
                'section (or type = "memory"); add [store] with type = "redis" or "s3"'
            )
        return get_runtime(self.cfg)


async def _read_all(store: Any) -> AsyncIterator[tuple[bytes, bytes]]:
    """Every entry in the store, read a few at a time with the store's bulk time budget
    where it has one (the hot-path budget is far too short for megabytes)."""
    read = getattr(store, "get_bulk", store.get_many)
    digests = [d async for d in store.scan()]
    for i in range(0, len(digests), _READ_BATCH):
        chunk = digests[i : i + _READ_BATCH]
        for d, blob in zip(chunk, await read(chunk), strict=True):
            if blob is not None:  # None: listed but expired or deleted since
                yield d, blob


async def _cmd_stats(ctx: _Ctx, args: argparse.Namespace) -> dict[str, Any]:
    from memcat.core import codec

    codecs: Counter[str] = Counter()
    origin: Counter[str] = Counter()
    total_bytes = 0
    total_ms = 0
    corrupt = 0
    found = 0
    names = {codec.CODEC_PCM16: "pcm16", codec.CODEC_MULAW: "mulaw", codec.CODEC_OPUS: "opus"}
    async for _, blob in _read_all(ctx.runtime().store):
        found += 1
        try:
            e = codec.decode(blob)
        except codec.CorruptEntryError:
            corrupt += 1
            continue
        total_bytes += len(blob)
        total_ms += e.duration_ms
        codecs[names.get(e.codec, str(e.codec))] += 1
        if e.flags & codec.FLAG_WARMED:
            origin["warmed"] += 1
        elif e.flags & codec.FLAG_CAPTURED:
            origin["captured"] += 1
        else:
            origin["rendered"] += 1
    return {
        "namespace": ctx.cfg.namespace,
        "entries": found,
        "bytes": total_bytes,
        "audio_minutes": round(total_ms / 60000, 2),
        "kb_per_minute": round(total_bytes / 1024 / (total_ms / 60000), 1) if total_ms else 0,
        "codecs": dict(codecs),
        "origin": dict(origin),
        "corrupt": corrupt,
    }


def _turn_clauses(ctx: _Ctx, turn: str) -> list[str]:
    out: list[str] = []
    for sentence in _SENT.split(turn.strip()):
        out.extend(
            c.text for c in ctx.service._memcat_split(sentence) if c.cacheable and c.text.strip()
        )
    return out


def _read_calls(path: Path) -> list[list[str]]:
    """Each JSONL line's ``agent_turns``."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise CLIError(f"memcat: {path} not found") from None
    except (OSError, UnicodeDecodeError) as e:
        raise CLIError(f"memcat: cannot read {path}: {e}") from None
    calls: list[list[str]] = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            call = json.loads(line)
        except json.JSONDecodeError as e:
            raise CLIError(f"memcat: {path}:{n}: not valid JSON ({e.msg})") from None
        turns = call.get("agent_turns", []) if isinstance(call, dict) else None
        if not isinstance(turns, list):
            raise CLIError(f'memcat: {path}:{n}: expected {{"agent_turns": [...]}}')
        calls.append([str(t) for t in turns])
    return calls


async def _cmd_warm(ctx: _Ctx, args: argparse.Namespace) -> dict[str, Any]:
    from memcat.core.keys import normalize_text
    from memcat.pipecat.render import RenderPool

    phrases: list[str] = list(args.phrase or [])
    ranked: list[tuple[str, int]] = []
    if args.from_file:
        counts: Counter[str] = Counter()
        first: dict[str, str] = {}
        for turns in await asyncio.to_thread(_read_calls, Path(args.from_file)):
            seen: set[str] = set()
            for turn in turns:
                for clause in _turn_clauses(ctx, str(turn)):
                    k = normalize_text(clause)
                    first.setdefault(k, clause)
                    seen.add(k)
            counts.update(seen)
        ranked = [(first[k], n) for k, n in counts.most_common(args.top) if n >= args.min_calls]
        phrases.extend(t for t, _ in ranked)
    if args.dry_run:
        return {"would_render": len(phrases), "top": ranked[:20]}
    svc = ctx.service  # first, so a bad [service] is reported before a missing [store]
    rt = ctx.runtime()
    rt.acquire()
    spec = ctx.spec
    pool = RenderPool.shared(spec.base, (), {**spec.args, "sample_rate": spec.sample_rate})
    # What MemcatTTSMixin.start() wires up, without starting a Pipecat pipeline. These are
    # Memcat's own attributes; the mixin has no public way to warm outside a pipeline.
    svc._memcat_rt, svc._memcat_pool = rt, pool
    before = rt.metrics.snapshot()
    t0 = time.monotonic()
    svc._memcat_warm(phrases)
    await rt.worker.drain_all(timeout=args.timeout)
    m = rt.metrics.snapshot()
    await pool.close()
    stored = m["entries_stored"] - before["entries_stored"]
    store_errors = m["store_errors"] - before["store_errors"]
    if phrases and not stored and store_errors:
        raise CLIError(f"memcat: warm stored nothing: {store_errors} store errors")
    return {
        "phrases": len(phrases),
        "stored": stored,
        "render_failures": m["render_failures"] - before["render_failures"],
        "store_errors": store_errors,
        "chars_rendered": svc._memcat_metrics.counters["chars_rendered"],
        "seconds": round(time.monotonic() - t0, 1),
    }


def _parse_age(s: str) -> float:
    m = re.fullmatch(r"(\d+)([smhd])", s.strip())
    if not m or not int(m.group(1)):
        raise CLIError("memcat: --older-than takes a positive age, e.g. 30d, 12h, 90m")
    return int(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


async def _confirm_purge_all(namespace: str, yes: bool) -> None:
    if yes:
        return
    if not sys.stdin.isatty():
        raise CLIError(
            f"memcat: purge --all deletes every entry in namespace '{namespace}'; "
            "pass --yes to confirm"
        )
    # The prompt goes to stderr: stdout carries only the JSON result.
    print(f"Delete every entry in namespace '{namespace}'? [y/N] ", end="", file=sys.stderr)
    sys.stderr.flush()
    answer = await asyncio.to_thread(sys.stdin.readline)
    if answer.strip().lower() not in ("y", "yes"):
        raise CLIError("memcat: purge --all cancelled")


async def _cmd_purge(ctx: _Ctx, args: argparse.Namespace) -> dict[str, Any]:
    if not (args.all or args.text or args.older_than):
        raise CLIError("memcat purge: pass --text, --all or --older-than")
    age = _parse_age(args.older_than) if args.older_than else 0.0
    rt = ctx.runtime()
    if args.all:
        await _confirm_purge_all(ctx.cfg.namespace, args.yes)
        return {"deleted": await rt.purge(None)}
    if args.text:
        return {"deleted": await rt.purge_text(args.text, ctx.cfg)}
    idle = getattr(rt.store, "idle_seconds", None)
    if idle is None:
        raise CLIError("memcat: --older-than needs a Redis or S3 store")
    old: list[bytes] = []

    async def check(batch: list[bytes]) -> None:
        ages = await idle(batch)
        old.extend(x for x, a in zip(batch, ages, strict=True) if a is not None and a >= age)

    batch: list[bytes] = []
    async for d in rt.store.scan():
        batch.append(d)
        if len(batch) >= 500:
            await check(batch)
            batch = []
    if batch:
        await check(batch)
    return {"deleted": await rt.purge(old) if old else 0}


def _prepare_export(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if next(out.glob(f"*{_ENTRY}"), None) is not None:
        # Leftover entries would come back on import, purged ones included.
        raise CLIError(f"memcat: {out} already holds an export; export to a new directory")


async def _cmd_export(ctx: _Ctx, args: argparse.Namespace) -> dict[str, Any]:
    store = ctx.runtime().store
    out = Path(args.to)
    try:
        await asyncio.to_thread(_prepare_export, out)
        digests: list[str] = []
        async for d, blob in _read_all(store):
            await asyncio.to_thread((out / f"{d.hex()}{_ENTRY}").write_bytes, blob)
            digests.append(d.hex())
        manifest = {"namespace": ctx.cfg.namespace, "format": 1, "digests": digests}
        text = json.dumps(manifest, indent=2) + "\n"
        await asyncio.to_thread((out / "manifest.json").write_text, text)
    except OSError as e:
        raise CLIError(f"memcat: cannot export to {out}: {e.strerror or e}") from None
    return {"exported": len(digests), "to": str(out)}


def _read_manifest(src: Path) -> tuple[str | None, list[str] | None]:
    """The namespace and digests ``memcat export`` recorded in ``manifest.json``
    (None for whatever it does not say)."""
    try:
        manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise CLIError(f"memcat: cannot read {src / 'manifest.json'}: {e}") from None
    if not isinstance(manifest, dict):
        return None, None
    ns, digests = manifest.get("namespace"), manifest.get("digests")
    return (
        ns if isinstance(ns, str) else None,
        [str(h) for h in digests] if isinstance(digests, list) else None,
    )


async def _cmd_import(ctx: _Ctx, args: argparse.Namespace) -> dict[str, Any]:
    from memcat.core import codec

    src = Path(args.from_dir)
    rt = ctx.runtime()
    if not await asyncio.to_thread(src.is_dir):
        raise CLIError(f"memcat: {src} not found")
    ns, listed = await asyncio.to_thread(_read_manifest, src)
    if ns != ctx.cfg.namespace and not args.force:
        where = f"namespace '{ns}'" if ns is not None else "an unknown namespace (no manifest)"
        raise CLIError(
            f"memcat: {src} holds {where}, not '{ctx.cfg.namespace}'; "
            "pass --force to import it into this namespace anyway"
        )
    if listed is None:
        listed = sorted(p.stem for p in await asyncio.to_thread(lambda: [*src.glob(f"*{_ENTRY}")]))
    n = skipped = 0
    for h in listed:  # only what the export wrote, never other files beside it
        try:
            d = bytes.fromhex(h)  # also keeps names from the manifest inside src
            if len(d) != 16:
                raise ValueError(h)
            blob = await asyncio.to_thread((src / f"{h}{_ENTRY}").read_bytes)
            codec.decode(blob)  # never import something that would not replay
        except (ValueError, OSError, codec.CorruptEntryError):
            skipped += 1
            continue
        await rt.store.put(d, blob, ctx.cfg.ttl_s)
        await rt.store.publish_new(d)
        n += 1
    return {"imported": n, "skipped": skipped}


def _secret(serve_conf: dict[str, Any]) -> str | None:
    secret = serve_conf.get("secret")
    if secret is not None and not isinstance(secret, str):
        raise CLIError('memcat: [serve] secret must be a string (quote it: secret = "...")')
    return secret or os.environ.get("MEMCAT_SERVE_SECRET") or None


def _number(serve_conf: dict[str, Any], key: str, default: Any, kind: type) -> Any:
    try:
        return kind(serve_conf.get(key, default))
    except (TypeError, ValueError):
        raise CLIError(f"memcat: [serve] {key} must be a number") from None


def _make_server(conf: dict[str, Any], args: argparse.Namespace) -> Any:
    from memcat.serve import MAX_TEXT_CHARS, QUEUE_TIMEOUT_S, Server

    spec = _Service(conf)
    make = spec.factory(_memcat_config(conf))
    spec.check(make, spec.sample_rate or 16000)
    serve_conf = _section(conf, "serve")
    pool_size = getattr(args, "pool_size", None) or _number(serve_conf, "pool_size", 4, int)
    max_waiting = serve_conf.get("max_waiting")
    try:
        return Server(
            make,
            pool_size=pool_size,
            secret=_secret(serve_conf),
            max_text_chars=_number(serve_conf, "max_text_chars", MAX_TEXT_CHARS, int),
            max_waiting=None if max_waiting is None else _number(serve_conf, "max_waiting", 0, int),
            queue_timeout_s=_number(serve_conf, "queue_timeout_s", QUEUE_TIMEOUT_S, float),
            insecure_no_auth=bool(getattr(args, "insecure_no_auth", False)),
        )
    except ValueError as e:
        raise CLIError(f"memcat: bad [serve] option: {e}") from None


def _serve(args: argparse.Namespace) -> int:
    from aiohttp import web
    from loguru import logger

    from memcat.serve import is_loopback

    conf = _read_config(Path(args.config))
    serve_conf = _section(conf, "serve")
    host = str(args.host or serve_conf.get("host", "127.0.0.1"))
    if not _secret(serve_conf) and not is_loopback(host):
        if not args.insecure_no_auth:
            raise CLIError(
                f"memcat: refusing to serve on {host} without a secret: anyone who can reach "
                "it could spend your TTS credits. Set [serve] secret or MEMCAT_SERVE_SECRET "
                "(clients send it as X-VAPI-SECRET), or pass --insecure-no-auth if something "
                "in front of memcat already authenticates requests"
            )
        logger.warning(f"memcat serve: listening on {host} with no secret (--insecure-no-auth)")
    server = _make_server(conf, args)
    web.run_app(
        server.app(),
        host=host,
        port=args.port or _number(serve_conf, "port", 8080, int),
        print=None,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    # --config is accepted before or after the command. On the subcommands it defaults
    # to SUPPRESS so that leaving it out there keeps the value given before the command.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS, help="path to memcat.toml")
    p = argparse.ArgumentParser(prog="memcat", description="Memcat TTS cache tools")
    p.add_argument("--config", default="memcat.toml", help="path to memcat.toml")
    sub = p.add_subparsers(dest="cmd", required=True)

    def cmd(name: str, help: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help, parents=[common])

    w = cmd("warm", "render the most repeated clauses ahead of time")
    w.add_argument("--from", dest="from_file", help='calls.jsonl: {"call_id", "agent_turns"}')
    w.add_argument("--top", type=int, default=500)
    w.add_argument("--min-calls", type=int, default=3, help="only clauses heard in this many calls")
    w.add_argument("--phrase", action="append", help="a phrase to render (repeatable)")
    w.add_argument("--dry-run", action="store_true")
    w.add_argument("--timeout", type=float, default=600.0)
    cmd("stats", "what the cache holds")
    pu = cmd("purge", "delete entries")
    which = pu.add_mutually_exclusive_group()
    which.add_argument("--text")
    which.add_argument("--all", action="store_true", help="every entry in the namespace")
    which.add_argument(
        "--older-than",
        help="entries idle this long, e.g. 30d (Redis: since last written or served; "
        "S3: since written — S3 keeps no access time)",
    )
    pu.add_argument("--yes", action="store_true", help="confirm --all without asking")
    e = cmd("export", "copy every entry to a directory")
    e.add_argument("--to", required=True)
    i = cmd("import", "load entries from an export")
    i.add_argument("--from", dest="from_dir", required=True)
    i.add_argument(
        "--force",
        action="store_true",
        help="import even if the export came from another namespace (or has no manifest)",
    )
    sv = cmd("serve", "TTS endpoint for hosted platforms (Vapi custom voice)")
    sv.add_argument("--host", help="address to listen on (default 127.0.0.1)")
    sv.add_argument("--port", type=int)
    sv.add_argument("--pool-size", type=int, help="speakers per sample rate (default 4)")
    sv.add_argument(
        "--insecure-no-auth",
        action="store_true",
        help="allow a non-loopback --host without a secret (only behind your own auth)",
    )
    return p


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    import memcat

    ctx = _Ctx(_read_config(Path(args.config)), args.cmd)
    try:
        cmd = {
            "warm": _cmd_warm,
            "stats": _cmd_stats,
            "purge": _cmd_purge,
            "export": _cmd_export,
            "import": _cmd_import,
        }[args.cmd]
        return await cmd(ctx, args)
    finally:
        await memcat.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns 0 on success; configuration and input errors print
    ``memcat: <message>`` to stderr and return 1 (usage errors exit 2, as argparse does)."""
    args = _parser().parse_args(argv)
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("MEMCAT_LOG", "WARNING"))
    try:
        if args.cmd == "serve":
            return _serve(args)
        result = asyncio.run(_run(args))
    except CLIError as e:
        print(e, file=sys.stderr)
        return 1
    except StoreError as e:  # includes PurgeNotifyError: entries deleted, others not told
        print(f"memcat {args.cmd}: store error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
