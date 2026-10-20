"""``memcat serve``: Memcat as a TTS endpoint for hosted voice platforms.

Platforms such as Vapi call a custom TTS URL with text and a sample rate and play
the raw PCM they get back. ``memcat serve`` answers those requests through a
Memcat-wrapped Pipecat service, so repeated lines come from the cache and only the
rest reaches the provider.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import ipaddress
import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aiohttp import web
from loguru import logger
from pipecat.frames.frames import (
    ControlFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    InterruptionWorkerFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from memcat.pipecat import compat

__all__ = [
    "ANONYMOUS_CALL",
    "SUPPORTED_RATES",
    "ProviderError",
    "Server",
    "Speaker",
    "SpeakerBusy",
    "SpeakerUnavailable",
    "is_loopback",
]

SUPPORTED_RATES = (8000, 16000, 22050, 24000, 44100, 48000)
REQUEST_TIMEOUT_S = 30.0
START_TIMEOUT_S = 30.0
RESET_TIMEOUT_S = 5.0
MAX_TEXT_CHARS = 2000  # default cap on one request's text
QUEUE_TIMEOUT_S = 10.0  # default longest wait for a free speaker
WAITING_PER_SPEAKER = 4  # default queue bound: this many waiting requests per speaker
ANONYMOUS_CALL = "memcat-serve:anonymous"  # the one call every request without an id shares


def is_loopback(host: str) -> bool:
    """True for ``localhost`` and loopback addresses."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # a hostname or interface name: assume it is reachable from outside


def _local_host(request: web.Request) -> bool:
    try:
        host = request.url.host
    except ValueError:  # a malformed Host header
        return False
    return host is not None and is_loopback(host)


class ProviderError(Exception):
    """The pipeline reported an error while speaking a request."""


class SpeakerUnavailable(Exception):
    """The speaker's pipeline could not start or has stopped."""


class SpeakerBusy(Exception):
    """No speaker became free in time."""


@dataclass
class _Mark(ControlFrame):
    """Queued ahead of each request's text. The service emits it after every earlier
    utterance's frames, so frames before it never belong to the request."""

    request: int = 0


class _Request:
    """One request's view of the pipeline output: audio for its own TTS context only."""

    def __init__(self, rid: int):
        self.id = rid
        self.queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue()
        self.live = False  # set by the request's _Mark; cleared by its TTSStoppedFrame
        self.context_id: str | None = None
        # An error seen before the marker passed: maybe this request's, maybe earlier
        # interrupted work's. Kept only to explain a later failure of this request.
        self.early_error: Exception | None = None

    def feed(self, frame: Frame) -> None:
        if not isinstance(frame, (TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame)):
            return
        cid = frame.context_id
        if cid:
            if self.context_id is None:
                self.context_id = cid
            elif cid != self.context_id:
                return
        if isinstance(frame, TTSAudioRawFrame):
            if frame.audio:
                self.queue.put_nowait(frame.audio)
        elif isinstance(frame, TTSStoppedFrame):
            self.live = False
            self.queue.put_nowait(None)

    def fail(self, exc: Exception) -> None:
        self.queue.put_nowait(exc)


class _Tap(FrameProcessor):
    """Hands the current request its audio, and reports when an interruption has passed."""

    def __init__(self) -> None:
        super().__init__()
        self.request: _Request | None = None
        self.flushed: asyncio.Event | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        req = self.request
        if isinstance(frame, _Mark):
            if req is not None and frame.request == req.id:
                req.live = True
            return
        if isinstance(frame, InterruptionFrame) and self.flushed is not None:
            self.flushed.set()
        if req is not None and req.live:
            req.feed(frame)
        await self.push_frame(frame, direction)


class Speaker:
    """One wrapped TTS service in its own pipeline, speaking one request at a time."""

    _ids = itertools.count(1)

    def __init__(self, make_service: Callable[[], Any], sample_rate: int):
        self._make = make_service
        self.sample_rate = sample_rate
        self.lock = asyncio.Lock()
        self.load = 0  # requests assigned to this speaker: speaking plus waiting
        self._tap = _Tap()
        self.service: Any = None
        self._worker: Any = None
        self._task: asyncio.Task[None] | None = None
        self._starting: asyncio.Future[None] | None = None
        self._broken = False
        self._stopping: asyncio.Future[None] | None = None

    @property
    def dead(self) -> bool:
        """The pipeline ran and has since stopped, or could not be cleared after a
        failed request (late frames from it could reach the next one)."""
        return self._broken or (self._task is not None and self._task.done())

    async def start(self) -> None:
        """Start the pipeline once; concurrent and later callers wait for the same start."""
        if self._starting is None:
            self._starting = asyncio.ensure_future(self._start())
        await asyncio.shield(self._starting)

    async def _start(self) -> None:
        svc = self._make()
        compat.disable_playback_pause(svc)  # no transport will report playback ending
        self.service = svc
        self._worker = PipelineWorker(
            Pipeline([svc, self._tap]),
            params=PipelineParams(audio_out_sample_rate=self.sample_rate, enable_metrics=False),
            cancel_on_idle_timeout=False,
            idle_timeout_secs=None,
            enable_rtvi=False,
            enable_turn_tracking=False,
            check_dangling_tasks=False,
        )
        started = asyncio.Event()

        @self._worker.event_handler("on_pipeline_started")
        async def _started(_w: Any, _f: Any) -> None:
            started.set()

        @self._worker.event_handler("on_pipeline_error")
        async def _error(_w: Any, frame: ErrorFrame) -> None:
            req = self._tap.request
            if req is None:
                return
            err = ProviderError(str(frame.error))
            if req.live:
                req.fail(req.early_error or err)
            elif req.early_error is None:
                req.early_error = err

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(self._worker)
        self._task = asyncio.ensure_future(runner.run())
        self._task.add_done_callback(self._on_stopped)
        waiter = asyncio.ensure_future(started.wait())
        await asyncio.wait(
            {waiter, self._task}, timeout=START_TIMEOUT_S, return_when=asyncio.FIRST_COMPLETED
        )
        if not started.is_set():
            waiter.cancel()
            self._task.cancel()
            raise SpeakerUnavailable("pipeline did not start")

    def _on_stopped(self, _task: asyncio.Task[None]) -> None:
        req = self._tap.request
        if req is not None:
            req.fail(SpeakerUnavailable("pipeline stopped"))

    async def speak(
        self,
        text: str,
        call_id: str,
        write: Callable[[bytes], Any],
        *,
        queue_timeout: float | None = None,
        abandoned: Callable[[], bool] = lambda: False,
    ) -> int:
        """Speak ``text`` for ``call_id``, passing audio to ``write`` as it arrives.

        Waits at most ``queue_timeout`` seconds for earlier requests (``SpeakerBusy``).
        If ``abandoned()`` is true once it is this request's turn, nothing is synthesized
        and ``ConnectionResetError`` is raised. Otherwise raises ``TimeoutError``,
        ``ProviderError`` or ``SpeakerUnavailable``, or whatever ``write`` raises. However
        the request ends, the next one hears none of it.
        """
        await self.start()
        try:
            async with asyncio.timeout(queue_timeout):
                await self.lock.acquire()
        except TimeoutError:
            raise SpeakerBusy("no speaker became free in time") from None
        try:
            if self.dead:
                raise SpeakerUnavailable("pipeline stopped")
            if abandoned():
                raise ConnectionResetError("client went away before its turn")
            req = _Request(next(self._ids))
            self._tap.request = req
            # Admission counts distinct calls, so each utterance is attributed to the
            # platform's call. ``_memcat_call_id`` is Memcat's own attribute (the mixin sets
            # a fresh one per pipeline start), not Pipecat internals.
            self.service._memcat_call_id = call_id
            n = 0
            clean = False
            try:
                await self._worker.queue_frames(
                    [_Mark(request=req.id), TTSSpeakFrame(text, append_to_context=False)]
                )
                deadline = time.monotonic() + REQUEST_TIMEOUT_S
                while True:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise TimeoutError
                    item = await asyncio.wait_for(req.queue.get(), left)
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    n += len(item)
                    try:
                        await asyncio.wait_for(write(item), deadline - time.monotonic())
                    except TimeoutError:
                        raise ConnectionResetError("client stopped reading") from None
                clean = True
            finally:
                self._tap.request = None
                if not clean:
                    await self._reset()
            return n
        finally:
            self.lock.release()

    async def _reset(self) -> None:
        """Interrupt whatever the pipeline is still saying and wait for it to clear."""
        if self.dead:
            return
        flushed = asyncio.Event()
        self._tap.flushed = flushed
        try:
            await self._worker.queue_frame(InterruptionWorkerFrame(), FrameDirection.UPSTREAM)
            await asyncio.wait_for(flushed.wait(), RESET_TIMEOUT_S)
        except Exception as e:
            # A late interruption could cut off the next request: retire this speaker.
            logger.warning(f"memcat serve: interrupting the speaker failed ({type(e).__name__})")
            self._broken = True
            self._stopping = asyncio.ensure_future(self.stop())
        finally:
            self._tap.flushed = None

    async def stop(self) -> None:
        if self._starting is not None and not self._starting.done():
            self._starting.cancel()
            with contextlib.suppress(BaseException):
                await self._starting
        if self._task is None:
            return
        with contextlib.suppress(Exception):
            await self._worker.queue_frame(EndFrame())
        try:
            await asyncio.wait_for(asyncio.shield(self._task), 5)
        except BaseException:
            self._task.cancel()


class Server:
    """A pool of speakers per sample rate behind an aiohttp app."""

    def __init__(
        self,
        make_service: Callable[[int], Any],
        *,
        pool_size: int = 4,
        secret: str | None = None,
        max_text_chars: int = MAX_TEXT_CHARS,
        max_waiting: int | None = None,
        queue_timeout_s: float = QUEUE_TIMEOUT_S,
        insecure_no_auth: bool = False,
    ):
        """``max_waiting`` bounds the requests per sample rate waiting for a speaker
        (default ``WAITING_PER_SPEAKER * pool_size``); past it, requests get 503.

        Without a ``secret`` the server answers only loopback clients that are not web
        pages: requests carrying an ``Origin`` header or a ``Host`` other than a
        loopback address get 403, so a page in the operator's browser cannot spend TTS
        credits (DNS rebinding included). ``insecure_no_auth`` lifts that for a server
        behind something else that authenticates requests."""
        if secret is not None and not isinstance(secret, str):
            raise ValueError("secret must be a string")
        if pool_size < 1:
            raise ValueError("pool_size must be at least 1")
        if max_text_chars < 1:
            raise ValueError("max_text_chars must be at least 1")
        if max_waiting is None:
            max_waiting = WAITING_PER_SPEAKER * pool_size
        if max_waiting < 0:
            raise ValueError("max_waiting must not be negative")
        if queue_timeout_s <= 0:
            raise ValueError("queue_timeout_s must be positive")
        self._make = make_service
        self._pool_size = pool_size
        self._secret = secret
        self._local_only = not secret and not insecure_no_auth
        self._max_text_chars = max_text_chars
        self._max_waiting = max_waiting
        self._queue_timeout = queue_timeout_s
        self._pools: dict[int, list[Speaker]] = {}
        self._served = 0
        self._rejected_busy = 0

    def _acquire(self, rate: int) -> Speaker | None:
        """Assign a request to the least-loaded speaker for ``rate``, adding one while the
        pool has room and every speaker is busy. None when the pool and its queue are full.
        The caller must :meth:`_release` the speaker."""
        pool = self._pools.setdefault(rate, [])
        pool[:] = [s for s in pool if not s.dead]
        if sum(s.load for s in pool) >= self._pool_size + self._max_waiting:
            return None
        best = min(pool, key=lambda s: s.load, default=None)
        if best is None or (best.load > 0 and len(pool) < self._pool_size):
            best = Speaker(lambda: self._make(rate), rate)
            pool.append(best)
        best.load += 1
        return best

    def _release(self, sp: Speaker) -> None:
        sp.load -= 1

    def _drop(self, sp: Speaker) -> None:
        pool = self._pools.get(sp.sample_rate, [])
        if sp in pool:
            pool.remove(sp)

    def app(self) -> web.Application:
        app = web.Application(client_max_size=1 << 20, middlewares=[self._guard])
        app.router.add_post("/vapi", self._vapi)
        app.router.add_post("/v1/tts", self._generic)
        app.router.add_get("/healthz", self._health)
        app.on_cleanup.append(self._cleanup)
        return app

    async def _cleanup(self, _app: web.Application) -> None:
        for pool in self._pools.values():
            for sp in pool:
                await sp.stop()
        import memcat

        await memcat.aclose()

    @web.middleware
    async def _guard(self, request: web.Request, handler: Any) -> web.StreamResponse:
        if self._local_only and ("Origin" in request.headers or not _local_host(request)):
            return web.json_response({"error": "forbidden"}, status=403)
        if request.method == "POST":
            if not self._authorized(request):
                return web.json_response({"error": "unauthorized"}, status=401)
            # Browsers send other types cross-site without asking (CORS "simple" requests).
            if request.content_type != "application/json":
                return web.json_response(
                    {"error": "Content-Type must be application/json"}, status=415
                )
        return await handler(request)

    def _authorized(self, request: web.Request) -> bool:
        if not self._secret:
            return True
        got = request.headers.get("X-VAPI-SECRET", "")
        return hmac.compare_digest(got.encode(), self._secret.encode())

    async def _vapi(self, request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
            msg = body["message"]
            if msg.get("type") != "voice-request":
                raise ValueError("message.type must be voice-request")
            text = str(msg["text"])
            rate = int(msg["sampleRate"])
            call_id = str((msg.get("call") or {}).get("id") or ANONYMOUS_CALL)
        except Exception as e:
            return web.json_response({"error": f"bad request: {type(e).__name__}"}, status=400)
        return await self._respond(request, text, rate, call_id)

    async def _generic(self, request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
            text = str(body["text"])
            rate = int(body.get("sample_rate", 24000))
            call_id = str(body.get("call_id") or ANONYMOUS_CALL)
        except Exception as e:
            return web.json_response({"error": f"bad request: {type(e).__name__}"}, status=400)
        return await self._respond(request, text, rate, call_id)

    async def _respond(
        self, request: web.Request, text: str, rate: int, call_id: str
    ) -> web.StreamResponse:
        if not text.strip():
            return web.json_response({"error": "text is empty"}, status=400)
        if len(text) > self._max_text_chars:
            return web.json_response(
                {"error": "text too long", "maxChars": self._max_text_chars}, status=413
            )
        if rate not in SUPPORTED_RATES:
            return web.json_response(
                {"error": "unsupported sampleRate", "supportedRates": list(SUPPORTED_RATES)},
                status=400,
            )
        speaker = self._acquire(rate)
        if speaker is None:
            self._rejected_busy += 1
            return self._busy()
        try:
            return await self._speak(request, speaker, text, call_id)
        finally:
            self._release(speaker)

    async def _speak(
        self, request: web.Request, speaker: Speaker, text: str, call_id: str
    ) -> web.StreamResponse:
        try:
            await speaker.start()
        except Exception as e:
            self._drop(speaker)
            logger.error(f"memcat serve: speaker failed to start ({type(e).__name__}: {e})")
            return web.json_response(
                {"error": "speaker unavailable"}, status=503, headers={"Retry-After": "5"}
            )

        resp = web.StreamResponse(status=200, headers={"Content-Type": "application/octet-stream"})
        resp.enable_chunked_encoding()
        prepared = False

        async def write(chunk: bytes) -> None:
            nonlocal prepared
            if not prepared:
                await resp.prepare(request)
                prepared = True
            await resp.write(chunk)

        def abandoned() -> bool:
            transport = request.transport
            return transport is None or transport.is_closing()

        # Errors before the first byte get a status; after it, the stream just ends short.
        try:
            await speaker.speak(
                text, call_id, write, queue_timeout=self._queue_timeout, abandoned=abandoned
            )
        except SpeakerBusy:
            self._rejected_busy += 1
            return self._busy()
        except ConnectionError:
            logger.debug("memcat serve: client disconnected or stopped reading mid-stream")
            if request.transport is not None:
                # A client that stopped reading would otherwise stall the final write.
                request.transport.abort()
            return resp
        except TimeoutError:
            if not prepared:
                return web.json_response({"error": "timeout"}, status=504)
            logger.warning("memcat serve: synthesis timed out mid-stream")
        except ProviderError as e:
            if not prepared:
                return web.json_response(
                    {"error": "provider error", "detail": str(e)[:200]}, status=502
                )
            logger.warning(f"memcat serve: provider error mid-stream ({type(e).__name__})")
            logger.debug(f"memcat serve: provider error mid-stream: {e}")
        except SpeakerUnavailable:
            self._drop(speaker)
            if not prepared:
                return web.json_response(
                    {"error": "speaker unavailable"}, status=503, headers={"Retry-After": "5"}
                )
            logger.warning("memcat serve: speaker stopped mid-stream")
        else:
            if not prepared:
                return web.json_response({"error": "no audio produced"}, status=502)
            self._served += 1
        with contextlib.suppress(ConnectionError):
            await resp.write_eof()
        return resp

    @staticmethod
    def _busy() -> web.Response:
        return web.json_response({"error": "busy"}, status=503, headers={"Retry-After": "1"})

    async def _health(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            return web.json_response({"ok": True})  # liveness only: counters need the secret
        stats: dict[str, Any] = {
            "ok": True,
            "served": self._served,
            "rejected_busy": self._rejected_busy,
            "speakers": {r: len(p) for r, p in self._pools.items()},
        }
        svc = next((s.service for p in self._pools.values() for s in p if s.service), None)
        if svc is not None:
            stats["cache"] = (await svc.memcat_stats()).get("process", {})
        return web.json_response(stats)
