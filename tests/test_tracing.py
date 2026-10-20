from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402
from pipecat.services.tts_service import TTSService  # noqa: E402
from pipecat.utils.tracing.service_decorators import traced_tts  # noqa: E402

from memcat import MemcatConfig, MemoryStore, cached  # noqa: E402
from tests.fakes import FakeELWSTTS, FakeHTTPTTS  # noqa: E402
from tests.harness import Driver  # noqa: E402
from tests.util import drain  # noqa: E402

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)


class TracedHTTP(FakeHTTPTTS):
    @traced_tts
    async def run_tts(self, text: str, context_id: str):  # type: ignore[override]
        async for f in FakeHTTPTTS.run_tts(self, text, context_id):
            yield f


async def test_memcat_attributes_on_tts_span() -> None:
    store = MemoryStore()
    s = "How can I help you today?"
    a = cached(TracedHTTP)(memcat=MemcatConfig(store=store, admit_after_calls=1))
    async with Driver(a, tracing=True) as d:
        await d.say(s)
    await drain(a._memcat_cfg)
    _exporter.clear()
    b = cached(TracedHTTP)(memcat=MemcatConfig(store=store))
    async with Driver(b, tracing=True) as d:
        await d.say(s)
    spans = [sp for sp in _exporter.get_finished_spans() if sp.name == "tts"]
    assert spans, "no tts span"
    attrs = spans[-1].attributes or {}
    assert attrs.get("memcat.hit") is True
    assert attrs.get("memcat.chars_cached") == len(s)
    assert isinstance(TTSService, type)


class TracedEL(FakeELWSTTS):
    @traced_tts
    async def run_tts(self, text: str, context_id: str):  # type: ignore[override]
        async for f in FakeELWSTTS.run_tts(self, text, context_id):
            yield f


async def test_memcat_attributes_on_segmented_tts_span() -> None:
    """Segmented mode opens the turn's context itself (the provider would have, in
    run_tts): through the traced method, so the span exists and gets Memcat's
    attributes. Provider streams inside it leave no spans behind."""
    store = MemoryStore()
    s = "Thanks for waiting today, your order 48213 has shipped."
    hit = "Thanks for waiting today,"
    a = cached(TracedEL)(memcat=MemcatConfig(store=store, admit_after_calls=1))
    async with Driver(a, tracing=True) as d:
        await d.say(hit)
    await drain(a._memcat_cfg)
    _exporter.clear()
    b = cached(TracedEL)(memcat=MemcatConfig(store=store))
    async with Driver(b, tracing=True) as d:
        await d.say(s)
        await asyncio.sleep(0.3)
    spans = [sp for sp in _exporter.get_finished_spans() if sp.name == "tts"]
    hits = [sp for sp in spans if (sp.attributes or {}).get("memcat.hit") is True]
    assert hits, [dict(sp.attributes or {}) for sp in spans]
    assert hits[-1].attributes["memcat.chars_cached"] == len(hit) + 1  # type: ignore[index]
    assert not b._tts_spans
