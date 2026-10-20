from __future__ import annotations

import os
import sys

import pytest
from loguru import logger

import memcat
import memcat.config
from tests import fakes

logger.remove()
# Look up sys.stderr per message: pytest swaps it during capture and closes the old one.
logger.add(lambda m: sys.stderr.write(m), level=os.environ.get("MEMCAT_TEST_LOG", "WARNING"))


@pytest.fixture(autouse=True)
async def _close_memcat():
    # Each test gets its own fake-voice tones (the table would otherwise grow past
    # what 16 kHz audio can carry) and its own default store.
    fakes._FREQS.clear()
    memcat.config._default_store = None
    yield
    await memcat.aclose()
    memcat.config._default_store = None
