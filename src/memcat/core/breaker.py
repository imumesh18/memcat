"""Circuit breaker around the shared store."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

__all__ = ["Breaker"]

_MAX_OKS = 10_000  # successes remembered in the window; enough to judge the error ratio


class Breaker:
    """Opens when the store is failing, not when it is merely busy.

    It opens after at least ``threshold`` errors in ``window_s`` that are also at
    least ``min_error_ratio`` of all store calls in that window, so a few slow reads
    among thousands under load do not take the cache offline. It stays open
    ``cooldown_s``; after that one request is let through as a probe, and any
    success closes it.

    A probe that never reports back (its caller was cancelled by a barge-in, or
    hung) must not keep the breaker open forever: callers report a cancelled
    call with :meth:`abandon`, and a probe still out after ``probe_timeout_s``
    is given up so the next request probes again.
    """

    def __init__(
        self,
        threshold: int = 5,
        window_s: float = 10.0,
        cooldown_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        on_open: Callable[[], None] | None = None,
        min_error_ratio: float = 0.5,
        probe_timeout_s: float = 15.0,
    ):
        self._threshold = threshold
        self._window = window_s
        self._cooldown = cooldown_s
        self._ratio = min_error_ratio
        self._clock = clock
        self._errors: deque[float] = deque()
        self._oks: deque[float] = deque()
        self._open_until = 0.0
        self._open = False
        self._probing = False
        self._probe_deadline = 0.0
        self._probe_timeout = probe_timeout_s
        self._on_open = on_open

    @property
    def is_open(self) -> bool:
        return self._open

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        while self._errors and self._errors[0] < cutoff:
            self._errors.popleft()
        while self._oks and self._oks[0] < cutoff:
            self._oks.popleft()

    def allow(self) -> bool:
        if not self._open:
            return True
        now = self._clock()
        if self._probing and now < self._probe_deadline:
            return False
        if now < self._open_until:
            return False
        self._probing = True  # a new probe, or one replacing a probe that never reported
        self._probe_deadline = now + self._probe_timeout
        return True

    def abandon(self) -> None:
        """A store call let through by :meth:`allow` was cancelled before it finished.
        If it was the probe, the next request probes instead. (A cancelled call that
        was not the probe may free the slot early; that costs one extra probe.)"""
        self._probing = False

    def success(self) -> None:
        """A store call worked. Closes the breaker once its cooldown has passed, whether
        the call was the lookup probe or background work (a write after recovery)."""
        now = self._clock()
        if self._open:
            if self._probing or now >= self._open_until:
                self._open = False
                self._probing = False
                self._errors.clear()
                self._oks.clear()
            return
        self._oks.append(now)
        if len(self._oks) > _MAX_OKS:
            self._oks.popleft()

    def failure(self) -> None:
        now = self._clock()
        if self._open:
            if self._probing:
                self._probing = False
                self._open_until = now + self._cooldown
            return
        self._errors.append(now)
        self._prune(now)
        n_err = len(self._errors)
        if n_err >= self._threshold and n_err / (n_err + len(self._oks)) >= self._ratio:
            self._open = True
            self._open_until = now + self._cooldown
            self._errors.clear()
            self._oks.clear()
            if self._on_open is not None:
                self._on_open()
