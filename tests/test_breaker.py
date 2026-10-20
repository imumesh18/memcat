from __future__ import annotations

from memcat.core.breaker import Breaker


def test_breaker_cycle() -> None:
    now = [0.0]
    opened = []
    b = Breaker(clock=lambda: now[0], on_open=lambda: opened.append(1))
    for _ in range(4):
        b.failure()
    assert b.allow()
    b.failure()
    assert b.is_open and not b.allow() and opened == [1]
    now[0] += 31
    assert b.allow()  # one probe
    assert not b.allow()  # only one
    b.failure()  # probe failed: open again
    assert not b.allow()
    now[0] += 31
    assert b.allow()
    b.success()
    assert not b.is_open and b.allow()


def test_errors_outside_window_do_not_open() -> None:
    now = [0.0]
    b = Breaker(clock=lambda: now[0])
    for _ in range(10):
        b.failure()
        now[0] += 3
    assert not b.is_open


def test_a_few_slow_reads_under_load_do_not_open() -> None:
    now = [0.0]
    b = Breaker(clock=lambda: now[0])
    for i in range(1000):
        now[0] += 0.005
        if i % 100 == 0:
            b.failure()
        else:
            b.success()
    assert not b.is_open
    now[0] += 11  # later, a real outage: every call fails
    for _ in range(5):
        now[0] += 0.005
        b.failure()
    assert b.is_open
