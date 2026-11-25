"""Windows timer-resolution request with guaranteed pairing (ADR 0006 ruling 3).

Each TimerRequest calls timeBeginPeriod at most once and timeEndPeriod at most once,
and only if the begin call succeeded. release() is idempotent. Windows refcounts
these calls per process, so one request per env is enough for correct pairing.
"""
from __future__ import annotations

import sys


def _winmm():  # indirection so tests can substitute a fake
    import ctypes
    return ctypes.WinDLL("winmm")


class TimerRequest:
    def __init__(self, period_ms: int | None):
        self.period = None
        if period_ms and period_ms > 0 and sys.platform == "win32":
            if _winmm().timeBeginPeriod(int(period_ms)) == 0:  # TIMERR_NOERROR
                self.period = int(period_ms)

    @property
    def active(self) -> bool:
        return self.period is not None

    def release(self) -> None:
        p, self.period = self.period, None
        if p is not None:
            _winmm().timeEndPeriod(p)
