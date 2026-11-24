"""Wall-clock time that never goes backwards and never jumps.

Parts are linked to machine cycles by time, so a clock correction in the middle of a part can
link it to the wrong cycle or make it look like two parts. Time synchronisation does step
clocks: one development PC was stepped back 0.65 s every 32 s. This clock follows the system
clock by running slightly fast or slow instead. Everything that stamps a cycle or a part must
use it, so both sides of the link share one time base.
"""
from __future__ import annotations

import threading
import time

MAX_SLEW = 0.05          # corrects at most 5% faster or slower than real time
RESYNC_S = 30.0          # a difference this large is a clock set forward, such as at boot


class StationClock:
    def __init__(self, wall=time.time, mono=time.monotonic):
        self._wall, self._mono = wall, mono
        self._lock = threading.Lock()
        self._now, self._last = float(wall()), float(mono())

    def time(self) -> float:
        with self._lock:
            mono = float(self._mono())
            elapsed = max(0.0, mono - self._last)
            self._last = mono
            self._now += elapsed
            error = float(self._wall()) - self._now
            if error > RESYNC_S:
                self._now += error
            else:
                limit = MAX_SLEW * elapsed
                self._now += max(-limit, min(limit, error))
            return self._now


_clock = StationClock()
station_time = _clock.time
