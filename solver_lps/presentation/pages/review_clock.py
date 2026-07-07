"""Single playback clock shared by every review source."""

from __future__ import annotations

import time


class ReviewClock:
    def __init__(self, sources=(), *, time_fn=time.monotonic):
        self._time_fn = time_fn
        self._sources = []
        self.position_s = 0.0
        self.paused = False
        self._started_at = self._time_fn()
        self.replace_sources(sources)

    @property
    def duration_s(self):
        durations = [
            float(getattr(source, "playback_duration_s", 0.0) or 0.0)
            for source in self._sources
        ]
        positive = [duration for duration in durations if duration > 0.0]
        return min(positive) if positive else 0.0

    @property
    def state(self):
        return {
            "paused": self.paused,
            "position_s": self.position_s,
            "duration_s": self.duration_s,
        }

    def replace_sources(self, sources):
        self._sources = [source for source in sources if source is not None]
        self._push_position()

    def _clamp(self, position_s):
        duration_s = self.duration_s
        position_s = max(0.0, float(position_s))
        return min(position_s, duration_s) if duration_s > 0.0 else position_s

    def _push_position(self):
        for source in self._sources:
            source.set_playback(position_s=self.position_s, paused=True)

    def sample(self):
        if not self.paused:
            self.position_s = self._clamp(self._time_fn() - self._started_at)
            if self.duration_s > 0.0 and self.position_s >= self.duration_s:
                self.paused = True
        self._push_position()
        return self.position_s

    def seek_absolute(self, position_s, *, paused=True):
        self.position_s = self._clamp(position_s)
        self.paused = bool(paused)
        self._started_at = self._time_fn() - self.position_s
        self._push_position()
        return self.position_s

    def seek_relative(self, delta_s):
        self.sample()
        return self.seek_absolute(self.position_s + float(delta_s), paused=True)

    def seek_frames(self, delta_frames):
        periods = [
            float(getattr(source, "nominal_frame_period_s", 0.0) or 0.0)
            for source in self._sources
        ]
        positive = [period for period in periods if period > 0.0]
        frame_period_s = min(positive) if positive else 1.0 / 25.0
        return self.seek_relative(int(delta_frames) * frame_period_s)

    def toggle_pause(self):
        self.sample()
        self.paused = not self.paused
        self._started_at = self._time_fn() - self.position_s
        self._push_position()
        return self.paused
