"""Event-driven execution trace (Phase A6).

TraceRecorder produces real TraceEvent objects with real wall-clock
timestamps and durations measured from actual work — never a simulated or
pre-scripted sequence. The frontend (Phase C3) replays exactly this list;
it must never invent steps that didn't happen or hide ones that failed.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from app.contracts import TraceEvent, TraceStatus, TraceStep


class TraceRecorder:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []
        self._starts: dict[str, float] = {}

    def start(self, step: TraceStep) -> None:
        self._starts[step] = time.perf_counter()

    def record(
        self,
        step: TraceStep,
        status: TraceStatus,
        detail: str = "",
        data: dict | None = None,
    ) -> TraceEvent:
        start = self._starts.get(step)
        duration_ms = max(0, int((time.perf_counter() - start) * 1000)) if start is not None else 0
        event = TraceEvent(
            step=step,
            status=status,
            detail=detail,
            data=data or {},
            timestamp=datetime.now(timezone.utc),
            duration_ms=duration_ms,
        )
        self.events.append(event)
        return event

    def error(self, detail: str, data: dict | None = None) -> TraceEvent:
        """Append a terminal ERROR step, in addition to whatever FAILED step
        preceded it — so a client can find failures either way."""
        return self.record("ERROR", "FAILED", detail, data)

    def as_dicts(self) -> list[dict]:
        return [e.model_dump(mode="json") for e in self.events]
