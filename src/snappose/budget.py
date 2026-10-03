from __future__ import annotations

import time

# Default split of the time budget across stages (spec 3.3.3); remainder is reserve.
SHARES = {"s0": 0.05, "s2": 0.05, "s3": 0.55, "s4": 0.05, "s5": 0.20}


class Budget:
    def __init__(self, total_ms: float | None):
        self.t0 = time.perf_counter()
        self.total_ms = total_ms if total_ms and total_ms > 0 else None

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0

    def remaining_ms(self) -> float:
        return float("inf") if self.total_ms is None else self.total_ms - self.elapsed_ms()

    @property
    def deadline(self) -> float | None:
        return None if self.total_ms is None else self.t0 + self.total_ms / 1000.0

    def reserve_ms(self, *stages: str) -> float:
        """Time to keep free for the given later stages."""
        if self.total_ms is None:
            return 0.0
        return self.total_ms * sum(SHARES[s] for s in stages)

    def exhausted(self) -> bool:
        return self.remaining_ms() <= 0
