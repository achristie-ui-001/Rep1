import time
import threading


class RateLimiter:
    """Token-bucket rate limiter. Thread-safe."""

    def __init__(self, rate_per_second: float):
        self.rate = rate_per_second
        self.tokens = rate_per_second
        self.last_update = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return

        # Not enough tokens — wait outside the lock
        wait_time = (1 - self.tokens) / self.rate
        time.sleep(wait_time)
        with self._lock:
            self.tokens = 0
            self.last_update = time.monotonic()
