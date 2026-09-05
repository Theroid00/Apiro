"""Bound concurrent model requests and collect process-level Ollama telemetry."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import threading
import time
from typing import Callable, TypeVar


T = TypeVar("T")


class ModelCallScheduler:
    """Serialize accounting around a bounded number of concurrent model calls.

    A single instance is shared by the runtime resources. Each invocation is an
    HTTP attempt, so retries are visible as additional calls. Ollama's
    ``prompt_eval_count`` and ``eval_count`` fields provide exact token counts
    when the server returns them.
    """

    def __init__(self, max_concurrency: int = 2):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self.max_concurrency = max_concurrency
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self._lock = threading.Lock()
        self._totals = self._empty_stats()
        self._by_purpose: dict[str, dict] = defaultdict(self._empty_stats)

    @staticmethod
    def _empty_stats() -> dict:
        return {
            "calls": 0,
            "retries": 0,
            "failures": 0,
            "timeouts": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "queue_seconds": 0.0,
            "inference_seconds": 0.0,
        }

    @staticmethod
    def _tokens(response) -> tuple[int, int]:
        try:
            payload = response.json()
        except Exception:
            return 0, 0
        return (
            int(payload.get("prompt_eval_count") or 0),
            int(payload.get("eval_count") or 0),
        )

    def call(
        self,
        purpose: str,
        operation: Callable[[], T],
        *,
        is_retry: bool = False,
    ) -> T:
        """Run one model HTTP attempt inside the shared concurrency bound."""
        queued_at = time.monotonic()
        with self._semaphore:
            queue_seconds = time.monotonic() - queued_at
            started = time.monotonic()
            try:
                response = operation()
            except Exception as exc:
                self._record(
                    purpose,
                    failed=True,
                    timed_out="timeout" in exc.__class__.__name__.lower(),
                    is_retry=is_retry,
                    queue_seconds=queue_seconds,
                    inference_seconds=time.monotonic() - started,
                )
                raise

            prompt_tokens, completion_tokens = self._tokens(response)
            self._record(
                purpose,
                failed=False,
                timed_out=False,
                is_retry=is_retry,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                queue_seconds=queue_seconds,
                inference_seconds=time.monotonic() - started,
            )
            return response

    def _record(
        self,
        purpose: str,
        *,
        failed: bool,
        timed_out: bool,
        is_retry: bool,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        queue_seconds: float,
        inference_seconds: float,
    ) -> None:
        with self._lock:
            for stats in (self._totals, self._by_purpose[purpose]):
                stats["calls"] += 1
                stats["retries"] += int(is_retry)
                stats["failures"] += int(failed)
                stats["timeouts"] += int(timed_out)
                stats["prompt_tokens"] += prompt_tokens
                stats["completion_tokens"] += completion_tokens
                stats["queue_seconds"] += queue_seconds
                stats["inference_seconds"] += inference_seconds

    def snapshot(self) -> dict:
        """Return a JSON-serializable, internally consistent counter snapshot."""
        with self._lock:
            return {
                "max_concurrency": self.max_concurrency,
                "totals": self._rounded(deepcopy(self._totals)),
                "by_purpose": {
                    purpose: self._rounded(deepcopy(stats))
                    for purpose, stats in sorted(self._by_purpose.items())
                },
            }

    @staticmethod
    def _rounded(stats: dict) -> dict:
        stats["queue_seconds"] = round(stats["queue_seconds"], 6)
        stats["inference_seconds"] = round(stats["inference_seconds"], 6)
        return stats

    def delta(self, before: dict) -> dict:
        """Return counters accrued since an earlier :meth:`snapshot`."""
        after = self.snapshot()
        purposes = set(before.get("by_purpose", {})) | set(after["by_purpose"])

        def subtract(new: dict, old: dict) -> dict:
            return self._rounded({
                key: new.get(key, 0) - old.get(key, 0)
                for key in self._empty_stats()
            })

        return {
            "max_concurrency": self.max_concurrency,
            "totals": subtract(after["totals"], before.get("totals", {})),
            "by_purpose": {
                purpose: subtract(
                    after["by_purpose"].get(purpose, {}),
                    before.get("by_purpose", {}).get(purpose, {}),
                )
                for purpose in sorted(purposes)
                if after["by_purpose"].get(purpose, {})
                != before.get("by_purpose", {}).get(purpose, {})
            },
        }
