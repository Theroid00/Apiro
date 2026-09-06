"""Tests for bounded model execution and Ollama usage accounting."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time

from apiro.application.model_scheduler import ModelCallScheduler


class _Response:
    def __init__(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def json(self):
        return {
            "prompt_eval_count": self.prompt_tokens,
            "eval_count": self.completion_tokens,
        }


def test_scheduler_counts_tokens_failures_and_purposes():
    scheduler = ModelCallScheduler(max_concurrency=2)
    before = scheduler.snapshot()

    scheduler.call("generation", lambda: _Response(11, 3))
    try:
        scheduler.call(
            "entropy",
            lambda: (_ for _ in ()).throw(TimeoutError("fail")),
            is_retry=True,
        )
    except TimeoutError:
        pass

    delta = scheduler.delta(before)
    assert delta["totals"]["calls"] == 2
    assert delta["totals"]["failures"] == 1
    assert delta["totals"]["timeouts"] == 1
    assert delta["totals"]["retries"] == 1
    assert delta["totals"]["prompt_tokens"] == 11
    assert delta["totals"]["completion_tokens"] == 3
    assert delta["by_purpose"]["generation"]["calls"] == 1
    assert delta["by_purpose"]["entropy"]["failures"] == 1


def test_scheduler_enforces_global_concurrency_bound():
    scheduler = ModelCallScheduler(max_concurrency=2)
    state = {"active": 0, "peak": 0}
    lock = threading.Lock()

    def operation():
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.02)
        with lock:
            state["active"] -= 1
        return _Response()

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda _index: scheduler.call("test", operation), range(6)))

    assert state["peak"] == 2
    assert scheduler.snapshot()["totals"]["calls"] == 6
