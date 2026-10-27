"""
Per-task time budget (PDFLibero).

PDFLibero gives up on a render after 15 minutes and refunds the customer,
but the render server used to keep working on the abandoned task, slowing
down the next customers. A task started with a budget now stops itself at
the next checkpoint once the budget is spent.
"""

import threading
import time

_local = threading.local()


class TaskDeadlineExceeded(RuntimeError):
    pass


def set_deadline(seconds: float | None) -> None:
    _local.deadline = time.monotonic() + seconds if seconds and seconds > 0 else None


def clear() -> None:
    _local.deadline = None


def check(stage: str = "") -> None:
    deadline = getattr(_local, "deadline", None)
    if deadline is not None and time.monotonic() > deadline:
        raise TaskDeadlineExceeded(f"task time budget exceeded during {stage or 'processing'}")
