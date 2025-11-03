"""Optional Day 4 challenge: mini distributed training simulator.
Fill in producer threads that compute partial gradients and an aggregator that synchronizes them.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Callable, List


@dataclass
class GradientChunk:
    worker_id: int
    values: List[float]
    ready_at: float


def gradient_worker(worker_id: int, aggregator: "GradientAggregator", work_fn: Callable[[], List[float]]):
    """Worker template: compute partial gradients then hand them to the aggregator."""
    # TODO: call work_fn(), track timing, and push GradientChunk to aggregator.queue
    raise NotImplementedError


class GradientAggregator:
    """Collects gradient chunks and measures wait times before update."""

    def __init__(self, expected_workers: int):
        self.expected_workers = expected_workers
        self.queue: "Queue[GradientChunk]" = Queue()
        self.update_ready = threading.Event()
        self.batches: List[GradientChunk] = []
        self.lock = threading.Lock()
        self.wait_times: List[float] = []

    def collect(self):
        """Blocking collection loop. Stop once all workers have delivered a chunk."""
        # TODO: block until queue has expected_workers items, log any idle time
        raise NotImplementedError

    def aggregate(self) -> List[float]:
        """Combine gradient chunks (e.g., average element-wise)."""
        # TODO: implement simple averaging of worker gradients
        raise NotImplementedError


def simulate_round(num_workers: int = 2):
    """Kick off workers + aggregator for one training round."""
    # TODO: wire up worker threads, aggregator collection, and timing output
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: run multiple rounds, capture aggregator wait times, and print summary.
    pass
