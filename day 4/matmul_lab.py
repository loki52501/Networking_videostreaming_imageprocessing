"""Day 4 Matrix Multiplication Lab Template.
Fill in the TODO sections while working through the hands-on exercises.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Tuple

import numpy as np


def manual_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute C = A x B using explicit loops.

    Replace the body with triple-loop multiplication that matches `np.dot`.
    """
    # TODO: allocate result array and implement nested loops
    raise NotImplementedError


def threaded_matmul(a: np.ndarray, b: np.ndarray, num_threads: int) -> np.ndarray:
    """Split the manual matmul across multiple threads.

    Suggested approach: divide the outer `i` loop into chunks per thread.
    Capture timing inside your experiment runner, not inside this function.
    """
    # TODO: implement thread-parallel version (can reuse manual_matmul logic per slice)
    raise NotImplementedError


def time_call(fn, *args, **kwargs) -> Tuple[np.ndarray, float]:
    """Utility to time a callable and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def run_experiments() -> None:
    """Set up matrices, run manual vs numpy vs threaded variants, and log findings."""
    # TODO: choose matrix dimensions (start with 100x50 @ 50x30) and seed RNG for reproducibility.
    # TODO: record timings for manual_matmul and np.dot, validate results with np.allclose.
    # TODO: iterate over [1, 2, 4] threads (and optionally higher), logging timing + speedup.
    pass


if __name__ == "__main__":
    # TODO: call run_experiments() and persist your observations (stdout or file).
    pass
