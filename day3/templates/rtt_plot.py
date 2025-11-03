"""
Template for plotting RTT samples and basic statistics.

Usage:
    python rtt_plot.py --input ../logs/rtt_baseline.txt --title "Baseline RTT"
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RTT measurements over time.")
    parser.add_argument("--input", type=Path, required=True, help="Path to RTT log file.")
    parser.add_argument("--title", type=str, default="RTT vs Time")
    parser.add_argument("--save", type=Path, help="Optional path to save the plot as PNG.")
    return parser.parse_args()


def load_samples(path: Path) -> List[float]:
    if not path.exists():
        raise FileNotFoundError(f"RTT log not found: {path}")
    samples: List[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            # TODO: Adjust parsing logic to match your log format (e.g., "iter RTT_ms").
            parts = line.split()
            try:
                samples.append(float(parts[-1]))
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Could not parse line: {line}") from exc
    return samples


def summarise(samples: List[float]) -> None:
    if not samples:
        raise ValueError("No RTT samples were provided.")
    mean = statistics.mean(samples)
    stdev = statistics.pstdev(samples)
    percentile_95 = statistics.quantiles(samples, n=100)[94]
    print(f"Count: {len(samples)}")
    print(f"Mean RTT: {mean:.3f} ms")
    print(f"p95 RTT: {percentile_95:.3f} ms")
    print(f"Std-dev: {stdev:.3f} ms")


def plot(samples: List[float], title: str, save_path: Path | None) -> None:
    fig, ax = plt.subplots()
    ax.plot(samples, marker="o", linestyle="-", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Probe #")
    ax.set_ylabel("RTT (ms)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.input)
    summarise(samples)
    plot(samples, args.title, args.save)


if __name__ == "__main__":
    main()
