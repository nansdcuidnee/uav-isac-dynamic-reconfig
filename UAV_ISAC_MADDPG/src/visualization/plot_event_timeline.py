"""Plot and text timeline helpers for dynamic events."""

from __future__ import annotations

import matplotlib.pyplot as plt


def write_event_timeline_text(event_history, save_path: str) -> None:
    """Write a simple event timeline text file."""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Event Timeline\n")
        f.write("=====================================\n")
        for item in event_history:
            f.write(
                f"Step {item['step']}: {item['event_type']} agent {item['agent_id']} ({item['reason']})\n"
            )
        f.write("=====================================\n")


def plot_event_timeline(event_history, max_steps: int, save_path: str) -> None:
    """Plot a minimal event timeline figure."""
    plt.figure(figsize=(12, 2.8))
    plt.hlines(y=1, xmin=0, xmax=max_steps, color="black", linewidth=1.2)
    for item in event_history:
        x = item["step"]
        label = f"{item['event_type']} UAV-{item['agent_id']}"
        plt.vlines(x=x, ymin=0.88, ymax=1.12, color="#1f77b4", linewidth=2)
        plt.text(x, 1.15, label, rotation=25, ha="left", va="bottom", fontsize=9)
    plt.xlim(0, max_steps)
    plt.ylim(0.8, 1.3)
    plt.yticks([])
    plt.xlabel("Step")
    plt.title("Dynamic Event Timeline")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
