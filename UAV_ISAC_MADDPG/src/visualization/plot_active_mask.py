"""Plot helpers for active-mask evolution."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_active_mask(active_mask_history, save_path: str) -> None:
    """Render active mask history as a binary heatmap."""
    mask_array = np.array(active_mask_history, dtype=int)
    plt.figure(figsize=(12, 4))
    plt.imshow(mask_array.T, aspect="auto", cmap="binary", interpolation="none")
    plt.xlabel("Step")
    plt.ylabel("Agent ID")
    plt.title("Active Mask Changes Over Steps")
    plt.yticks(range(mask_array.shape[1]))
    plt.xticks(range(0, mask_array.shape[0], 5))
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.set_label("Active Status")
    cbar.set_ticklabels(["Inactive", "Active"])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
