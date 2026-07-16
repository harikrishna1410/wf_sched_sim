import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List


def plot_dashboard(
    results: Dict[str, Dict[str, Any]],
    stage_order: List[str],
    min_makespan: float,
    walltime_limit: float = None,
    save_path: str = "dashboard.png",
):
    """
    Generates a 2-panel dashboard plot comparing makespans and average occupancy.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    policies = list(results.keys())
    makespans = [results[p]["makespan"] for p in policies]
    occupancies = [results[p]["occupancy"] for p in policies]

    # Capitalize policy names for labels
    labels = [p.replace("_", " ").title() for p in policies]

    # Custom color palette
    colors = ["#4A90E2", "#50E3C2", "#F5A623", "#D0021B", "#9013FE", "#417505"][
        : len(policies)
    ]

    # Panel 1: Makespan
    bars1 = ax1.bar(
        labels, makespans, color=colors, alpha=0.85, edgecolor="black", linewidth=1
    )
    ax1.set_ylabel("Makespan (seconds)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Makespan Comparison (Lower is Better)", fontsize=13, fontweight="bold", pad=15
    )
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Add horizontal line for theoretical minimum
    ax1.axhline(
        min_makespan,
        color="#E74C3C",
        linestyle="--",
        linewidth=1.5,
        label=f"Theoretical Min ({min_makespan:.1f}s)",
    )

    # Add walltime limit if set
    if walltime_limit is not None:
        ax1.axhline(
            walltime_limit,
            color="#8E44AD",
            linestyle="-.",
            linewidth=1.5,
            label=f"Walltime Limit ({walltime_limit:.1f}s)",
        )

    ax1.legend(loc="upper right")

    # Add values on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (max(makespans) * 0.01),
            f"{yval:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Panel 2: Occupancy
    bars2 = ax2.bar(
        labels, occupancies, color=colors, alpha=0.85, edgecolor="black", linewidth=1
    )
    ax2.set_ylabel("Average Occupancy (%)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Worker Occupancy Comparison (Higher is Better)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle="--", alpha=0.3)

    # Add values on top of bars
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 1,
            f"{yval:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Dashboard plot saved to {save_path}.")


def plot_timeline(
    results: Dict[str, Dict[str, Any]],
    num_workers: int,
    stage_names: List[str],
    save_path: str = "large_scale_fractions.png",
):
    """
    Generates a multi-panel stacked area timeline plot showing task iterations over time.
    """
    num_policies = len(results)
    fig, axs = plt.subplots(
        num_policies, 1, figsize=(14, 5 * num_policies), sharex=True, sharey=True
    )
    if num_policies == 1:
        axs = [axs]

    # Generate dynamic color map for stages and their iterations
    # Map stages to matplotlib colormaps
    cmaps = ["Blues", "Oranges", "Greens", "Reds", "Purples", "YlOrBr", "PuRd"]

    # Find all categories present across all histories
    # A category looks like "A_1", "A_2", "Idle"
    all_cats = set()
    for p in results:
        hist = results[p]["history"]
        for _, breakdown in hist:
            all_cats.update(breakdown.keys())

    all_cats.discard("Idle")

    # Order categories: stage order, then iteration order
    sorted_cats = []
    for st_name in stage_names:
        # Find all iteration numbers for this stage
        st_cats = [c for c in all_cats if c.startswith(st_name + "_")]
        st_cats.sort(key=lambda x: int(x.split("_")[1]))
        sorted_cats.extend(st_cats)

    # Generate colors
    colors_map = {}
    labels_map = {}

    for st_idx, st_name in enumerate(stage_names):
        cmap_name = cmaps[st_idx % len(cmaps)]
        cmap = plt.get_cmap(cmap_name)

        # Get iterations for this stage
        st_cats = [c for c in sorted_cats if c.startswith(st_name + "_")]
        n_iters = len(st_cats)

        for i_idx, c in enumerate(st_cats):
            iter_num = c.split("_")[1]
            # Map index to color value in range [0.3, 0.9]
            val = 0.3 + 0.6 * (i_idx + 1) / max(n_iters, 1)
            colors_map[c] = cmap(val)

            # Label
            suffix = (
                "+"
                if (st_idx < 2 and iter_num == "4") or (st_idx >= 2 and iter_num == "3")
                else ""
            )
            labels_map[c] = f"{st_name} - Iter {iter_num}{suffix}"

    colors_map["Idle"] = "#E5E8E8"
    labels_map["Idle"] = "Idle Workers"

    max_makespan = max(results[p]["makespan"] for p in results)
    grid_t = np.linspace(0, max_makespan, 1000)

    for i, (p, res) in enumerate(results.items()):
        makespan = res["makespan"]
        hist = res["history"]
        ax = axs[i]

        hist_t = np.array([h[0] for h in hist])
        indices = np.searchsorted(hist_t, grid_t, side="right") - 1
        indices = np.clip(indices, 0, len(hist_t) - 1)

        past_makespan = grid_t > makespan

        y_data = {}
        for c in sorted_cats:
            y_arr = np.array([h[1].get(c, 0) for h in hist])
            y_res = y_arr[indices] / num_workers
            y_res[past_makespan] = 0.0
            y_data[c] = y_res

        y_list = [y_data[c] * 100.0 for c in sorted_cats]

        y_idle = 100.0 - sum(y_list)
        y_idle[past_makespan] = 0.0
        y_idle = np.clip(y_idle, 0.0, 100.0)

        ax.stackplot(
            grid_t,
            *y_list,
            y_idle,
            labels=[labels_map[c] for c in sorted_cats] + [labels_map["Idle"]],
            colors=[colors_map[c] for c in sorted_cats] + [colors_map["Idle"]],
            alpha=0.9,
        )

        ax.set_title(
            f"Worker State Breakdown & Task Iterations - {p.replace('_', ' ').title()}",
            fontsize=13,
            fontweight="bold",
            pad=10,
        )
        ax.set_ylabel("Worker Fraction (%)", fontsize=11, fontweight="bold")
        ax.set_ylim(-2, 102)
        ax.grid(True, linestyle="--", alpha=0.3)

        ax.axvline(
            makespan,
            color="#E74C3C",
            linestyle="--",
            linewidth=1.5,
            label=f"Makespan ({makespan:.1f}s)",
        )

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            fontsize=9,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axs[-1].set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Timeline plot saved to {save_path}.")
