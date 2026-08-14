"""Plot makespan and node utilization vs node count from scaling_results.csv.

Utilization = total_task_work / (workers * makespan), the fraction of available
worker-time spent computing (1 = perfectly packed). total_task_work uses the
expected mean task duration per distribution.
"""
import argparse
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 15,
})

# Expected mean work (seconds) per pipeline for each distribution label.
MEANS = {
    "pinned_default": 36.89 * 60.0,             # 1 stage
    "ordered_default": (11.61 + 26.39) * 60.0,  # 2 stages summed per pipeline
    "ordered_equal_stages": (19.0 + 19.0) * 60.0,
}

STYLE = {
    "pinned": dict(color="tab:blue", marker="o"),
    "ordered": dict(color="tab:green", marker="s"),
}
POLICY_LABELS = {
    "pinned": "Unordered, single-stage",
    "ordered": "Dynamic, multi-stage",
}
TEST_TITLES = {
    "1": "Test 1: fixed 4x ratio (pipes = 4 x workers)",
    "2": "Test 2: fixed 104448 pipelines",
    "3": "Test 3: equalized stages",
}


def style_ax(ax, logx=True):
    if logx:
        ax.set_xscale("log", base=2)
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(True, which="major", alpha=0.3)


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["nodes"] = int(r["nodes"])
        r["nworkers"] = int(r["nworkers"])
        r["pipeline_num"] = int(r["pipeline_num"])
        r["makespan"] = float(r["makespan"])
        work = r["pipeline_num"] * MEANS[r["distribution"]]
        r["util"] = 100.0 * work / (r["nworkers"] * r["makespan"])
    return rows


def series(rows, test, policy):
    sel = sorted((r for r in rows if r["test"] == test and r["policy"] == policy),
                 key=lambda r: r["nodes"])
    return sel


def plot(path, out):
    rows = load(path)
    # Strong-scaling only: Test 2 (fixed 104448 pipelines), single column.
    test = "2"

    fig, (ax_ms, ax_ut) = plt.subplots(
        2, 1, figsize=(7.5, 5.8), sharex=True,
        gridspec_kw={"hspace": 0.0})
    for policy in ("pinned", "ordered"):
        s = series(rows, test, policy)
        if not s:
            continue
        x = [r["nodes"] for r in s]
        ax_ms.plot(x, [r["makespan"] / 60.0 for r in s],
                   label=POLICY_LABELS[policy], **STYLE[policy])
        ax_ut.plot(x, [r["util"] for r in s],
                   label=POLICY_LABELS[policy], **STYLE[policy])
    ax_ms.set_ylabel("Runtime (minutes)", y=0.6)
    ax_ut.set_ylabel("Node utilization (%)", y=0.4)
    ax_ut.set_xlabel("Node count")
    ax_ms.set_ylim(bottom=0)
    ax_ut.set_ylim(0, 105)
    for ax in (ax_ms, ax_ut):
        style_ax(ax)
    # Drop the top panel's lowest y-tick so its label doesn't collide with the
    # bottom panel's top label across the shared (zero-gap) boundary.
    ax_ms.yaxis.set_major_locator(MaxNLocator(nbins="auto", prune="lower"))
    ax_ms.legend()

    ax_ms.set_title("Scheduling Simulations vs Node Count")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {out}")

    # Report the single-point Test 3 for reference.
    for r in (r for r in rows if r["test"] == "3"):
        print(f"Test 3 (n={r['nodes']}, {r['policy']}, {r['distribution']}): "
              f"makespan={r['makespan']/60:.1f} min, util={r['util']:.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="../data/scaling_results.csv")
    p.add_argument("--out", default="plot_scaling.png")
    args = p.parse_args()
    plot(args.inp, args.out)


if __name__ == "__main__":
    main()
