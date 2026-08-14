"""Plot makespan and node utilization vs std dev for the fine sweep.

Reads variance_fine_results.csv (from variance_fine_runner.py). Four series,
each distinguished by color + marker + linestyle:

  A_ord  ordered, both stages vary
  B_ord  ordered, stage-2 varies (stage-1 fixed)
  D_ord  ordered, stage-1 varies (stage-2 fixed)
  pin    pinned,  mean=20

Two figures are written: makespan (mins) and node utilization (%), both vs the
swept standard deviation (mins).
"""
import argparse
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
})

# color + marker + linestyle all distinct per series.
SERIES = {
    "A_ord": dict(color="tab:green",  marker="s", linestyle="-",
                  label=r"$\mu_1=\mu_2 = 10$, $\sigma_1=\sigma_2$ varies"),
    "B_ord": dict(color="tab:blue",   marker="o", linestyle="--",
                  label=r"$\mu_1=\mu_2 = 10$, $\sigma_1=1$, $\sigma_2$ varies"),
    "D_ord": dict(color="tab:red",    marker="^", linestyle=":",
                  label=r"$\mu_1=\mu_2 = 10$, $\sigma_2=1$, $\sigma_1$ varies"),
    "pin":   dict(color="tab:purple", marker="D", linestyle="-.",
                  label=r"$\mu=20$, $\sigma$ varies"),
}
ORDER = ["A_ord", "B_ord", "D_ord", "pin"]


def style_ax(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(True, which="major", alpha=0.3)


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["stdev"] = float(r["stdev"])
        r["makespan"] = float(r["makespan"])
        r["nworkers"] = int(r["nworkers"])
        r["pipeline_num"] = int(r["pipeline_num"])
        s1 = float(r["stage1_mean"])
        s2 = float(r["stage2_mean"]) if r["stage2_mean"] not in ("", None) else 0.0
        work_s = r["pipeline_num"] * (s1 + s2) * 60.0
        r["util"] = 100.0 * work_s / (r["nworkers"] * r["makespan"])
    return rows


def series(rows, key):
    return sorted((r for r in rows if r["series"] == key),
                  key=lambda r: r["stdev"])


def grouped_legend(ax, loc="best", bbox_to_anchor=None):
    """Legend with 'ordered'/'pinned' section headers above their line groups.

    Header rows use an invisible handle so only the text shows; the empty handle
    slot is then stripped from each header row so its text is flush with the
    legend's left edge.
    """
    blank = Line2D([], [], linestyle="none", marker="none")
    handles = {key: Line2D([], [], **SERIES[key]) for key in ORDER}
    hdr_ordered = "Dynamic, multi-stage"
    hdr_pinned = "Unordered, single-stage"
    headers = {hdr_ordered, hdr_pinned}
    entries = [
        (blank, hdr_ordered),
        (handles["A_ord"], SERIES["A_ord"]["label"]),
        (handles["B_ord"], SERIES["B_ord"]["label"]),
        (handles["D_ord"], SERIES["D_ord"]["label"]),
        (blank, hdr_pinned),
        (handles["pin"], SERIES["pin"]["label"]),
    ]
    h, l = zip(*entries)
    leg = ax.legend(h, l, loc=loc, bbox_to_anchor=bbox_to_anchor,
                    handletextpad=0.6, labelspacing=0.35,
                    fontsize=plt.rcParams["legend.fontsize"] * 0.9)

    # Strip the (empty) handle box from header rows so their text is flush-left.
    for col in leg._legend_handle_box.get_children():   # VPacker per column
        for row in col.get_children():                  # HPacker per row
            text_area = row.get_children()[-1]
            txt = text_area.get_children()[0]
            if txt.get_text() in headers:
                row._children = [text_area]


def plot_metric(rows, out, ykey, ylabel, title, ylim=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    for key in ORDER:
        s = series(rows, key)
        if not s:
            continue
        x = [r["stdev"] for r in s]
        y = [(r[ykey] / 60.0 if ykey == "makespan" else r[ykey]) for r in s]
        ax.plot(x, y, **SERIES[key])
    ax.set_xlabel(r"Task-duration $\sigma$ (minutes)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=0)
    style_ax(ax)
    grouped_legend(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {out}")


def plot_stacked(rows, out, title, ms_ylim, util_ylim=(0, 105)):
    """Compact two-panel column: runtime (top) over node util (bottom).

    Shared x-axis, x tick labels only on the bottom panel, and zero vertical
    space between the panels.
    """
    fig, (ax_ms, ax_ut) = plt.subplots(
        2, 1, figsize=(7.5, 5.8), sharex=True,
        gridspec_kw={"hspace": 0.0})

    # Nudge each y-label away from the shared (zero-gap) boundary so the two
    # vertical labels don't collide where the panels meet.
    for ax, ykey, ylabel, ylim, lab_y in (
            (ax_ms, "makespan", "Runtime (minutes)", ms_ylim, 0.6),
            (ax_ut, "util", "Node utilization (%)", util_ylim, 0.4)):
        for key in ORDER:
            s = series(rows, key)
            if not s:
                continue
            x = [r["stdev"] for r in s]
            y = [(r[ykey] / 60.0 if ykey == "makespan" else r[ykey]) for r in s]
            ax.plot(x, y, **SERIES[key])
        ax.set_ylabel(ylabel, y=lab_y)
        if ylim:
            ax.set_ylim(*ylim)
        style_ax(ax)

    ax_ut.set_xlabel(r"Task-duration $\sigma$ (minutes)")
    ax_ms.set_title(title)
    grouped_legend(ax_ms, loc="upper left")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="../data/variance_fine_results.csv")
    p.add_argument("--out-makespan", default="plot_variance_fine_makespan.png")
    p.add_argument("--out-util", default="plot_variance_fine_util.png")
    p.add_argument("--ymin-makespan", type=float, default=60.0)
    p.add_argument("--ymax-makespan", type=float, default=180.0)
    p.add_argument("--stacked", action="store_true",
                   help="single compact column (runtime over node util)")
    p.add_argument("--out-stacked", default="plot_variance_fine_stacked.png")
    args = p.parse_args()
    rows = load(args.inp)
    ms_ylim = (args.ymin_makespan, args.ymax_makespan)
    if args.stacked:
        plot_stacked(rows, args.out_stacked,
                     "256 node, 104,448 pipelines simulated",
                     ms_ylim)
        return
    plot_metric(rows, args.out_makespan, "makespan", "Runtime (minutes)",
                r"Runtime vs $\sigma$ (256 nodes, 104448 pipelines)",
                ylim=ms_ylim)
    plot_metric(rows, args.out_util, "util", "Node utilization (%)",
                r"Node util. vs $\sigma$ (256 nodes, 104448 pipelines)",
                ylim=(0, 105))


if __name__ == "__main__":
    main()
