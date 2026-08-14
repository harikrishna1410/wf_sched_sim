"""Fine std-dev sweep at 256 nodes / 104448 pipelines.

Four series, sweeping standard deviation directly (variance = stdev**2, passed
to the generator which reads pair[1] as variance):

  A_ord  ordered, both stages mean=10, both variances = stdev**2   stdev 1..10
  B_ord  ordered, stage1 fixed (10, var=1), stage2 (10, stdev**2)  stdev 1..10
  D_ord  ordered, stage2 fixed (10, var=1), stage1 (10, stdev**2)  stdev 1..10
  pin    pinned,  mean=20, variance = stdev**2                     stdev 2,4..20

Per-run telemetry: telemetry_vfine_{series}_s{stdev}.csv
Summary CSV (incremental append): variance_fine_results.csv
"""
import argparse
import csv
import os
import time

from workflow_gen import (
    build_pinned_pipeline_workflow,
    build_multi_stage_ordered_workflow,
    build_system,
    MulitStageOrdered,
)
from wf_sched_sim.mapper import SerialGeneralMapper
from wf_sched_sim.orderer import FIFOOrderer
from wf_sched_sim.simulator import Simulator

WORKERS_PER_NODE = 102
NODES = 256
NWORKERS = NODES * WORKERS_PER_NODE  # 26112
PIPELINES = 104448

ORD_STDEVS = list(range(1, 11))          # 1,2,...,10
PIN_STDEVS = list(range(2, 21, 2))       # 2,4,...,20

# Extended sweep: ordered out to 20, pinned out to 40 (same increments).
ORD_STDEVS_EXT = list(range(1, 21))      # 1,2,...,20
PIN_STDEVS_EXT = list(range(2, 41, 2))   # 2,4,...,40


def make_configs(extended=False):
    """Build (series, policy, stdev, task_distr) tuples for all four series."""
    ord_stdevs = ORD_STDEVS_EXT if extended else ORD_STDEVS
    pin_stdevs = PIN_STDEVS_EXT if extended else PIN_STDEVS
    cfgs = []
    for s in ord_stdevs:
        v = s * s
        cfgs.append(("A_ord", "ordered", s, [[10, v], [10, v]]))
        cfgs.append(("B_ord", "ordered", s, [[10, 1], [10, v]]))
        cfgs.append(("D_ord", "ordered", s, [[10, v], [10, 1]]))
    for s in pin_stdevs:
        cfgs.append(("pin", "pinned", s, [[20, s * s]]))
    return cfgs


RESULT_FIELDS = [
    "series", "policy", "stdev", "variance", "nodes", "nworkers",
    "pipeline_num", "distribution", "stage1_mean", "stage1_var",
    "stage2_mean", "stage2_var", "makespan", "completed_tasks",
    "workflow_done", "unscheduled", "wall_seconds",
]


def run_one(policy, task_distr, pipeline_num, nworkers, tag, freq=100):
    if policy == "pinned":
        wf = build_pinned_pipeline_workflow(task_distr=task_distr,
                                            pipeline_num=pipeline_num)
        orderer = FIFOOrderer()
    elif policy == "ordered":
        wf = build_multi_stage_ordered_workflow(task_distr=task_distr,
                                                pipeline_num=pipeline_num)
        orderer = MulitStageOrdered()
    else:
        raise ValueError(f"unknown policy: {policy}")

    system = build_system(nworkers)
    mapper = SerialGeneralMapper(name="greedy_mapper")
    sim = Simulator(workflow_model=wf, system=system, mapper=mapper,
                    orderer=orderer)

    fname = f"{os.getcwd()}/telemetry_{tag}.csv"
    t0 = time.perf_counter()
    result = sim.run(telemetry=True, fname=fname, freq=freq)
    wall = time.perf_counter() - t0
    return result, wall


def append_result(out_path, row):
    new_file = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in RESULT_FIELDS})


def main():
    parser = argparse.ArgumentParser(
        description="Fine std-dev sweep at 256 nodes / 104448 pipelines.")
    parser.add_argument("--series", type=str, default="A_ord,B_ord,D_ord,pin",
                        help="comma-separated subset of the four series")
    parser.add_argument("--freq", type=int, default=100)
    parser.add_argument("--extended", action="store_true",
                        help="extend ordered stdev to 20 and pinned to 40")
    parser.add_argument("--out", type=str, default=None,
                        help="summary CSV path (default depends on --extended)")
    args = parser.parse_args()

    default_out = ("variance_fine_extended_results.csv" if args.extended
                   else "variance_fine_results.csv")
    out_name = args.out or default_out
    tag_prefix = "vfine_extended" if args.extended else "vfine"

    want = {s.strip() for s in args.series.split(",") if s.strip()}
    out_path = os.path.join(os.getcwd(), out_name)
    selected = [c for c in make_configs(extended=args.extended) if c[0] in want]
    print(f"Running {len(selected)} configs -> {out_path}", flush=True)

    for series, policy, stdev, task_distr in selected:
        tag = f"{tag_prefix}_{series}_s{stdev}"
        result, wall = run_one(policy, task_distr, PIPELINES, NWORKERS, tag,
                               freq=args.freq)

        s1_mean, s1_var = task_distr[0]
        if len(task_distr) > 1:
            s2_mean, s2_var = task_distr[1]
            dist_label = f"m{s1_mean}v{s1_var}_m{s2_mean}v{s2_var}"
        else:
            s2_mean = s2_var = ""
            dist_label = f"m{s1_mean}v{s1_var}"

        row = {
            "series": series,
            "policy": policy,
            "stdev": stdev,
            "variance": stdev * stdev,
            "nodes": NODES,
            "nworkers": NWORKERS,
            "pipeline_num": PIPELINES,
            "distribution": dist_label,
            "stage1_mean": s1_mean,
            "stage1_var": s1_var,
            "stage2_mean": s2_mean,
            "stage2_var": s2_var,
            "makespan": result.get("makespan"),
            "completed_tasks": result.get("completed_tasks"),
            "workflow_done": result.get("workflow_done"),
            "unscheduled": result.get("unscheduled"),
            "wall_seconds": round(wall, 3),
        }
        append_result(out_path, row)
        print(
            f"[{series:<6}] {policy:<7} stdev={stdev:<3} {dist_label:<16} "
            f"makespan={row['makespan']:.1f}s wall={wall:.1f}s "
            f"done={row['workflow_done']} unsched={row['unscheduled']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
