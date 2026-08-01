"""Scaling-test runner for the sbnd workflow simulator.

Builds on workflow_gen.py (generator-driven builders) to sweep node counts under
three regimes:

  Test 1  fixed 4x oversubscription: pipelines = 4 * workers, sweep node counts.
  Test 2  fixed total work:          pipelines = 104448,      sweep node counts.
  Test 3  homogeneous task types:    256-node / 104448-pipeline ordered run with
                                     both stages sharing one distribution.

node -> workers mapping: workers = nodes * WORKERS_PER_NODE (sbnd baseline is
256 nodes = 26112 workers). See scaling_runner.py --help.

Results are appended incrementally to a summary CSV so a partial sweep survives
an interrupt. Per-run telemetry is written to distinct filenames.
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
FIXED_PIPELINES = 104448

# Default node counts for the sweeps.
TEST1_NODES = [256, 512, 1024, 2048]  # 4x ratio -> up to 835,584 pipelines
TEST2_NODES = [256, 512, 1024, 2048, 4096, 8192]  # fixed 104448 pipelines

# Distributions (mean, variance) per stage, minutes -- from workflow_gen.py.
PINNED_DISTR = [[36.89, 11.72 ** 2]]
ORDERED_DISTR = [[11.61, 1.91 ** 2], [26.39, 10.38 ** 2]]
# Test 3: both stages set to the mean of the two ordered stages.
#   mean mu  = (11.61 + 26.39) / 2 = 19.0
#   mean sig = (1.91 + 10.38) / 2  = 6.145
ORDERED_EQ_DISTR = [[19.0, 6.145 ** 2], [19.0, 6.145 ** 2]]

RESULT_FIELDS = [
    "test", "policy", "nodes", "nworkers", "pipeline_num", "ratio",
    "distribution", "makespan", "completed_tasks", "workflow_done",
    "unscheduled", "wall_seconds",
]


def run_one(policy, task_distr, pipeline_num, nworkers, tag, freq=100):
    """Run a single simulation and return a summary row dict."""
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

    return {
        "policy": policy,
        "pipeline_num": pipeline_num,
        "nworkers": nworkers,
        "makespan": result.get("makespan"),
        "completed_tasks": result.get("completed_tasks"),
        "workflow_done": result.get("workflow_done"),
        "unscheduled": result.get("unscheduled"),
        "wall_seconds": round(wall, 3),
    }


def append_result(out_path, row):
    """Append one result row to the summary CSV, writing the header if new."""
    new_file = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in RESULT_FIELDS})


def log_run(row):
    print(
        f"[test {row['test']}] {row['policy']:<7} "
        f"nodes={row['nodes']:<5} workers={row['nworkers']:<7} "
        f"pipes={row['pipeline_num']:<8} "
        f"makespan={row['makespan']:.1f}s "
        f"wall={row['wall_seconds']:.1f}s "
        f"done={row['workflow_done']} unsched={row['unscheduled']}",
        flush=True,
    )


def sweep(test, nodes_list, pipeline_fn, policies, distrs, out_path, freq):
    """Run one test across node counts for each policy.

    pipeline_fn(nworkers) -> pipeline_num for this test.
    distrs maps policy -> (task_distr, distribution_label).
    """
    for nodes in nodes_list:
        nworkers = nodes * WORKERS_PER_NODE
        pipeline_num = pipeline_fn(nworkers)
        for policy in policies:
            task_distr, dist_label = distrs[policy]
            tag = f"{policy}_test{test}_n{nodes}"
            row = run_one(policy, task_distr, pipeline_num, nworkers, tag,
                          freq=freq)
            row.update({
                "test": test,
                "nodes": nodes,
                "ratio": round(pipeline_num / nworkers, 3),
                "distribution": dist_label,
            })
            log_run(row)
            append_result(out_path, row)


def main():
    parser = argparse.ArgumentParser(
        description="Node-count scaling sweeps for the sbnd simulator.")
    parser.add_argument("--tests", type=str, default="1,2,3",
                        help="comma-separated subset of {1,2,3} to run")
    parser.add_argument("--max-nodes", type=int, default=2048,
                        help="cap node count for Test 1 (4x ratio) sweep")
    parser.add_argument("--freq", type=int, default=100,
                        help="telemetry sampling frequency (completions)")
    parser.add_argument("--out", type=str, default="scaling_results.csv",
                        help="summary CSV path (rows appended)")
    args = parser.parse_args()

    tests = {t.strip() for t in args.tests.split(",") if t.strip()}
    out_path = os.path.join(os.getcwd(), args.out)

    both = ["pinned", "ordered"]

    # Test 1: fixed 4x ratio, capped at --max-nodes.
    if "1" in tests:
        nodes_list = [n for n in TEST1_NODES if n <= args.max_nodes]
        distrs = {
            "pinned": (PINNED_DISTR, "pinned_default"),
            "ordered": (ORDERED_DISTR, "ordered_default"),
        }
        sweep(1, nodes_list, lambda w: 4 * w, both, distrs, out_path, args.freq)

    # Test 2: fixed total pipelines across the full node range.
    if "2" in tests:
        distrs = {
            "pinned": (PINNED_DISTR, "pinned_default"),
            "ordered": (ORDERED_DISTR, "ordered_default"),
        }
        sweep(2, TEST2_NODES, lambda w: FIXED_PIPELINES, both, distrs,
              out_path, args.freq)

    # Test 3: 256 nodes, fixed pipelines, ordered only, homogeneous stages.
    # (pinned is single-stage, so "same task types" is meaningless there.)
    if "3" in tests:
        distrs = {"ordered": (ORDERED_EQ_DISTR, "ordered_equal_stages")}
        sweep(3, [256], lambda w: FIXED_PIPELINES, ["ordered"], distrs,
              out_path, args.freq)


if __name__ == "__main__":
    main()
