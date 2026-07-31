import argparse
import csv
import math
import os
import time
import numpy as np
from wf_sched_sim.workflow import Workflow, WorkflowTask, WorkflowModel
from wf_sched_sim.system import ComputeNode, NodeTopology, SystemModel
from wf_sched_sim.mapper import HeterogeneousMapper, PartitionedMapper
from wf_sched_sim.simulator import Simulator
from wf_sched_sim.orderer import (
    FIFOOrderer,
    ShortestFirstOrderer,
    LongestFirstOrderer,
)

MOFA_STAGES = [
    {
        "name": "generate_linkers",
        "mean_duration": 0.37,
        "fanout": 1.0,
        "nslots": {"cpu": 1, "gpu": 1},
    },
    {
        "name": "process_linkers",
        "mean_duration": 0.12,
        "fanout": 0.228,
        "nslots": {"cpu": 1},
    },
    {
        "name": "assemble_mofs",
        "mean_duration": 3.0,
        "fanout": 0.125,
        "nslots": {"cpu": 1},
    },
    {
        "name": "validate_structure",
        "mean_duration": 224.0,
        "fanout": 0.086,
        "nslots": {"cpu": 1, "gpu": 1},
    },
    {
        "name": "optimize_cells",
        "mean_duration": 1512.7,
        "fanout": 0.0035,
        "nslots": {"cpu": 12, "gpu": 12},
        "nnodes": 2,
    },
    {
        "name": "estimate_adsorption",
        "mean_duration": 2100.0,
        "fanout": 1.0,
        "nslots": {"cpu": 1},
    },
    {
        "name": "retrain",
        "mean_duration": 96.5,
        "fanout": 1.0,
        "nslots": {"cpu": 12, "gpu": 12},
        "nnodes": 1,
    },
]


def compute_stage_counts(stages, target_terminal=1):
    n = len(stages)
    counts = [0] * n
    counts[n - 1] = target_terminal
    for i in range(n - 2, -1, -1):
        counts[i] = math.ceil(counts[i + 1] / stages[i]["fanout"])
    return counts


def build_mofa_workflows(stages, target_terminal=1, cv=0.2, seed=42):
    rng = np.random.default_rng(seed)
    stage_counts = compute_stage_counts(stages, 1)
    stage_names = [s["name"] for s in stages]
    n = len(stages)

    print(f"  Stage counts: {dict(zip(stage_names, stage_counts))}")
    print(f"  Total tasks:  {sum(stage_counts)}")

    workflows = []
    for wf_id in range(target_terminal):
        wf = Workflow(name=f"wf_{wf_id}")

        stage_tasks = []
        for s in range(n):
            stage = stages[s]
            count = stage_counts[s]
            tasks_this_stage = []
            for t in range(count):
                std = stage["mean_duration"] * cv
                dur = max(rng.normal(stage["mean_duration"], std), 1e-4)
                task_name = f"{stage_names[s]}_{t}"
                task = WorkflowTask(
                    name=task_name,
                    compute_cost=dur,
                    comm_size=0.0,
                    nslots=dict(stage["nslots"]),
                    nnodes=stage.get("nnodes", 1),
                )
                wf.add_task(task)
                tasks_this_stage.append(task_name)
            stage_tasks.append(tasks_this_stage)

        for s in range(n - 1):
            parent_names = stage_tasks[s]
            child_names = stage_tasks[s + 1]
            for c_idx, child_name in enumerate(child_names):
                parent_idx = c_idx % len(parent_names)
                wf.add_edge((parent_names[parent_idx], child_name))

        workflows.append(wf)

    return WorkflowModel(workflows)


GPU_STAGES = ["generate_linkers", "validate_structure"]
MULTINODE_STAGES = ["optimize_cells", "retrain"]
CPU_STAGES = ["process_linkers", "assemble_mofs", "estimate_adsorption"]


def partition_shared(nn):
    all_nodes = list(range(nn))
    return {s["name"]: all_nodes for s in MOFA_STAGES}


def partition_balanced(nn):
    gpu_end = max(1, int(nn * 0.25))
    multi_end = max(gpu_end + 1, nn)
    stage_nodes = {}
    for s in GPU_STAGES:
        stage_nodes[s] = list(range(0, gpu_end))
    for s in MULTINODE_STAGES:
        stage_nodes[s] = list(range(gpu_end, multi_end))
    for s in CPU_STAGES:
        stage_nodes[s] = list(range(0, nn))
    return stage_nodes


def partition_gpu_heavy(nn):
    gpu_end = max(1, int(nn * 0.5))
    multi_end = max(gpu_end + 1, nn)
    stage_nodes = {}
    for s in GPU_STAGES:
        stage_nodes[s] = list(range(0, gpu_end))
    for s in MULTINODE_STAGES:
        stage_nodes[s] = list(range(gpu_end, multi_end))
    for s in CPU_STAGES:
        stage_nodes[s] = list(range(0, nn))
    return stage_nodes


def build_system_model(num_nodes):
    compute_node = ComputeNode(
        compute_slots={"cpu": 1.0, "gpu": 1.0},
        compute_slot_counts={"cpu": 102, "gpu": 12},
    )
    topology = NodeTopology()
    for i in range(num_nodes):
        topology.add_node(i)
    return SystemModel(compute_node, topology)


def run_with_orderer(
    label, orderer_factory, mapper_factory, stages, target_terminal, num_nodes, cv, seed
):
    t0 = time.time()
    wf_model = build_mofa_workflows(
        stages, target_terminal=target_terminal, cv=cv, seed=seed
    )
    system = build_system_model(num_nodes)
    print("Done building system")
    mapper = mapper_factory()
    orderer = orderer_factory(wf_model)
    sim = Simulator(wf_model, system, mapper, orderer)
    result = sim.run(debug=False)
    elapsed = time.time() - t0

    total_tasks = sum(len(wf.tasks) for wf in wf_model.workflows.values())
    print(f"  [{label}]")
    print(f"    Makespan:        {result['makespan']:.4f}s")
    print(f"    Completed tasks: {result['completed_tasks']} / {total_tasks}")
    print(f"    Sim wall time:   {elapsed:.2f}s")
    result["wall_time"] = elapsed
    result["mapper"] = mapper._name
    result["strict"] = mapper._strict
    return result


def main():
    parser = argparse.ArgumentParser(
        description="MOFA workflow scheduling simulation with heterogeneous resources"
    )
    parser.add_argument(
        "--target-terminal",
        type=str,
        default="1",
        help="Comma-separated list, e.g. 1,5,10",
    )
    parser.add_argument(
        "--num-nodes",
        type=str,
        default="32",
        help="Comma-separated list, e.g. 32,256,8192",
    )
    parser.add_argument("--cv", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="mofa_heterogeneous.csv")
    args = parser.parse_args()

    num_nodes_list = [int(x) for x in args.num_nodes.split(",")]
    target_terminal_list = [int(x) for x in args.target_terminal.split(",")]

    orderers = {
        "FIFO": lambda wf_model: FIFOOrderer(),
        "Shortest First": lambda wf_model: ShortestFirstOrderer(),
        "Longest First": lambda wf_model: LongestFirstOrderer(),
    }

    header = [
        "orderer",
        "mapper",
        "strict",
        "num_nodes",
        "target_terminal",
        "cv",
        "seed",
        "makespan",
        "completed_tasks",
        "total_tasks",
        "wall_time",
    ]
    existing = []
    if os.path.exists(args.output):
        with open(args.output, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            existing = list(reader)

    all_new_rows = {}

    for tt in target_terminal_list:
        for nn in num_nodes_list:
            mapper_factories = {
                # "heterogeneous": lambda: HeterogeneousMapper(
                #     name="heterogeneous", strict=False
                # ),
                "partitioned-shared": lambda: PartitionedMapper(
                    name="partitioned-shared",
                    stage_nodes=partition_shared(nn),
                ),
                "partitioned-balanced": lambda: PartitionedMapper(
                    name="partitioned-balanced",
                    stage_nodes=partition_balanced(nn),
                ),
                "partitioned-gpu-heavy": lambda: PartitionedMapper(
                    name="partitioned-gpu-heavy",
                    stage_nodes=partition_gpu_heavy(nn),
                ),
            }

            for mapper_label, mapper_factory in mapper_factories.items():
                print(
                    f"\n=== target_terminal={tt}, num_nodes={nn}, mapper={mapper_label}, CV={args.cv} ==="
                )
                results = {}
                for label, factory in orderers.items():
                    results[label] = run_with_orderer(
                        label,
                        factory,
                        mapper_factory,
                        MOFA_STAGES,
                        tt,
                        nn,
                        args.cv,
                        args.seed,
                    )

                print("\n--- Summary ---")
                baseline = results["FIFO"]["makespan"]
                for label, res in results.items():
                    speedup = baseline / res["makespan"]
                    print(
                        f"  {label:25s}  makespan={res['makespan']:.4f}s  speedup={speedup:.2f}x"
                    )

                total_tasks = sum(compute_stage_counts(MOFA_STAGES, 1)) * tt
                for label, res in results.items():
                    key = (
                        label,
                        res["mapper"],
                        str(res["strict"]),
                        str(nn),
                        str(tt),
                        str(args.cv),
                        str(args.seed),
                    )
                    all_new_rows[key] = [
                        label,
                        res["mapper"],
                        res["strict"],
                        nn,
                        tt,
                        args.cv,
                        args.seed,
                        f"{res['makespan']:.6f}",
                        res["completed_tasks"],
                        total_tasks,
                        f"{res['wall_time']:.2f}",
                    ]

    updated = []
    for row in existing:
        key = tuple(row[:7])
        if key in all_new_rows:
            updated.append(all_new_rows.pop(key))
        else:
            updated.append(row)
    for row in all_new_rows.values():
        updated.append(row)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(updated)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
