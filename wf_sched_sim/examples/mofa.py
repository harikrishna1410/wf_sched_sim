import argparse
import math
import time
import numpy as np
from wf_sched_sim.workflow import Workflow, WorkflowTask, WorkflowModel
from wf_sched_sim.system import ComputeNode, NodeTopology, SystemModel
from wf_sched_sim.mapper import SerialGeneralMapper
from wf_sched_sim.simulator import Simulator
from wf_sched_sim.orderer import (
    FIFOOrderer,
    PipelineOrderer,
    ShortestFirstOrderer,
    LongestFirstOrderer,
)

MOFA_STAGES = [
    {"name": "generate_linkers", "mean_duration": 0.37, "fanout": 1.0},
    {"name": "process_linkers", "mean_duration": 0.12, "fanout": 0.228},
    {"name": "assemble_mofs", "mean_duration": 3.0, "fanout": 0.125},
    {"name": "validate_structure", "mean_duration": 224.0, "fanout": 0.086},
    {"name": "optimize_cells", "mean_duration": 1512.7, "fanout": 0.0035},
    {"name": "estimate_adsorption", "mean_duration": 2100.0, "fanout": 1.0},
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
                    nslots={"worker": 1},
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
    print(f"nworkflows: {len(workflows)}")
    return WorkflowModel(workflows)


def build_system_model(num_workers):
    compute_node = ComputeNode(
        compute_slots={"worker": 1.0},
        compute_slot_counts={"worker": num_workers},
    )
    topology = NodeTopology()
    topology.add_node(0)
    return SystemModel(compute_node, topology)


def run_with_orderer(
    label, orderer_factory, stages, target_terminal, num_workers, cv, seed
):
    t0 = time.time()
    wf_model = build_mofa_workflows(
        stages, target_terminal=target_terminal, cv=cv, seed=seed
    )
    system = build_system_model(num_workers)
    mapper = SerialGeneralMapper(name="serial_general")
    orderer = orderer_factory(wf_model)
    sim = Simulator(wf_model, system, mapper, orderer)
    result = sim.run()
    elapsed = time.time() - t0

    total_tasks = sum(len(wf.tasks) for wf in wf_model.workflows.values())
    print(f"  [{label}]")
    print(f"    Makespan:        {result['makespan']:.4f}s")
    print(f"    Completed tasks: {result['completed_tasks']} / {total_tasks}")
    print(f"    Sim wall time:   {elapsed:.2f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="MOFA workflow scheduling simulation")
    parser.add_argument(
        "--target-terminal",
        type=int,
        default=1,
        help="Number of copies reaching the terminal stage",
    )
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument(
        "--cv",
        type=float,
        default=0.2,
        help="Coefficient of variation for task durations",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stage_counts = compute_stage_counts(MOFA_STAGES, args.target_terminal)
    total_wfs = stage_counts[0]
    print(
        f"MOFA simulation: {total_wfs} workflows, {args.num_workers} workers, "
        f"CV={args.cv}, target_terminal={args.target_terminal}"
    )

    orderers = {
        "FIFO": lambda wf_model: FIFOOrderer(),
        # "Shortest First":    lambda wf_model: ShortestFirstOrderer(),
        # "Longest First":     lambda wf_model: LongestFirstOrderer(),
    }

    results = {}
    for label, factory in orderers.items():
        results[label] = run_with_orderer(
            label,
            factory,
            MOFA_STAGES,
            args.target_terminal,
            args.num_workers,
            args.cv,
            args.seed,
        )

    print("\n--- Summary ---")
    baseline = results["FIFO"]["makespan"]
    for label, res in results.items():
        speedup = baseline / res["makespan"]
        print(f"  {label:25s}  makespan={res['makespan']:.4f}s  speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()
