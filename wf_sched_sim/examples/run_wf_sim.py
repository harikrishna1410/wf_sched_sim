import argparse
import sys
import time

import numpy as np

from wf_sched_sim.config import parse_config
from wf_sched_sim.mapper import SerialGeneralMapper
from wf_sched_sim.orderer import FIFOOrderer
from wf_sched_sim.simulator import LargeScaleSimulator
from wf_sched_sim.system import ComputeNode, NodeTopology, SystemModel
from wf_sched_sim.wf_simulator import Simulator
from wf_sched_sim.workflow import Workflow, WorkflowModel, WorkflowTask


def generate_workloads(num_copies, lambda_val, stages, seed=42):
    np.random.seed(seed)
    K_arr = np.random.poisson(lambda_val, num_copies)
    durations = {}

    for stage in stages:
        name = stage["name"]
        means = stage["base_mean"] + K_arr * stage["per_event_mean"]
        vars_arr = stage["noise_std"] ** 2 + K_arr * (stage["per_event_std"] ** 2)
        durs = np.random.normal(means, np.sqrt(vars_arr))
        durs = np.clip(durs, 0.001, None)
        durations[name] = durs

    return durations


def build_workflow_model(stages, durations, num_copies):
    stage_names = [s["name"] for s in stages]
    workflows = []

    for copy_id in range(num_copies):
        wf = Workflow(name=f"copy_{copy_id}")
        for stage_name in stage_names:
            task = WorkflowTask(
                name=stage_name,
                compute_cost=float(durations[stage_name][copy_id]),
                comm_size=0.0,
                nslots={"worker": 1},
            )
            wf.add_task(task)

        for i in range(len(stage_names) - 1):
            wf.add_edge((stage_names[i], stage_names[i + 1]))

        workflows.append(wf)

    return WorkflowModel(workflows)


def build_system_model(num_workers):
    compute_node = ComputeNode(
        compute_slots={"worker": 1.0},
        compute_slot_counts={"worker": num_workers},
    )
    topology = NodeTopology()
    topology.add_node(0)
    return SystemModel(compute_node, topology)


def main():
    parser = argparse.ArgumentParser(
        description="Run the new DAG-based simulator on an existing pipeline config and compare with LargeScaleSimulator."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to TOML configuration file.",
    )
    args = parser.parse_args()

    print(f"Loading configuration from {args.config}...")
    try:
        config = parse_config(args.config)
    except Exception as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)

    num_copies = config["num_copies"]
    num_workers = config["num_workers"]
    stages = config["stages"]
    seed = config["seed"]
    lambda_val = config["lambda_val"]

    print(
        f"\nSettings: {num_copies} copies, {num_workers} workers, {len(stages)} stages"
    )

    durations = generate_workloads(num_copies, lambda_val, stages, seed)

    # --- New simulator ---
    print("\n--- New DAG-based Simulator ---")
    t0 = time.time()
    print("  Building workflow model...")
    workflow_model = build_workflow_model(stages, durations, num_copies)
    print(f"  Built in {time.time() - t0:.2f}s")

    t1 = time.time()
    print("  Building system model...")
    system = build_system_model(num_workers)
    print(f"  Built in {time.time() - t1:.2f}s")

    mapper = SerialGeneralMapper(name="serial_general")
    orderer = FIFOOrderer()
    sim = Simulator(workflow_model, system, mapper, orderer)
    print("  Running simulation...")
    t2 = time.time()
    result = sim.run()
    print(f"  Simulation done in {time.time() - t2:.2f}s")
    new_time = time.time() - t0

    print(f"  Makespan:        {result['makespan']:.2f}s")
    print(f"  Completed tasks: {result['completed_tasks']}")
    print(f"  All done:        {result['workflow_done']}")
    print(f"  Unscheduled:     {result['unscheduled']}")
    print(f"  Sim wall time:   {new_time:.2f}s")

    # --- Old simulator ---
    print("\n--- Old LargeScaleSimulator (fifo) ---")
    old_sim = LargeScaleSimulator(
        num_copies=num_copies,
        num_workers=num_workers,
        seed=seed,
        walltime_limit=config["walltime_limit"],
        lambda_val=lambda_val,
        stages=stages,
    )
    old_result = old_sim.run("fifo", track_history=False)

    print(f"\n--- Comparison ---")
    print(f"  New makespan:  {result['makespan']:.2f}s")
    print(f"  Old makespan:  {old_result['makespan']:.2f}s")
    diff = abs(result["makespan"] - old_result["makespan"])
    print(f"  Difference:    {diff:.4f}s")


if __name__ == "__main__":
    main()
