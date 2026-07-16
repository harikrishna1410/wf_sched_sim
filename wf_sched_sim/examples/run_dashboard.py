import argparse
import sys
from wf_sched_sim.config import parse_config
from wf_sched_sim.simulator import LargeScaleSimulator
from wf_sched_sim.plotting import plot_dashboard


def main():
    parser = argparse.ArgumentParser(
        description="Run scheduling simulations and generate a dashboard-style comparison plot."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to TOML configuration file.",
    )
    parser.add_argument(
        "-p",
        "--policies",
        nargs="+",
        default=["relaxed_bulk_lrpf", "relaxed_bulk_fifo", "random_greedy", "fifo"],
        help="List of policies to compare. Options: relaxed_bulk_lrpf, relaxed_bulk_fifo, pipeline_lrpf, pipeline_srpf, global_lrpf, random_greedy, fifo",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="dashboard.png",
        help="Path to save the output plot.",
    )

    args = parser.parse_args()

    print(f"Loading configuration from {args.config}...")
    try:
        config = parse_config(args.config)
    except Exception as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)

    print("\nSimulation Settings:")
    print(f"  Copies: {config['num_copies']}")
    print(f"  Workers: {config['num_workers']}")
    print(f"  Walltime Limit: {config['walltime_limit']}s")
    print(f"  Stages: {config['stage_order']}")

    sim = LargeScaleSimulator(
        num_copies=config["num_copies"],
        num_workers=config["num_workers"],
        seed=config["seed"],
        walltime_limit=config["walltime_limit"],
        lambda_val=config["lambda_val"],
        stages=config["stages"],
    )

    # Calculate minimum makespan
    # Sum of expected durations of all stages for all copies
    total_expected_work = 0.0
    for stage in config["stages"]:
        expected_dur = (
            stage["base_mean"] + config["lambda_val"] * stage["per_event_mean"]
        )
        total_expected_work += expected_dur * config["num_copies"]

    min_makespan = total_expected_work / config["num_workers"]
    print(f"Theoretical minimum makespan: {min_makespan:.2f}s")

    results = {}
    for policy in args.policies:
        try:
            res = sim.run(policy, track_history=False)
            results[policy] = res
        except Exception as e:
            print(f"Error running policy {policy}: {e}")

    if not results:
        print("No simulation results generated. Exiting.")
        sys.exit(1)

    print("\nSimulation Summary:")
    print(
        f"{'Policy':<25} | {'Makespan (s)':<12} | {'Occupancy (%)':<15} | {'Completed Copies':<18}"
    )
    print("-" * 75)
    for p, res in results.items():
        print(
            f"{p:<25} | {res['makespan']:<12.2f} | {res['occupancy']:<15.2f} | {res['completed_copies']:<18}"
        )

    print(f"\nGenerating dashboard plot and saving to {args.output}...")
    plot_dashboard(
        results=results,
        stage_order=config["stage_order"],
        min_makespan=min_makespan,
        walltime_limit=config["walltime_limit"],
        save_path=args.output,
    )
    print("Done!")


if __name__ == "__main__":
    main()
