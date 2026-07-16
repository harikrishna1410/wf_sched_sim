import argparse
import sys
from wf_sched_sim.config import parse_config
from wf_sched_sim.simulator import LargeScaleSimulator
from wf_sched_sim.plotting import plot_timeline


def main():
    parser = argparse.ArgumentParser(
        description="Run scheduling simulations and generate a detailed stacked area timeline plot shaded by worker task iteration."
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
        default=[
            "relaxed_bulk_lrpf",
            "relaxed_bulk_fifo",
            "seeded_lrpf",
            "random_greedy",
        ],
        help="List of policies to compare. Options: relaxed_bulk_lrpf, relaxed_bulk_fifo, pipeline_lrpf, pipeline_srpf, global_lrpf, random_greedy, fifo",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="large_scale_fractions.png",
        help="Path to save the output timeline plot.",
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

    results = {}
    for policy in args.policies:
        try:
            res = sim.run(policy, track_history=True)
            results[policy] = res
        except Exception as e:
            print(f"Error running policy {policy}: {e}")

    if not results:
        print("No simulation results generated. Exiting.")
        sys.exit(1)

    print(f"\nGenerating timeline plot and saving to {args.output}...")
    plot_timeline(
        results=results,
        num_workers=config["num_workers"],
        stage_names=config["stage_order"],
        save_path=args.output,
    )
    print("Done!")


if __name__ == "__main__":
    main()
