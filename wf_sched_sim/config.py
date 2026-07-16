import toml
from typing import Dict, Any


def parse_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        data = toml.load(f)

    sim_section = data.get("simulation", {})
    workload_section = data.get("workload", {})
    stages_section = data.get("stages", {})

    # Defaults
    num_copies = sim_section.get("num_copies", 100000)
    num_workers = sim_section.get("num_workers", 25600)
    seed = sim_section.get("seed", 42)
    walltime_limit = sim_section.get("walltime_limit", None)
    if walltime_limit == "None" or walltime_limit == "none" or walltime_limit == "null":
        walltime_limit = None

    lambda_val = workload_section.get("lambda_val", 15.0)
    stage_order = workload_section.get("stage_order", [])

    # If stage_order is not defined, try to infer from keys
    if not stage_order:
        stage_order = list(stages_section.keys())

    # Compile stages data
    stages = []
    for stage_name in stage_order:
        st_data = stages_section.get(stage_name, {})
        stages.append(
            {
                "name": stage_name,
                "base_mean": float(st_data.get("base_mean", 0.0)),
                "per_event_mean": float(st_data.get("per_event_mean", 0.0)),
                "per_event_std": float(st_data.get("per_event_std", 0.0)),
                "noise_std": float(st_data.get("noise_std", 0.0)),
            }
        )

    return {
        "num_copies": num_copies,
        "num_workers": num_workers,
        "seed": seed,
        "walltime_limit": walltime_limit,
        "lambda_val": lambda_val,
        "stages": stages,
        "stage_order": stage_order,
    }
