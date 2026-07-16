# wf_sched_sim: Workflow Scheduling Simulator

`wf_sched_sim` is a high-performance, event-driven Python simulator designed to evaluate and compare scheduling policies for complex stochastic workflows on HPC systems. It supports arbitrary stages, walltime limits, and dynamic task runtime estimation based on normalized preceding step durations.

---

## Installation

This package is optimized for deployment with `uv`. To install it in editable mode with all dependencies:

```bash
uv pip install -e .
```

---

## Package Structure

```
.
├── examples/            # Configuration templates
│   ├── 3_stages.toml    # Standard A->B->C config running to completion
│   └── 5_stages.toml    # 5-stage ABCDE config with a fixed walltime limit
├── pyproject.toml       # Project metadata and dependencies
└── wf_sched_sim/        # Main simulator library package
    ├── __init__.py      # Exposed simulator interface
    ├── config.py        # TOML config parser
    ├── plotting.py      # Dashboard and timeline plotting utilities
    ├── simulator.py     # Core event-driven simulation loop
    └── examples/        # Entry point executables
        ├── __init__.py
        ├── run_dashboard.py # CLI executable for dashboards
        └── run_timeline.py  # CLI executable for timelines
```

---

## Supported Prioritization Schemes (Policies)

You can select which policies to run using the `-p` or `--policies` command-line argument. The supported options are:

1. **`relaxed_bulk_lrpf`** (Stage-by-Stage + LRPF):
   Prioritizes running all Step A tasks first to resolve stochastic uncertainty early. Once the A queue is empty, B, C, D, ... tasks are pooled and scheduled using **Longest Remaining Path First (LRPF)** based on precursor runtimes.
2. **`relaxed_bulk_fifo`** (Stage-by-Stage + FIFO):
   Stage-prioritized wave execution, but without LRPF optimization (tasks in subsequent stages run in FIFO order).
3. **`global_lrpf`** (Dynamic Global LRPF):
   Pools all ready tasks of any type and dynamically schedules the one with the longest remaining expected runtime (no stage barriers).
4. **`pipeline_lrpf`** (Depth-First + LRPF):
   Prioritizes downstream completion (e.g. C > B > A), with B/C queues sorted by LRPF.
5. **`pipeline_srpf`** (Depth-First + SRPF):
   Prioritizes downstream completion, but sorts B/C queues using Shortest Remaining Path First.
6. **`fifo`** (FIFO Greedy):
   Schedules tasks in simple FIFO order of readiness.
7. **`random_greedy`** (Random Greedy):
   Randomly selects any ready task from the queue.
8. **`seeded_lrpf`** (Seeded LRPF):
   Seeds Step A tasks with their expected runtime, and B/C/D/E with 0 remaining runtime at the start. Once A completes, B/C/D/E are updated with their actual estimated remaining runtimes. Because B and C are almost always estimated to take longer than A, this dynamically prioritizes completing active copy pipelines (Depth-First) while sorting B/C by LRPF.

---

## Command Line Usage

Once installed, two main scripts are exposed as CLI commands:

### 1. Generate Comparative Dashboard Plot (`wf-dashboard`)

Generates a two-panel bar chart comparing the **Makespan** and **Average Occupancy** of the specified scheduling policies.

```bash
uv run wf-dashboard -c examples/3_stages.toml -p relaxed_bulk_lrpf relaxed_bulk_fifo random_greedy -o dashboard.png
```

* **Arguments**:
  * `-c, --config`: Path to the TOML configuration file (required).
  * `-p, --policies`: Space-separated list of policies to evaluate.
  * `-o, --output`: Path to save the resulting image (default: `dashboard.png`).

### 2. Generate Stacked Area Timeline Plot (`wf-timeline`)

Generates a multi-panel stacked area chart breaking down worker states (by stage and task iteration count) over time.

```bash
uv run wf-timeline -c examples/3_stages.toml -p relaxed_bulk_lrpf relaxed_bulk_fifo random_greedy -o timeline.png
```

* **Arguments**:
  * `-c, --config`: Path to the TOML configuration file (required).
  * `-p, --policies`: Space-separated list of policies to plot.
  * `-o, --output`: Path to save the resulting image (default: `large_scale_fractions.png`).

---

## Configuration Guide (TOML)

You can define your own pipeline stages, worker pools, and walltimes in the config files:

```toml
[simulation]
num_copies = 100000        # Number of workflow sequences to run
num_workers = 25600        # Number of parallel workers (HPC nodes)
seed = 42
# Set to "None" or a float limit. If a limit is set, tasks running at walltime 
# are terminated and their work is lost, reducing overall occupancy.
walltime_limit = 320.0     

[workload]
lambda_val = 15.0          # Average event count (Poisson parameter)
stage_order = ['A', 'B', 'C', 'D', 'E'] # Determines the sequence flow

[stages.A]
base_mean = 2.0            # Base runtime coefficient (seconds)
per_event_mean = 0.5333    # Extra time spent per event (seconds)
per_event_std = 0.1        # Event scaling standard deviation
noise_std = 0.5            # Constant additive Gaussian noise (seconds)

[stages.B]
base_mean = 2.0
per_event_mean = 2.0
per_event_std = 0.5
noise_std = 0.5

# ... define other stages identically
```
