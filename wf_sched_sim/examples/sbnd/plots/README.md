# sc26_xloop_sbnd — Artifact for the sbnd scheduling figures

Code and data to reproduce two figures from the paper, both produced with the
`wf_sched_sim` HPC workflow-scheduling simulator applied to the sbnd
reconstruction pipeline:

- **Figure 6** (`plot_scaling.pdf`) — strong scaling: fixed 104,448 pipelines
  swept over node count, comparing an unordered single-stage (FIFO) policy against
  a dynamic multi-stage (LPT) policy.
- **Figure 7** (`plot_variance_fine_extended_stacked.pdf`) — sensitivity to
  task-duration variance at a fixed 256-node / 104,448-pipeline problem.


## Figure 6 — strong scaling

```bash
# Plot from the included data (fast):
python plot_scaling.py                       # writes plot_scaling.pdf
# Or regenerate the data first (a few minutes):
python ../scaling_runner.py --tests 2 --out ../data/scaling_results.csv
python plot_scaling.py
```

Panels: makespan (top) and node utilization (bottom) vs node count (log2), for
both policies. Only Test 2 (fixed 104,448 pipelines) is drawn.

## Figure 7 — variance sensitivity

```bash
cd fig7
# Plot from the included data (fast):
python plot_variance_fine.py --stacked \
    --in ../data/variance_fine_extended_results.csv \
    --out-stacked plot_variance_fine_extended_stacked.pdf \
    --ymin-makespan 60 --ymax-makespan 320
# Or regenerate the data first (80 configs, tens of minutes):
python ../variance_fine_runner.py --extended --out ../data/variance_fine_extended_results.csv
```

Panels: makespan (top) and node utilization (bottom) vs task-duration sigma, for
four series (three LPT variants + one FIFO). Restrict with `--series` (e.g.
`--series pin`) to run one series at a time.

## Reproducibility note

Task durations are drawn from Gaussian distributions with an unseeded RNG, so a
fresh run reproduces the trends and values to within sampling noise rather than
bit-for-bit. The shipped `*_results.csv` files are the exact data used for the
published figures.

## Environment used for the published figures

Python 3.12, numpy 2.5.1, matplotlib 3.11.0, networkx 3.6.1, toml 0.10.2.
