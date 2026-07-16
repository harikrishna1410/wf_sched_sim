import time
import heapq
import numpy as np
import bisect
from typing import Dict, Any, List


class LargeScaleSimulator:
    def __init__(
        self,
        num_copies: int = 100000,
        num_workers: int = 25600,
        seed: int = 42,
        walltime_limit: float = None,
        lambda_val: float = 15.0,
        stages: List[Dict[str, Any]] = None,
    ):
        self.num_copies = num_copies
        self.num_workers = num_workers
        self.seed = seed
        self.walltime_limit = walltime_limit
        self.lambda_val = lambda_val
        self.stages = stages if stages is not None else []
        self.L = len(self.stages)

        np.random.seed(seed)
        self.workloads = self._generate_workloads()

    def _generate_workloads(self) -> Dict[str, np.ndarray]:
        K_arr = np.random.poisson(self.lambda_val, self.num_copies)
        durations = {}
        expected_durs = {}

        for idx, stage in enumerate(self.stages):
            name = stage["name"]
            base_mean = stage["base_mean"]
            per_event_mean = stage["per_event_mean"]
            per_event_std = stage["per_event_std"]
            noise_std = stage["noise_std"]

            means = base_mean + K_arr * per_event_mean
            vars_arr = noise_std**2 + K_arr * (per_event_std**2)
            durs = np.random.normal(means, np.sqrt(vars_arr))
            durs = np.clip(durs, 0.001, None)

            durations[name] = durs
            expected_durs[name] = base_mean + self.lambda_val * per_event_mean

        return {"durations": durations, "expected_durs": expected_durs, "K": K_arr}

    def run(self, policy: str, track_history: bool = False) -> Dict[str, Any]:
        """
        Runs the simulation under the specified policy.
        Returns:
            Dictionary with makespan, occupancy, completed_copies, and history (if tracked)
        """
        # Reset seed for reproducibility
        np.random.seed(self.seed)
        t0 = time.time()

        durs_dict = self.workloads["durations"]
        mean_durs = [
            self.workloads["expected_durs"][stage["name"]] for stage in self.stages
        ]
        stage_names = [stage["name"] for stage in self.stages]

        # State: states[copy_id, stage_idx]
        # 0 = PENDING, 1 = RUNNING, 2 = COMPLETED
        states = np.zeros((self.num_copies, self.L), dtype=np.int8)

        # Initialize ready queues for each stage
        ready_queues = [[] for _ in range(self.L)]

        # Setup initial queue for stage 0
        sum_mean_durs = sum(mean_durs)
        for idx in range(self.num_copies):
            if policy in ["relaxed_bulk_lrpf", "pipeline_lrpf", "global_lrpf"]:
                key = -sum_mean_durs
            elif policy == "seeded_lrpf":
                key = -mean_durs[0]
            elif policy == "pipeline_srpf":
                key = sum_mean_durs
            else:
                key = 0.0  # FIFO / Random
            ready_queues[0].append((float(key), idx))

        event_queue = []
        idle_workers = list(range(self.num_workers))
        worker_task = [None] * self.num_workers

        current_time = 0.0
        total_work_completed = 0.0
        completed_copies_count = 0

        # Task iteration tracking per worker
        worker_task_counts = [
            {st["name"]: 0 for st in self.stages} for _ in range(self.num_workers)
        ]

        # O(1) History Tracking variables
        cats = []
        for s, stage in enumerate(self.stages):
            st_name = stage["name"]
            max_cap = 4 if s < 2 else 3
            for iter_num in range(1, max_cap + 1):
                cats.append(f"{st_name}_{iter_num}")

        state_counts = {c: 0 for c in cats}
        state_counts["Idle"] = self.num_workers

        history = [(0.0, state_counts.copy())]

        def log_state():
            history.append((current_time, state_counts.copy()))

        def schedule_tasks():
            nonlocal total_work_completed
            while idle_workers:
                best_stage = -1
                best_idx_in_queue = -1

                # Check if we have ready tasks
                has_tasks = any(len(q) > 0 for q in ready_queues)
                if not has_tasks:
                    break

                # Selection logic depending on policy
                if policy.startswith("relaxed_bulk"):
                    for s in range(self.L):
                        if ready_queues[s]:
                            best_stage = s
                            best_idx_in_queue = 0
                            break

                elif policy.startswith("pipeline"):
                    for s in range(self.L - 1, -1, -1):
                        if ready_queues[s]:
                            best_stage = s
                            best_idx_in_queue = 0
                            break

                elif policy in ["global_lrpf", "seeded_lrpf"]:
                    best_key = float("inf")
                    for s in range(self.L):
                        if ready_queues[s]:
                            key = ready_queues[s][0][0]
                            if key < best_key:
                                best_key = key
                                best_stage = s
                                best_idx_in_queue = 0

                elif policy == "random_greedy":
                    sizes = [len(q) for q in ready_queues]
                    total = sum(sizes)
                    rand_val = np.random.randint(0, total)

                    cum = 0
                    for s in range(self.L):
                        cum += sizes[s]
                        if rand_val < cum:
                            best_stage = s
                            best_idx_in_queue = np.random.randint(0, sizes[s])
                            break

                elif policy == "fifo":
                    best_key = float("inf")
                    for s in range(self.L):
                        if ready_queues[s]:
                            key = ready_queues[s][0][0]
                            if key < best_key:
                                best_key = key
                                best_stage = s
                                best_idx_in_queue = 0
                else:
                    raise ValueError(f"Unknown policy: {policy}")

                if best_stage == -1:
                    break

                # Schedule the chosen task
                _, copy_id = ready_queues[best_stage].pop(best_idx_in_queue)
                states[copy_id, best_stage] = 1  # RUNNING
                dur = durs_dict[stage_names[best_stage]][copy_id]

                worker_id = idle_workers.pop()

                # Update O(1) tracking state
                st_name = stage_names[best_stage]
                worker_task_counts[worker_id][st_name] += 1
                iter_num = worker_task_counts[worker_id][st_name]
                max_cap = 4 if best_stage < 2 else 3
                iter_lbl = min(iter_num, max_cap)
                state_lbl = f"{st_name}_{iter_lbl}"

                state_counts["Idle"] -= 1
                state_counts[state_lbl] += 1

                worker_task[worker_id] = (copy_id, best_stage, state_lbl)
                heapq.heappush(
                    event_queue, (current_time + dur, worker_id, copy_id, best_stage)
                )

            if track_history:
                log_state()

        # Start initial scheduling
        schedule_tasks()

        # Main event loop
        terminated_by_walltime = False
        while event_queue:
            event_time, worker_id, copy_id, stage_idx = heapq.heappop(event_queue)

            # Check walltime limit
            if self.walltime_limit is not None and event_time > self.walltime_limit:
                terminated_by_walltime = True
                current_time = self.walltime_limit
                break

            current_time = event_time

            # Process completion and update O(1) state counts
            _, _, state_lbl = worker_task[worker_id]
            state_counts[state_lbl] -= 1
            state_counts["Idle"] += 1

            states[copy_id, stage_idx] = 2  # COMPLETED
            dur = durs_dict[stage_names[stage_idx]][copy_id]
            total_work_completed += dur

            # If it's the last stage, increment completed copies
            if stage_idx == self.L - 1:
                completed_copies_count += 1
            else:
                next_stage = stage_idx + 1
                states[copy_id, next_stage] = 0  # PENDING

                # Estimate runtime for remaining stages using predecessor normalization
                actual_pred_dur = dur
                mean_pred_dur = mean_durs[stage_idx]
                scale_factor = actual_pred_dur / mean_pred_dur

                est_rem_path = sum(
                    mean_durs[j] * scale_factor for j in range(next_stage, self.L)
                )

                # Sorting key
                if policy in [
                    "relaxed_bulk_lrpf",
                    "pipeline_lrpf",
                    "global_lrpf",
                    "seeded_lrpf",
                ]:
                    key = -est_rem_path
                elif policy == "pipeline_srpf":
                    key = est_rem_path
                elif policy == "fifo":
                    key = current_time
                else:
                    key = 0.0

                bisect.insort(ready_queues[next_stage], (float(key), copy_id))

            # Free worker and schedule next tasks
            worker_task[worker_id] = None
            idle_workers.append(worker_id)
            schedule_tasks()

        if terminated_by_walltime:
            makespan = self.walltime_limit
        else:
            makespan = current_time

        avg_occupancy = (total_work_completed / (self.num_workers * makespan)) * 100.0

        sim_duration = time.time() - t0
        print(
            f"Policy {policy} finished in {sim_duration:.2f}s. Makespan: {makespan:.2f}s, Occupancy: {avg_occupancy:.2f}%"
        )

        res = {
            "makespan": makespan,
            "occupancy": avg_occupancy,
            "completed_copies": completed_copies_count,
            "sim_time": sim_duration,
        }
        if track_history:
            res["history"] = history
        return res
