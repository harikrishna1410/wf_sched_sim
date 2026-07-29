import heapq

from .mapper import Mapper
from .orderer import TaskOrderer
from .system import SystemModel
from .task_heap import TaskHeap
from .workflow import WorkflowModel, WorkflowTask


class Simulator:
    def __init__(
        self,
        workflow_model: WorkflowModel,
        system: SystemModel,
        mapper: Mapper,
        orderer: TaskOrderer,
    ):
        self._workflow_model = workflow_model
        self._system = system
        self._mapper = mapper
        self._orderer = orderer

    def _push_task(self, task_heap, task):
        key = self._orderer.key(task, self._workflow_model, self._system)
        sig = Mapper._sig(task)
        task_heap.push(key, task, sig)

    def run(self, debug=False):
        event_queue = []
        waiting = TaskHeap()
        completed_count = 0
        current_time = 0.0

        ready = self._workflow_model.ready_tasks()
        waiting = self._schedule_initial(ready, waiting, event_queue, current_time)

        if debug:
            print(
                f"  [init] events={len(event_queue)} waiting={len(waiting)} free_slots={self._system.free_slots}"
            )

        completed_names = {}
        while event_queue:
            current_time, wf_name, task, slots = heapq.heappop(event_queue)
            completed_count += 1

            if isinstance(slots, tuple):
                self._system.deallocate([slots])
            else:
                self._system.deallocate(slots)
            self._workflow_model.mark_completed(
                {wf_name: [task.name]}, {wf_name: [current_time]}
            )

            new_ready = self._workflow_model.ready_tasks([wf_name])
            waiting = self._schedule(new_ready, waiting, event_queue, current_time)

            completed_stage = task.name.rsplit("_", 1)[0]
            if completed_stage not in completed_names:
                completed_names[completed_stage] = 1
            else:
                completed_names[completed_stage] += 1

            if debug and completed_count % 10000 == 0:
                waiting_names = {}
                for _, _, w in waiting:
                    stage = w.name.rsplit("_", 1)[0]
                    waiting_names[stage] = waiting_names.get(stage, 0) + 1
                print(
                    f"  [t={current_time:.4f}] completed={completed_count} events={len(event_queue)} waiting={len(waiting)} free_slots={self._system.free_slots} waiting_by_stage={waiting_names} completed_by_stage={completed_names}"
                )

        return {
            "makespan": current_time,
            "completed_tasks": completed_count,
            "workflow_done": self._workflow_model.done,
            "unscheduled": len(waiting),
        }

    def _schedule_initial(self, ready, waiting, event_queue, current_time):
        for wf_name, tasks in ready.items():
            for task in tasks:
                self._push_task(waiting, task)
        if not waiting:
            return waiting
        return self._assign(waiting, event_queue, current_time)

    def _schedule(self, new_ready, waiting, event_queue, current_time):
        for wf_name, tasks in new_ready.items():
            for task in tasks:
                self._push_task(waiting, task)

        if self._system.free_slots == 0 or not waiting:
            return waiting

        return self._assign(waiting, event_queue, current_time)

    def _assign(self, waiting, event_queue, current_time):
        allocated, unallocated = self._mapper.map(
            waiting, self._workflow_model, self._system
        )
        for task, slots in allocated:
            if isinstance(slots, tuple):
                compute = self._system.get_compute(slots[0])
            else:
                compute = self._system.get_compute(slots[0][0])
            task.start_time = max(task.start_time, current_time)
            completion_time = task.start_time + task.compute_cost / compute
            heapq.heappush(
                event_queue, (completion_time, task.workflow.name, task, slots)
            )
        return unallocated
