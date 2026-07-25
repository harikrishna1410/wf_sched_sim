import heapq
from collections import deque
from .workflow import WorkflowModel, WorkflowTask
from .system import SystemModel
from .mapper import Mapper
from .orderer import TaskOrderer


class Simulator:
    def __init__(self, workflow_model: WorkflowModel, system: SystemModel, mapper: Mapper, orderer: TaskOrderer):
        self._workflow_model = workflow_model
        self._system = system
        self._mapper = mapper
        self._orderer = orderer

    def run(self):
        event_queue = []
        waiting = deque()
        completed_count = 0
        current_time = 0.0

        ready = self._workflow_model.ready_tasks()
        waiting = self._schedule_initial(ready, waiting, event_queue, current_time)

        while event_queue:
            current_time, wf_name, task, slot = heapq.heappop(event_queue)
            completed_count += 1

            self._system.deallocate([slot])
            self._workflow_model.mark_completed({wf_name: [task.name]}, {wf_name: [current_time]})

            new_ready = self._workflow_model.ready_tasks([wf_name])
            waiting = self._schedule(new_ready, waiting, event_queue, current_time)

        return {
            "makespan": current_time,
            "completed_tasks": completed_count,
            "workflow_done": self._workflow_model.done,
            "unscheduled": len(waiting),
        }

    def _schedule_initial(self, ready, waiting, event_queue, current_time):
        all_tasks = []
        for wf_name, tasks in ready.items():
            all_tasks.extend(tasks)
        if not all_tasks:
            return waiting
        all_tasks = self._orderer.order(all_tasks, self._workflow_model, self._system)
        waiting.extend(all_tasks)
        return self._assign(waiting, event_queue, current_time)

    def _schedule(self, new_ready, waiting, event_queue, current_time):
        for wf_name, tasks in new_ready.items():
            for task in tasks:
                self._orderer.insert(waiting, task, self._workflow_model, self._system)

        if self._system.free_slots == 0 or not waiting:
            return waiting

        return self._assign(waiting, event_queue, current_time)

    def _assign(self, waiting, event_queue, current_time):
        allocated, unallocated = self._mapper.map(waiting, self._workflow_model, self._system)
        for task, slot in allocated:
            compute = self._system.get_compute(slot[0])
            task.start_time = max(task.start_time, current_time)
            completion_time = task.start_time + task.compute_cost / compute
            heapq.heappush(event_queue, (completion_time, task.workflow.name, task, slot))
        return unallocated
