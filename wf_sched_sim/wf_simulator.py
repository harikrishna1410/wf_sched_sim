import heapq
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
        waiting = []
        completed_count = 0
        current_time = 0.0

        ready = self._workflow_model.ready_tasks()
        self._schedule_initial(ready, waiting, event_queue, current_time)

        while event_queue:
            current_time, wf_name, task, slot = heapq.heappop(event_queue)
            completed_count += 1

            self._system.deallocate([slot])
            self._workflow_model.mark_completed({wf_name: [task.name]}, {wf_name: [current_time]})

            new_ready = self._workflow_model.ready_tasks([wf_name])
            self._schedule(new_ready, waiting, event_queue, current_time)

        return {
            "makespan": current_time,
            "completed_tasks": completed_count,
            "workflow_done": self._workflow_model.done,
            "unscheduled": len(waiting),
        }

    def _schedule_initial(self, ready, waiting, event_queue, current_time):
        all_tasks = []
        for wf_name, tasks in ready.items():
            for task in tasks:
                all_tasks.append((wf_name, task))
        if not all_tasks:
            return
        all_tasks = self._orderer.order(all_tasks)
        self._assign(all_tasks, waiting, event_queue, current_time)

    def _schedule(self, new_ready, waiting, event_queue, current_time):
        for wf_name, tasks in new_ready.items():
            for task in tasks:
                self._orderer.insert(waiting, (wf_name, task))

        free = self._system.free_slots
        if free == 0 or not waiting:
            return
        to_schedule = waiting[:free]
        del waiting[:free]
        self._assign(to_schedule, waiting, event_queue, current_time)

    def _assign(self, to_schedule, waiting, event_queue, current_time):
        task_objs = [t[1] for t in to_schedule]
        slots = self._mapper.map(task_objs, self._workflow_model, self._system)
        for (wf_name, task), slot in zip(to_schedule, slots):
            if slot is None:
                self._orderer.insert(waiting, (wf_name, task))
                continue
            slot_name = slot[0]
            compute = self._system.get_compute(slot_name)
            task.start_time = max(task.start_time, current_time)
            completion_time = task.start_time + task.compute_cost / compute
            heapq.heappush(event_queue, (completion_time, wf_name, task, slot))
