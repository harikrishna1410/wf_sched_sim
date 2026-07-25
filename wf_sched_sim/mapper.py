from abc import ABC, abstractmethod

from .system import SystemModel
from .workflow import WorkflowModel, WorkflowTask


class Mapper(ABC):
    def __init__(self, name: str):
        super().__init__()
        self._name = name

    @abstractmethod
    def map(
        self, tasks: list[WorkflowTask], workflow: WorkflowModel, system: SystemModel
    ):
        pass


class SerialGeneralMapper(Mapper):
    def map(self, tasks, workflow: WorkflowModel, system: SystemModel):
        n = min(len(tasks), system.free_slots)
        batch = [tasks.popleft() for _ in range(n)]
        addresses = [None] * n
        allocated, failed = system.allocate(batch, addresses)
        for task in reversed(failed):
            tasks.appendleft(task)
        return allocated, tasks


class SortedPipelineMapper(Mapper):
    def __init__(self, name: str):
        super().__init__(name)
        self._wf_to_worker = None

    def _ensure_mapping(self, workflow_model: WorkflowModel, system_model: SystemModel):
        if self._wf_to_worker is not None:
            return
        num_workers = system_model.compute_slot_counts["worker"]
        wfs = sorted(
            workflow_model.workflows.values(), key=lambda w: w.total_cost, reverse=True
        )[:num_workers]
        self._wf_to_worker = {wf.name: i % num_workers for i, wf in enumerate(wfs)}

    def map(self, tasks, workflow: WorkflowModel, system: SystemModel):
        self._ensure_mapping(workflow, system)
        from collections import deque

        batch = []
        addresses = []
        skipped = deque()
        free = system.free_slots
        while tasks and len(batch) < free:
            task = tasks.popleft()
            wf_name = task.workflow.name
            if wf_name in self._wf_to_worker:
                worker_id = self._wf_to_worker[wf_name]
                if system.is_slot_free("worker", 0, worker_id):
                    batch.append(task)
                    addresses.append(("worker", 0, worker_id))
                else:
                    skipped.append(task)
            else:
                if system.free_slots > 0:
                    batch.append(task)
                    addresses.append(("worker", 0))
                else:
                    skipped.append(task)
        allocated, failed = system.allocate(batch, addresses)
        skipped.extend(failed)
        skipped.extend(tasks)
        for task, slot in allocated:
            if task.workflow.name not in self._wf_to_worker:
                self._wf_to_worker[task.workflow.name] = slot[-1]
        return allocated, skipped
