import heapq
from abc import ABC, abstractmethod

import numpy as np

from .system import SystemModel
from .workflow import WorkflowModel, WorkflowTask


class Mapper(ABC):
    def __init__(self, name: str, strict: bool = False):
        super().__init__()
        self._name = name
        self._strict = strict

    @abstractmethod
    def map(
        self, tasks, workflow: WorkflowModel, system: SystemModel
    ):
        pass

    @staticmethod
    def _sig(task):
        return (tuple(sorted(task.nslots.items())), task.nnodes)


class SerialGeneralMapper(Mapper):
    def map(self, tasks, workflow: WorkflowModel, system: SystemModel):
        allocated_list = []
        failed_sigs = set()
        remaining = system.free_slots

        while tasks and remaining > 0:
            item = tasks.pop(skip_groups=failed_sigs)
            if item is None:
                break
            key, cnt, task = item
            sig = self._sig(task)
            addresses = [None]
            alloc, fail = system.allocate([task], addresses)
            if alloc:
                allocated_list.append(alloc[0])
                remaining -= 1
            else:
                failed_sigs.add(sig)
                tasks.push(key, task, sig)

        return allocated_list, tasks


class HeterogeneousMapper(Mapper):
    def __init__(self, name: str, strict: bool = False):
        super().__init__(name, strict=strict)

    def map(self, tasks, workflow: WorkflowModel, system: SystemModel):
        allocated = []
        failed_sigs = set()

        while tasks:
            if system.free_slots == 0:
                break
            item = tasks.pop(skip_groups=failed_sigs)
            if item is None:
                break
            key, cnt, task = item
            sig = self._sig(task)
            slots = system.allocate_multi(task.nslots, task.nnodes)
            if slots is None:
                failed_sigs.add(sig)
                tasks.push(key, task, sig)
                if self._strict:
                    break
            else:
                allocated.append((task, slots))

        return allocated, tasks


class PartitionedMapper(Mapper):
    def __init__(self, name: str, stage_nodes: dict[str, list[int]], strict: bool = False):
        super().__init__(name, strict=strict)
        self._stage_nodes = {
            stage: np.array(nodes, dtype=np.intp) for stage, nodes in stage_nodes.items()
        }

    @staticmethod
    def _extract_stage(task):
        return task.name.rsplit("_", 1)[0]

    def map(self, tasks, workflow: WorkflowModel, system: SystemModel):
        allocated = []
        failed_sigs = set()

        while tasks:
            if system.free_slots == 0:
                break
            item = tasks.pop(skip_groups=failed_sigs)
            if item is None:
                break
            key, cnt, task = item
            sig = self._sig(task)
            stage = self._extract_stage(task)
            allowed = self._stage_nodes.get(stage)
            slots = system.allocate_multi(task.nslots, task.nnodes, allowed_nodes=allowed)
            if slots is None:
                failed_sigs.add(sig)
                tasks.push(key, task, sig)
                if self._strict:
                    break
            else:
                allocated.append((task, slots))

        return allocated, tasks


class SortedPipelineMapper(Mapper):
    def __init__(self, name: str, strict: bool = False):
        super().__init__(name, strict=strict)
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

        batch = []
        addresses = []
        free = system.free_slots

        while tasks and len(batch) < free:
            key, cnt, task = tasks.pop()
            sig = self._sig(task)
            wf_name = task.workflow.name
            if wf_name in self._wf_to_worker:
                worker_id = self._wf_to_worker[wf_name]
                if system.is_slot_free("worker", 0, worker_id):
                    batch.append(task)
                    addresses.append(("worker", 0, worker_id))
                else:
                    tasks.push(key, task, sig)
            else:
                if system.free_slots > 0:
                    batch.append(task)
                    addresses.append(("worker", 0))
                else:
                    tasks.push(key, task, sig)

        allocated, failed = system.allocate(batch, addresses)
        for f in failed:
            tasks.push(0, f, self._sig(f))
        for task, slot in allocated:
            if task.workflow.name not in self._wf_to_worker:
                self._wf_to_worker[task.workflow.name] = slot[-1]
        return allocated, tasks
