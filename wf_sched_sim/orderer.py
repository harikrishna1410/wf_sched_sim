from abc import ABC, abstractmethod
from bisect import insort

from .system import SystemModel
from .workflow import WorkflowModel, WorkflowTask


class TaskOrderer(ABC):
    @abstractmethod
    def order(
        self,
        tasks: list[WorkflowTask],
        workflow_model: WorkflowModel,
        system_model: SystemModel,
    ) -> list[WorkflowTask]:
        pass

    @abstractmethod
    def key(
        self,
        task: WorkflowTask,
        workflow_model: WorkflowModel,
        system_model: SystemModel,
    ):
        pass

    def insert(
        self,
        waiting,
        task: WorkflowTask,
        workflow_model: WorkflowModel,
        system_model: SystemModel,
    ):
        insort(waiting, task, key=lambda t: self.key(t, workflow_model, system_model))


class FIFOOrderer(TaskOrderer):
    def order(
        self,
        tasks: list[WorkflowTask],
        workflow_model: WorkflowModel,
        system_model: SystemModel,
    ) -> list[WorkflowTask]:
        tasks.sort(key=lambda t: self.key(t, workflow_model, system_model))
        return tasks

    def key(
        self,
        task: WorkflowTask,
        workflow_model: WorkflowModel,
        system_model: SystemModel,
    ):
        return task.start_time


class PipelineOrderer(TaskOrderer):
    def __init__(self, workflow_model: WorkflowModel):
        self._workflow_model = workflow_model

    def order(
        self,
        tasks: list[WorkflowTask],
        workflow_model: WorkflowModel,
        system_model: SystemModel,
    ) -> list[WorkflowTask]:
        tasks.sort(key=lambda t: self.key(t, workflow_model, system_model))
        return tasks

    def key(
        self,
        task: WorkflowTask,
        workflow_model: WorkflowModel,
        system_model: SystemModel,
    ):
        return -task.workflow.total_cost

    def insert(
        self,
        waiting,
        task: WorkflowTask,
        workflow_model: WorkflowModel,
        system_model: SystemModel,
    ):
        waiting.appendleft(task)
