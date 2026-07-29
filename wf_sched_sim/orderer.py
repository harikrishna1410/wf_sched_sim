from abc import ABC, abstractmethod

from .system import SystemModel
from .workflow import WorkflowModel, WorkflowTask


class TaskOrderer(ABC):
    @abstractmethod
    def key(
        self,
        task: WorkflowTask,
        workflow_model: WorkflowModel,
        system_model: SystemModel,
    ):
        pass


class FIFOOrderer(TaskOrderer):
    def key(self, task, workflow_model, system_model):
        return task.start_time


class PipelineOrderer(TaskOrderer):
    def __init__(self, workflow_model: WorkflowModel):
        self._workflow_model = workflow_model

    def key(self, task, workflow_model, system_model):
        return -task.workflow.total_cost


class ShortestFirstOrderer(TaskOrderer):
    def key(self, task, workflow_model, system_model):
        return task.compute_cost


class LongestFirstOrderer(TaskOrderer):
    def key(self, task, workflow_model, system_model):
        return -task.compute_cost
