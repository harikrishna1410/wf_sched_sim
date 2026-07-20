from abc import ABC, abstractmethod
from bisect import insort
from .workflow import WorkflowTask


class TaskOrderer(ABC):
    @abstractmethod
    def order(self, tasks: list[tuple[str, WorkflowTask]]) -> list[tuple[str, WorkflowTask]]:
        pass

    @abstractmethod
    def key(self, task: tuple[str, WorkflowTask]):
        pass

    def insert(self, sorted_list: list[tuple[str, WorkflowTask]], task: tuple[str, WorkflowTask]):
        insort(sorted_list, task, key=self.key)


class FIFOOrderer(TaskOrderer):
    def order(self, tasks: list[tuple[str, WorkflowTask]]) -> list[tuple[str, WorkflowTask]]:
        tasks.sort(key=self.key)
        return tasks

    def key(self, task: tuple[str, WorkflowTask]):
        return task[1].start_time
