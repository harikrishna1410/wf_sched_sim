from abc import ABC
from abc import abstractmethod
from .workflow import WorkflowModel
from .workflow import WorkflowTask
from .system import SystemModel


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
    """
    A general mapper that maps to a free compute slot
    """

    def map(
        self, tasks: list[WorkflowTask], workflow: WorkflowModel, system: SystemModel
    ):
        assert all(len(task.nslots) == 1 for task in tasks) and all(
            [sum(task.nslots.values()) == 1 for task in tasks]
        ), f"{type(self)} can only map serial tasks"
        return system.allocate([None] * len(tasks))
