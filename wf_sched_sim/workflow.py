from collections import deque
from dataclasses import dataclass

_DONE_SENTINAL = b"done"


@dataclass
class WorkflowTask:
    name: str
    compute_cost: float
    comm_size: float
    nslots: dict
    workflow: "Workflow" = None
    start_time: float = 0.0
    stop_time: float = 0.0


class Workflow:
    def __init__(self, name: str, output_queue: list | None = None):
        self._name = name
        self._tasks: dict[str, WorkflowTask] = {}
        self._successors: dict[str, list[str]] = {}
        self._predecessors: dict[str, list[str]] = {}
        self._dependency_counter: dict[str, int] = {}
        self._unfinished: set[str] = set()
        self._output_queue = output_queue

    @property
    def name(self):
        return self._name

    @property
    def tasks(self):
        return self._tasks

    @property
    def total_cost(self):
        return sum(t.compute_cost for t in self._tasks.values())

    @property
    def done(self):
        return len(self._unfinished) == 0

    def add_task(self, task: WorkflowTask):
        task.workflow = self
        self._tasks[task.name] = task
        self._successors[task.name] = []
        self._predecessors[task.name] = []
        self._dependency_counter[task.name] = 0
        self._unfinished.add(task.name)

    def add_edge(self, edge: tuple[str, str]):
        src, dst = edge
        self._successors[src].append(dst)
        self._predecessors[dst].append(src)
        self._dependency_counter[dst] += 1

    def set_output_queue(self, output_queue: list):
        self._output_queue = output_queue
        for task_name in self._unfinished:
            if self._dependency_counter[task_name] == 0:
                self._output_queue.append(self._tasks[task_name])

    def successors(self, task_name: str):
        return self._successors[task_name]

    def predecessors(self, task_name: str):
        return self._predecessors[task_name]

    def mark_completed(self, task_names: list[str], stop_times: list[float]):
        for task_id, task_name in enumerate(task_names):
            if task_name not in self._unfinished:
                continue

            self._unfinished.discard(task_name)
            self._tasks[task_name].stop_time = stop_times[task_id]
            for successor in self._successors[task_name]:
                self._dependency_counter[successor] -= 1
                if self._dependency_counter[successor] == 0:
                    self._tasks[successor].start_time = stop_times[task_id]
                    self._output_queue.append(self._tasks[successor])

        if len(self._unfinished) == 0:
            self._output_queue.append(_DONE_SENTINAL)


class WorkflowModel:
    def __init__(self, workflows: list[Workflow]):
        self._workflows: dict[str, Workflow] = {}
        self._output_queues = {w.name: deque() for w in workflows}
        for w in workflows:
            w.set_output_queue(self._output_queues[w.name])
            self._workflows[w.name] = w
        self._unfinished = set(self._workflows.keys())

    @property
    def workflows(self):
        return self._workflows

    @property
    def done(self):
        return len(self._unfinished) == 0

    def mark_completed(self, completed: dict[str, list[str]], stop_times: dict[str, list[int]]):
        for wf_name, task_names in completed.items():
            self._workflows[wf_name].mark_completed(task_names, stop_times=stop_times[wf_name])

    def ready_tasks(self, wf_names: list[str] | None = None) -> dict[str, list[WorkflowTask]]:
        ret = {}
        names = wf_names if wf_names is not None else self._output_queues.keys()
        for wf_name in names:
            output_queue = self._output_queues[wf_name]
            if not output_queue:
                continue
            tasks = []
            for item in output_queue:
                if item is _DONE_SENTINAL:
                    self._unfinished.discard(wf_name)
                else:
                    tasks.append(item)
            output_queue.clear()
            if tasks:
                ret[wf_name] = tasks
        return ret
