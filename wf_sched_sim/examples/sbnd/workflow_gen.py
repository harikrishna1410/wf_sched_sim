import argparse
import os
import random
import statistics
from typing import Literal

from wf_sched_sim.mapper import SerialGeneralMapper
from wf_sched_sim.orderer import FIFOOrderer, TaskOrderer
from wf_sched_sim.simulator import Simulator
from wf_sched_sim.system import ComputeNode, NodeTopology, SystemModel
from wf_sched_sim.workflow import Workflow, WorkflowModel, WorkflowTask
from wf_sched_sim.generator import Generator


##Custom orderer for multistage_ordered
class MulitStageOrdered(TaskOrderer):
    def key(self, task: WorkflowTask, workflow_model, system_model):
        return (int(task.name.split("_")[1]), -task.compute_cost)


def build_pinned_pipeline_workflow(task_file="./data/pipeline.txt",
                                       task_distr=None,
                                       pipeline_num=None):
    task_times_s = []
    
    if task_file and task_distr is None:
        with open("./data/pipeline.txt", "r") as f:
            task_times_s = [float(l) * 60.0 for l in f.readlines()[1:]]
            #print(f"mean: {statistics.mean(task_times_s)}")
            #print(f"ntasks: {len(task_times_s)}")
    elif task_distr is not None and pipeline_num is not None:
        g = Generator(task_distr)
        # Generator returns per-stage tuples in minutes. This is a single-stage
        # workflow, so flatten each 1-tuple to a scalar and convert to seconds,
        # matching the datafile path (which yields a list of scalar seconds).
        task_times_s = [t[0] * 60.0
                        for t in g.draw_task_times(pipeline_num=pipeline_num)]

    workflows = []
    for id, t in enumerate(task_times_s):
        w = Workflow(name=f"pipeline_{id}")
        w.add_task(
            WorkflowTask(
                name="stage_1",
                compute_cost=t,
                nslots={"worker": 1},
                nnodes=1,
                comm_size=0.0,
            )
        )
        workflows.append(w)
    model = WorkflowModel(workflows=workflows)
    return model


def build_multi_stage_ordered_workflow(task_file="./data/prio_runtimes.txt",
                                       task_distr=None,
                                       pipeline_num=None):
    task_times_s = []
    if task_file and task_distr is None:
        with open(task_file, "r") as f:
            task_times_s = [
                (float(l.strip().split()[0]), float(l.strip().split()[1]))
                for l in f.readlines()[1:]
            ]
    elif task_distr is not None and pipeline_num is not None:
        g = Generator(task_distr)
        # Generator draws times in minutes; convert to seconds to match datafile path
        task_times_s = [tuple(x * 60.0 for x in t)
                        for t in g.draw_task_times(pipeline_num=pipeline_num)]
    # random.shuffle(task_times_s)

    workflows = []
    for id, t in enumerate(task_times_s):
        w = Workflow(name=f"pipeline_{id}")
        w.add_task(
            WorkflowTask(
                name="stage_1",
                compute_cost=t[0],
                nslots={"worker": 1},
                nnodes=1,
                comm_size=0.0,
            )
        )
        w.add_task(
            WorkflowTask(
                name="stage_2",
                compute_cost=t[1],
                nslots={"worker": 1},
                nnodes=1,
                comm_size=0.0,
            )
        )
        w.add_edge(("stage_1", "stage_2"))
        workflows.append(w)
    model = WorkflowModel(workflows=workflows)
    return model


def build_system(nworkers=26112):
    node = ComputeNode(
        compute_slots={"worker": 1.0}, compute_slot_counts={"worker": nworkers}
    )
    topology = NodeTopology()
    topology.add_node(0)
    system = SystemModel(compute_node=node, node_topology=topology)
    return system


def main(policy: Literal["pinned", "ordered"] = "pinned",
         task_distr=None,
         pipeline_num=104448,
         suffix="_gen"):
    ## Build workflow
    if policy == "pinned":
        wf = build_pinned_pipeline_workflow(task_distr=task_distr,
                                            pipeline_num=pipeline_num)
    elif policy == "ordered":
        wf = build_multi_stage_ordered_workflow(task_distr=task_distr,
                                                pipeline_num=pipeline_num)
    else:
        raise ValueError

    ## 256 nodes, 102 workers per node
    system = build_system()

    # create a mapper
    mapper = SerialGeneralMapper(name="greedy_mapper")

    if policy == "pinned":
        # create an orderer
        orderer = FIFOOrderer()
    elif policy == "ordered":
        orderer = MulitStageOrdered()

    # Finally, create the simulator
    sim = Simulator(workflow_model=wf, system=system, mapper=mapper, orderer=orderer)

    result = sim.run(
        telemetry=True, fname=f"{os.getcwd()}/telemetry_{policy}{suffix}.csv", freq=100
    )
    for k, v in result.items():
        print(f"{k}:{v}")


if __name__ == "__main__":

    task_distr = [[36.89, 11.72**2]]
    main(policy="pinned", task_distr=task_distr, pipeline_num=104448, suffix="_gen")
    task_distr = [[11.61,  1.91**2],
                  [26.39, 10.38**2]]
    main(policy='ordered', task_distr=task_distr, pipeline_num=104448, suffix="_gen")
