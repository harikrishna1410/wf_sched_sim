import networkx as nx
import numpy as np
from networkx import Graph


class NodeTopology:
    def __init__(self):
        self._graph = Graph()

    @property
    def graph(self):
        return self._graph

    def add_node(self, id: int):
        self._graph.add_node(id)

    def add_inv_bandwidth(self, src: int, dest: int, inv_bandwidth: int = 1):
        self._graph.add_edge(src, dest, inv_bandwidth=inv_bandwidth)


class ComputeNode:
    def __init__(
        self, compute_slots: dict[str, float], compute_slot_counts: dict[str, int]
    ):
        self.compute_slots: dict[str, float] = compute_slots
        self.compute_slot_counts: dict[str, int] = compute_slot_counts
        self._graph = Graph()

    @property
    def graph(self):
        return self._graph

    def add_slot(self, id: str):
        self._graph.add_node(id)

    def add_inv_bandwidth(self, src: str, dest: str, inv_bandwidth: int = 1):
        self._graph.add_edge(src, dest, inv_bandwidth=inv_bandwidth)


class SystemModel:
    def __init__(self, compute_node: ComputeNode, node_topology: NodeTopology):
        self._compute_node = compute_node
        self._node_topology = node_topology
        self._allocated_flag = np.zeros(
            (
                len(self._compute_node.compute_slots),
                self._node_topology.graph.number_of_nodes(),
                max(compute_node.compute_slot_counts.values()),
            ),
            dtype=bool,
        )

        ## Block the slots > max_slots
        for max_slots in self._compute_node.compute_slot_counts.values():
            self._allocated_flag[:, :, max_slots:] = True

        self._slot_name_to_id = {
            k: idx for idx, k in enumerate(compute_node.compute_slots.keys())
        }
        self._slot_id_to_name = {
            idx: k for idx, k in enumerate(compute_node.compute_slots.keys())
        }
        self._free_slots = int(np.sum(~self._allocated_flag))

    @property
    def free_slots(self):
        return self._free_slots

    @property
    def compute_slot_counts(self):
        return self._compute_node.compute_slot_counts

    def is_slot_free(self, slot_name, node_id, slot_id):
        sid = self._slot_name_to_id[slot_name]
        return not self._allocated_flag[sid, node_id, slot_id]

    def get_lowest_inv_bandwidth(self, src: tuple[str, str], dest: tuple[str, str]):
        """Returns 1/bandwidth_i between two compute nodes.
        If a direct edge doesn't exist, returns the path with shortest sum(1/inv_bandwidth) along the path.
        """
        ## intra node comm
        if src[0] == dest[0]:
            try:
                edge = self._compute_node.graph.edges[src[1], dest[1]]
                return edge["inv_bandwidth"]
            except KeyError:
                ## edge doesn't exist. Find shortest path
                return nx.shortest_path_length(
                    self._compute_node.graph,
                    source=src[1],
                    target=dest[1],
                    weight="inv_bandwidth",
                )
        else:
            # internode
            try:
                edge = self._node_topology.graph.edges[src[0], dest[0]]
                return edge["inv_bandwidth"]
            except KeyError:
                return nx.shortest_path_length(
                    self._node_topology.graph,
                    source=src[0],
                    target=dest[0],
                    weight="inv_bandwidth",
                )

    def get_compute(self, slot_name: str):
        return self._compute_node.compute_slots[slot_name]

    def _try_allocate(self, address):
        if self._free_slots == 0:
            return None
        if address is None:
            if np.amin(self._allocated_flag):
                return None
            allocated_slot_id, allocated_node, allocated_slot = np.unravel_index(
                np.argmin(self._allocated_flag),
                shape=self._allocated_flag.shape,
            )
            self._allocated_flag[allocated_slot_id, allocated_node, allocated_slot] = (
                True
            )
            self._free_slots -= 1
            return (
                self._slot_id_to_name[allocated_slot_id],
                int(allocated_node),
                int(allocated_slot),
            )
        elif isinstance(address, str):
            slot_name = address
            slot_id = self._slot_name_to_id[slot_name]
            shape = self._allocated_flag.shape
            if np.amin(self._allocated_flag[slot_id, :, :]):
                return None
            allocated_node, allocated_slot = np.unravel_index(
                np.argmin(self._allocated_flag[slot_id, :, :]),
                shape=shape[1:],
            )
            self._allocated_flag[slot_id, allocated_node, allocated_slot] = True
            self._free_slots -= 1
            return (slot_name, int(allocated_node), int(allocated_slot))
        elif isinstance(address, tuple) and len(address) == 2:
            slot_name = address[0]
            slot_id = self._slot_name_to_id[slot_name]
            node_id = address[1]
            if np.amin(
                self._allocated_flag[
                    slot_id,
                    node_id,
                    : self._compute_node.compute_slot_counts[slot_name],
                ]
            ):
                return None
            allocated_slot = np.argmin(
                self._allocated_flag[
                    slot_id,
                    node_id,
                    : self._compute_node.compute_slot_counts[slot_name],
                ]
            )
            self._allocated_flag[slot_id, node_id, allocated_slot] = True
            self._free_slots -= 1
            return (slot_name, int(node_id), int(allocated_slot))
        elif isinstance(address, tuple) and len(address) == 3:
            slot_name = address[0]
            slot_id = self._slot_name_to_id[slot_name]
            node_id = address[1]
            slot = address[2]
            if self._allocated_flag[slot_id, node_id, slot]:
                return None
            self._allocated_flag[slot_id, node_id, slot] = True
            self._free_slots -= 1
            return (slot_name, node_id, slot)
        else:
            raise ValueError("Unknow address type in allocate!")

    def allocate(self, tasks, addresses):
        from collections import deque

        allocated = deque()
        unallocated = deque()
        for task, address in zip(tasks, addresses):
            if self._free_slots == 0:
                unallocated.append(task)
                continue
            slot = self._try_allocate(address)
            if slot is None:
                unallocated.append(task)
            else:
                allocated.append((task, slot))
        return allocated, unallocated

    def deallocate(self, addresses: list[tuple[str, int, int]]):
        for address in addresses:
            slot_name = address[0]
            slot_id = self._slot_name_to_id[slot_name]
            node_id = address[1]
            slot = address[2]
            if self._allocated_flag[slot_id, node_id, slot]:
                self._allocated_flag[slot_id, node_id, slot] = False
                self._free_slots += 1
            else:
                raise ValueError("Can't deallocate unallocated compute slot")
