import heapq

_counter = 0


def _next_counter():
    global _counter
    _counter += 1
    return _counter


class TaskHeap:
    def __init__(self):
        self._sub_heaps = {}
        self._size = 0

    def push(self, key, task, group):
        cnt = _next_counter()
        if group not in self._sub_heaps:
            self._sub_heaps[group] = []
        heapq.heappush(self._sub_heaps[group], (key, cnt, task))
        self._size += 1

    def pop(self, skip_groups=None):
        best_group = None
        best_head = None
        for group, heap in self._sub_heaps.items():
            if not heap:
                continue
            if skip_groups is not None and group in skip_groups:
                continue
            head = heap[0]
            if best_head is None or head < best_head:
                best_head = head
                best_group = group
        if best_group is None:
            return None
        self._size -= 1
        return heapq.heappop(self._sub_heaps[best_group])

    @property
    def groups(self):
        return {g for g, h in self._sub_heaps.items() if h}

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def __iter__(self):
        for heap in self._sub_heaps.values():
            yield from heap
