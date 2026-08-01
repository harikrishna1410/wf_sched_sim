import numpy as np

class Generator:
    def __init__(
        self,
        task_distr: list,
    ):
        self._task_distr = task_distr

    def draw_task_times(self, pipeline_num):
        # For now, do a simple Gaussian distribution for each task type
        # task_distr will be of form [[mean_a, sig2_a], [mean_b, sig2_b], ...]
        task_times = []
        for mean, sig2 in self._task_distr:
            durs = np.random.normal(mean, np.sqrt(sig2), pipeline_num)
            durs = np.clip(durs, 0.001, None)
            durs.sort()
            task_times.append(durs)
        return list(zip(*task_times))
        
