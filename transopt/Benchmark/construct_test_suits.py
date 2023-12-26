from pathlib import Path
from transopt.utils.Register import benchmark_registry
from transopt.Benchmark.BenchBase.ConfigOptBenchmark import TabularOptBenchmark
from transopt.utils import  check
from transopt.Benchmark.BenchBase import (
    TransferOptBenchmark,
    RemoteTransferOptBenchmark,
)


def construct_test_suits(
    tasks: dict = None, seed: int = 0, remote: bool = False, server_url: str = None
) -> TransferOptBenchmark:
    tasks = tasks or {}

    if remote:
        if server_url is None:
            raise ValueError("Server URL must be provided for remote testing.")
        test_suits = RemoteTransferOptBenchmark(server_url, seed)
    else:
        test_suits = TransferOptBenchmark(seed)

    for task_name, task_params in tasks.items():
        budget = task_params["budget"]
        workloads = task_params.get("workloads", [])
        konbs = task_params.get("knobs", None)
        params = task_params.get("params", {})
        tabular = task_params.get("tabular", False)

        if tabular:
            assert 'path' in task_params
            data_path = task_params['path']
            for workload in workloads:
                problem = TabularOptBenchmark(task_name, budget=budget, path=data_path, workload=workload,
                                              task_type='tabular', seed=seed, bounds = None)
                test_suits.add_task(problem)
        else:
            benchmark_cls = benchmark_registry.get(task_name)
            if benchmark_cls is None:
                raise KeyError(f"Task '{task_name}' not found in the benchmark registry.")

        for idx, workload in enumerate(workloads):
            problem = benchmark_cls(
                task_name=f"{task_name}_{workload}",
                task_id=idx,
                budget=budget,
                seed=seed,
                workload=workload,
                knobs=konbs,
                params=params,
            )
            test_suits.add_task(problem)

    return test_suits
