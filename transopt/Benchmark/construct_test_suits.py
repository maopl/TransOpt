from transopt.utils.Register import benchmark_registry
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
        benchmark = task_name
        budget = task_params["budget"]
        workloads = task_params["workloads"]
        params = task_params.get("params", {})

        benchmark_cls = benchmark_registry.get(benchmark)
        if benchmark_cls is None:
            raise KeyError(f"Task '{benchmark}' not found in the benchmark registry.")

        for idx, workload in enumerate(workloads):
            problem = benchmark_cls(
                task_name=f"{task_name}_{workload}",
                task_id=idx,
                budget=budget,
                seed=seed,
                workload=workload,
                params=params,
            )
            test_suits.add_task(problem)

    return test_suits
