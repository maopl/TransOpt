from transopt.benchmark.Synthetic.MovingPeakBenchmark import MovingPeakGenerator
from transopt.optimizer.pso.mQSO import mQSO
import numpy as np

if __name__ == "__main__":
    task_name = "example_task"
    budget = 100
    budget_type = "time"
    workloads = [1,2,3,4,5,6,7,8,9,10,11,12]  # Example workloads
    n_var = 1  # 1-dimensional
    
    generator = MovingPeakGenerator(
        task_name=task_name,
        budget=budget,
        budget_type=budget_type,
        workloads=workloads,
        n_var=n_var,
        n_step=12,
        seed=42,
        change_type='oscillatory',
        params={'input_dim': 1}
    )
    
    problems = generator.generate_benchmarks()

    for problem in problems:
        mQSO(visualization_over_optimization=1, peak_number=1, change_frequency=1, dimension=1,
              shift_severity=1, environment_number=1, run_number=1, benchmark_name='MovingPeakBenchmark')