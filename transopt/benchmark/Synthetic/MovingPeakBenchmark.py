import os
import math
import os
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Dict
from transopt.space.variable import *

from typing import Union, Dict
from transopt.space.variable import *

from sklearn.preprocessing import normalize

from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.space.search_space import SearchSpace
from transopt.space.fidelity_space import FidelitySpace

from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.space.search_space import SearchSpace
from transopt.space.fidelity_space import FidelitySpace

logger = logging.getLogger("MovingPeakBenchmark")

@problem_registry.register("MPB")
class MovingPeakGenerator:
    def __init__(
        self,
        task_name,
        budget,
        budget_type,
        workloads,
        shift_length=3.0,
        height_severity=4.0,
        width_severity=2.0,
        lam=0.5,
        n_peak=2,
        seed=19,
        change_type='markov',  # New parameter to specify change type
        **kwargs
    ):
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        else:
            self.seed = 0
        if 'input_dim' in kwargs['params']:
            self.input_dim = kwargs['params']['input_dim']
        else:
            self.input_dim = 1
        if 'task_type' in kwargs['params']:
            self.task_type = kwargs['params']['task_type']
        else:
            self.task_type = 'non-tabular'
        self.task_name = task_name
        self.budget = budget
        self.budget_type = budget_type
        self.workloads = workloads
        self.n_var = self.input_dim
        self.shift_length = shift_length
        self.height_severity = height_severity
        self.width_severity = width_severity
        self.lam = lam
        self.n_peak = n_peak
        self.change_type = change_type  # Store the change type
        self.var_bound = np.array([[0, 100]] * self.n_var)
        self.height_bound = np.array([[30, 70]] * n_peak)
        self.width_bound = np.array([[1.0, 12.0]] * n_peak)
        self.n_step = len(workloads)
        self.t = 0

        # Initialize peaks, widths, and heights
        current_peak = np.random.random(size=(n_peak, self.n_var)) * np.tile(
            self.var_bound[:, 1] - self.var_bound[:, 0], (n_peak, 1)
        ) + np.tile(self.var_bound[:, 0], (n_peak, 1))

        current_width = (
            np.random.random(size=(n_peak,))
            * (self.width_bound[:, 1] - self.width_bound[:, 0])
            + self.width_bound[:, 0]
        )

        current_height = (
            np.random.random(size=(n_peak,))
            * (self.height_bound[:, 1] - self.height_bound[:, 0])
            + self.height_bound[:, 0]
        )

        previous_shift = normalize(
            np.random.random(size=(n_peak, self.n_var)), axis=1, norm="l2"
        )

        self.peaks = []
        self.widths = []
        self.heights = []

        self.peaks.append(current_peak)
        self.widths.append(current_width)
        self.heights.append(current_height)

        for t in range(1, self.n_step):
            peak_shift = self.cal_peak_shift(previous_shift, t)
            width_shift = self.cal_width_shift()
            height_shift = self.cal_height_shift()
            current_peak = current_peak + peak_shift
            current_height = current_height + height_shift.squeeze()
            current_width = current_width + width_shift.squeeze()
            for i in range(self.n_peak):
                self._fix_bound(current_peak[i, :], self.var_bound)
            self._fix_bound(current_width, self.width_bound)
            self._fix_bound(current_height, self.height_bound)
            previous_shift = peak_shift
            self.peaks.append(current_peak)
            self.widths.append(current_width)
            self.heights.append(current_height)


    def generate_benchmarks(self):
        benchmarks = []
        for t in range(self.n_step):
            peak = self.peaks[t]
            height = self.heights[t]
            width = self.widths[t]
            problem = MovingPeakBenchmark(
                task_name=self.task_name,
                budget=self.budget,
                budget_type=self.budget_type,
                task_id=t,
                workload=t,
                peak=peak,
                height=height,
                width=width,
                seed=self.seed,
                input_dim=self.n_var,
                task_type=self.task_type
            )
            benchmarks.append(problem)
        return benchmarks

    def generate_benchmarks(self):
        benchmarks = []
        for t in range(self.n_step):
            peak = self.peaks[t]
            height = self.heights[t]
            width = self.widths[t]
            problem = MovingPeakBenchmark(
                task_name=self.task_name,
                budget=self.budget,
                budget_type=self.budget_type,
                task_id=t,
                workload=t,
                peak=peak,
                height=height,
                width=width,
                seed=self.seed,
                input_dim=self.n_var,
                task_type=self.task_type
            )
            benchmarks.append(problem)
        return benchmarks

    def cal_width_shift(self):
        width_change = np.random.random(size=(self.n_peak, 1))
        return self.width_severity * width_change

    def cal_height_shift(self):
        height_change = np.random.random(size=(self.n_peak, 1))
        return self.height_severity * height_change

    def cal_peak_shift(self, previous_shift, t):
        peak_change = np.random.random(size=(self.n_peak, self.n_var))
        if self.change_type == 'markov':
            return (1 - self.lam) * self.shift_length * peak_change + self.lam * previous_shift
        elif self.change_type == 'oscillatory':
            return self.shift_length * np.sin(2*t * np.pi / self.n_step) * peak_change
        elif self.change_type == 'poisson':
            return self.shift_length * np.random.poisson(lam=peak_change)
        elif self.change_type == 'autoregressive':
            return self.shift_length * (self.lam * previous_shift + (1 - self.lam) * peak_change)
        else:
            raise ValueError(f"Unknown change type: {self.change_type}")

    def change(self):
        if self.t < self.n_step - 1:
            self.t += 1

    def current_optimal(self, peak_shape=None):
        current_peak = self.peaks[self.t]
        current_height = self.heights[self.t]
        optimal_x = np.atleast_2d(current_peak[np.argmax(current_height)])
        optimal_y = self.f(optimal_x, peak_shape)
        return optimal_x, optimal_y

    def transfer(self, X):
        return (X + 1) * (self.var_bound[:, 1] - self.var_bound[:, 0]) / 2 + (
            self.var_bound[:, 0]
        )

    def normalize(self, X):
        return (
            2
            * (X - (self.var_bound[:, 0]))
            / (self.var_bound[:, 1] - self.var_bound[:, 0])
            - 1
        )

    @property
    def optimizers(self):
        current_peak = self.peaks[self.t]
        current_height = self.heights[self.t]
        optimal_x = np.atleast_2d(current_peak[np.argmax(current_height)])
        optimal_x = self.normalize(optimal_x)
        return optimal_x

    @staticmethod
    def _fix_bound(data, bound):
        for i in range(data.shape[0]):
            if data[i] < bound[i, 0]:
                data[i] = 2 * bound[i, 0] - data[i]
            elif data[i] > bound[i, 1]:
                data[i] = 2 * bound[i, 1] - data[i]
            while data[i] < bound[i, 0] or data[i] > bound[i, 1]:
                data[i] = data[i] * 0.5 + bound[i, 0] * 0.25 + bound[i, 1] * 0.25


class MovingPeakBenchmark(NonTabularProblem):
    problem_type = "synthetic"
    num_variables = []
    num_objectives = 1
    workloads = []
    fidelity = None
    problem_type = "synthetic"
    num_variables = []
    num_objectives = 1
    workloads = []
    fidelity = None
    def __init__(
        self,
        task_name,
        budget,
        budget_type,
        task_id,
        workload,
        peak,
        height,
        width,
        seed,
        input_dim,
        task_type="non-tabular",
        **kwargs
    ):
        self.dimension = input_dim
        self.peak = peak
        self.height = height
        self.width = width
        self.n_peak = len(peak)
        self.peak_shape = kwargs.get("peak_shape", "cone")
        self.task_id = task_id
        self.workload = workload
        
        self.input_dim = input_dim
        
        super(MovingPeakBenchmark, self).__init__(
            task_name=task_name, seed=seed, task_type=task_type, budget=budget, budget_type=budget_type, task_id=task_id, workload=workload
        )

        # Set the peak function based on the peak shape
        self.peak_function = self._select_peak_function(self.peak_shape)

    def _select_peak_function(self, peak_shape):
        if peak_shape == "cone":
            return self.peak_function_cone
        elif peak_shape == "sharp":
            return self.peak_function_sharp
        elif peak_shape == "hilly":
            return self.peak_function_hilly
        else:
            return self.peak_function_cone

    def peak_function_cone(self, x):
        distance = np.linalg.norm(np.tile(x, (self.n_peak, 1)) - self.peak, axis=1)
        return np.max(self.height - self.width * distance)

    def peak_function_sharp(self, x):
        distance = np.linalg.norm(np.tile(x, (self.n_peak, 1)) - self.peak, axis=1)
        return np.max(self.height / (1 + self.width * distance * distance))

    def peak_function_hilly(self, x):
        distance = np.linalg.norm(np.tile(x, (self.n_peak, 1)) - self.peak, axis=1)
        return np.max(
            self.height
            - self.width * distance * distance
            - 0.01 * np.sin(20.0 * distance * distance)
        )
        
    def current_optimal(self):
        current_peak = self.peak
        current_height = self.height
        optimal_x = np.atleast_2d(current_peak[np.argmax(current_height)])
        optimal_y = self.peak_function(optimal_x)
        return optimal_x, optimal_y

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])
        y = self.peak_function(X)
        results = {list(self.objective_info.keys())[0]: float(-y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results
    
    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (0, 100)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss

    def get_fidelity_space(self) -> FidelitySpace:
        fs = FidelitySpace([])
        return fs

    def get_objectives(self) -> Dict:
        return {'f1':'minimize'}

    def get_problem_type(self):
        return "synthetic"

def plot_1d_line_benchmarks(generator, steps=12):
    """
    Plots 1D line plots of the benchmarks generated by the MovingPeakGenerator in a 3x4 grid.

    Parameters
    ----------
    generator : MovingPeakGenerator
        An instance of the MovingPeakGenerator class.
    steps : int
        The number of steps to plot.
    """
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.flatten()

    x = np.linspace(0, 100, 500)  # 1D space

    for t in range(steps):
        peaks, widths, heights = generator.get_MPB()
        current_peaks = peaks[t]
        current_widths = widths[t]
        current_heights = heights[t]

        y = np.zeros_like(x)
        for peak, width, height in zip(current_peaks, current_widths, current_heights):
            y += height * np.exp(-((x - peak[0]) ** 2) / (2 * width ** 2))

        ax = axes[t]
        ax.plot(x, y, label=f'Step {t+1}')
        ax.set_title(f'Step {t+1}')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, max(current_heights) + 10)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Height')
        ax.legend(loc='upper right')

        # Annotate each peak with a red line and x-coordinate
        for i, peak in enumerate(current_peaks):
            ax.axvline(x=peak[0], color='red', linestyle='--', linewidth=1)
            ax.annotate(f'Peak {i+1}\n({peak[0]:.2f})', xy=(peak[0], current_heights[i]), 
                        xytext=(peak[0], current_heights[i] + 5),
                        arrowprops=dict(facecolor='black', arrowstyle='->'),
                        fontsize=8, ha='center')

    plt.tight_layout()
    plt.savefig(f'MovingPeakBenchmark_{generator.change_type}.png')



from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def plot_1d_line_benchmarks_with_gp(generator, steps=12):
    """
    Plots 1D line plots of the benchmarks generated by the MovingPeakGenerator in a 3x4 grid.

    Parameters
    ----------
    generator : MovingPeakGenerator
        An instance of the MovingPeakGenerator class.
    steps : int
        The number of steps to plot.
    """
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.flatten()

    x = np.linspace(0, 100, 500)  # 1D space

    # Sample 16 points from the first step
    sample_x = np.linspace(0, 100, 16).reshape(-1, 1)
    peaks, widths, heights = generator.get_MPB()
    current_peaks = peaks[0]
    current_widths = widths[0]
    current_heights = heights[0]

    # Calculate y values for the sampled points
    sample_y = np.zeros_like(sample_x).flatten()
    for peak, width, height in zip(current_peaks, current_widths, current_heights):
        sample_y += height * np.exp(-((sample_x.flatten() - peak[0]) ** 2) / (2 * width ** 2))

    # Fit Gaussian Process to the sampled data
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(sample_x, sample_y)

    # Predict using the Gaussian Process
    y_pred, sigma = gp.predict(x.reshape(-1, 1), return_std=True)

    for t in range(steps):
        peaks, widths, heights = generator.get_MPB()
        current_peaks = peaks[t]
        current_widths = widths[t]
        current_heights = heights[t]

        y = np.zeros_like(x)
        for peak, width, height in zip(current_peaks, current_widths, current_heights):
            y += height * np.exp(-((x - peak[0]) ** 2) / (2 * width ** 2))

        ax = axes[t]
        # Plot the Gaussian Process fit from first step
        ax.plot(x, y_pred, 'r--', label='GP Fit')
        ax.fill_between(x, y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color='k')
        
        # Plot current step function
        ax.plot(x, y, label=f'Step {t+1}')
        ax.set_title(f'Step {t+1}')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, max(current_heights) + 10)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Height')
        ax.legend(loc='upper right')

        # Annotate each peak with a red line and x-coordinate
        for i, peak in enumerate(current_peaks):
            ax.axvline(x=peak[0], color='red', linestyle='--', linewidth=1)
            ax.annotate(f'Peak {i+1}\n({peak[0]:.2f})', xy=(peak[0], current_heights[i]), 
                        xytext=(peak[0], current_heights[i] + 5),
                        arrowprops=dict(facecolor='black', arrowstyle='->'),
                        fontsize=8, ha='center')

    plt.tight_layout()
    plt.savefig('MovingPeakBenchmark.png')

if __name__ == "__main__":
    # Example usage
    n_var = 1  # 1-dimensional

    # Define the required arguments
    task_name = "example_task"
    budget = 100
    budget_type = "time"
    workloads = [1,2,3,4,5,6,7,8,9,10,11,12]  # Example workloads

    # Create an instance of MovingPeakGenerator with the required arguments
    generator = MovingPeakGenerator(
        task_name=task_name,
        budget=budget,
        budget_type=budget_type,
        workloads=workloads,
        n_var=n_var,
        n_step=12,
        seed=42,
        change_type='markov',
        params={'input_dim': 1}
    )

    plot_1d_line_benchmarks(generator, steps=12)
    # plot_1d_line_benchmarks_with_gp(generator, steps=12)
