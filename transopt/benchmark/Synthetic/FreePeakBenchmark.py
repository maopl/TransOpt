import os
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Dict
from transopt.space.variable import *

from sklearn.preprocessing import normalize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.space.search_space import SearchSpace
from transopt.space.fidelity_space import FidelitySpace

logger = logging.getLogger("FreePeakBenchmark")

class DynamicRotationPeakGenerator:
    def __init__(self, num_peaks=10, num_dimensions=1, change_type='small_step', change_frequency=10000, n_step=12):
        self.num_peaks = num_peaks
        self.num_dimensions = num_dimensions
        self.change_type = change_type
        self.change_frequency = change_frequency
        self.n_step = n_step
        self.t = 0

        # Initialize peaks
        self.heights = np.random.uniform(10, 100, num_peaks)
        self.widths = np.random.uniform(1, 10, num_peaks)
        self.positions = np.random.uniform(-5, 5, (num_peaks, num_dimensions))
        self.rotation_angles = np.random.uniform(-np.pi, np.pi, num_peaks)

        self.positions_history = [self.positions.copy()]

        for _ in range(1, n_step):
            self.apply_dynamic_changes()
            self.positions_history.append(self.positions.copy())

    def apply_dynamic_changes(self):
        """Applies dynamic changes to the landscape based on the specified change type."""
        if self.change_type == 'small_step':
            delta = 0.04 * np.linalg.norm(self.positions) * np.random.uniform(-1, 1)
        elif self.change_type == 'large_step':
            delta = np.linalg.norm(self.positions) * (0.04 * np.sign(np.random.uniform(-1, 1)) + (0.1 - 0.04) * np.random.uniform(-1, 1))
        elif self.change_type == 'random':
            delta = np.random.normal(0, 1)
        elif self.change_type == 'chaotic':
            A = 3.67
            delta = A * (self.positions - np.min(self.positions)) * (1 - (self.positions - np.min(self.positions)) / np.linalg.norm(self.positions))
        elif self.change_type == 'recurrent':
            P = 12
            delta = np.min(self.positions) + np.linalg.norm(self.positions) * (np.sin(2 * np.pi / P + np.random.uniform(0, 2*np.pi)) + 1) / 2
        elif self.change_type == 'recurrent_with_noise':
            noise = np.random.normal(0, 1) * 0.8
            P = 12
            delta = np.min(self.positions) + np.linalg.norm(self.positions) * (np.sin(2 * np.pi / P + np.random.uniform(0, 2*np.pi)) + 1) / 2 + noise
        else:
            raise ValueError("Invalid change type.")
        
        self.positions += delta

    def get_positions(self):
        return self.positions_history

class DynamicRotationPeakBenchmark:
    def __init__(self, num_peaks=10, num_dimensions=1, change_type='small_step', change_frequency=10000, n_step=12):
        self.generator = DynamicRotationPeakGenerator(num_peaks, num_dimensions, change_type, change_frequency, n_step)
        self.num_peaks = num_peaks
        self.num_dimensions = num_dimensions
        self.heights = np.random.uniform(10, 100, num_peaks)
        self.widths = np.random.uniform(1, 10, num_peaks)

    def evaluate(self, x, step):
        """Evaluates the benchmark function at a given point x."""
        x = np.array(x)
        positions = self.generator.get_positions()[step]
        fitness = np.array([self.heights[i] / (1 + self.widths[i] * np.sqrt(np.sum((x - positions[i])**2) / self.num_dimensions)) for i in range(self.num_peaks)])
        return np.max(fitness)
    
    def plot_1d_landscape(self, steps=12, resolution=100):
        """Plots the 1D landscape over multiple time steps."""
        x_vals = np.linspace(-5, 5, resolution)
        plt.figure(figsize=(10, 6))
        
        for step in range(steps):
            y_vals = [self.evaluate([x], step) for x in x_vals]
            plt.plot(x_vals, y_vals, label=f'Step {step+1}')
        
        plt.xlabel("x")
        plt.ylabel("Fitness")
        plt.title("Dynamic Rotation Peak Benchmark Over Time")
        plt.legend()
        plt.savefig(f"DynamicRotationPeakBenchmark_{self.num_dimensions}D_{self.generator.change_type}.png")
        
    
    def plot_2d_landscape(self, steps=12, resolution=100):
        """Plots the 2D landscape over multiple time steps."""
        x_vals = np.linspace(-5, 5, resolution)
        y_vals = np.linspace(-5, 5, resolution)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        plt.figure(figsize=(12, 8))
        
        for step in range(steps):
            Z = np.zeros_like(X)
            for i in range(resolution):
                for j in range(resolution):
                    Z[i, j] = self.evaluate([X[i, j], Y[i, j]], step)
            
            plt.contourf(X, Y, Z, levels=50, cmap='viridis')
            plt.colorbar(label='Fitness')
            plt.title(f"Dynamic Rotation Peak Benchmark - Step {step+1}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.savefig(f"DynamicRotationPeakBenchmark_2D_Step{step+1}.png")
            plt.clf()

# Example Usage
dynamic_benchmark = DynamicRotationPeakBenchmark(num_dimensions=1)
dynamic_benchmark.plot_1d_landscape()
dynamic_benchmark.plot_2d_landscape()