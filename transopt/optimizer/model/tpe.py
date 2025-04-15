import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from typing import Tuple

class TPE:
    def __init__(self, percentile: float = 25):
        self.percentile = percentile
        self.kde_l = None
        self.kde_g = None
        self.y_star = None

    def fit(self, x_samples: np.ndarray, y_samples: np.ndarray) -> None:
        self.x_samples = x_samples
        if len(y_samples.shape) == 2:
            y_samples = y_samples.flatten()
        self.y_samples = y_samples
        self.y_star = np.percentile(y_samples, self.percentile)
        
        # Split samples based on y_star
        self.x_l = x_samples[y_samples < self.y_star]
        self.x_g = x_samples[y_samples >= self.y_star]
        
        # Use multivariate KDE for high-dimensional data
        self.kde_l = gaussian_kde(self.x_l.T)
        self.kde_g = gaussian_kde(self.x_g.T)

    def get_density_estimates(self, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p_lx = self.kde_l(x_test.T)
        p_gx = self.kde_g(x_test.T)
        return p_lx, p_gx

    def compute_probability_improvement(self, x_test: np.ndarray) -> np.ndarray:
        p_lx, p_gx = self.get_density_estimates(x_test)
        pi = p_gx / (p_lx + 1e-12)  # Avoid division by zero
        return pi

    def predict(self, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pred = self.compute_probability_improvement(x_test).reshape(-1, 1)
        std = np.zeros_like(pred)
        return pred, std
    


