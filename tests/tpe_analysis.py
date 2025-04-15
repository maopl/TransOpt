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

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        return self.compute_probability_improvement(x_test)

# Example usage with high-dimensional data
np.random.seed(42)
x_samples = np.random.uniform(-10, 10, (100, 5))  # 5-dimensional data
y_samples = np.sum((x_samples - 3) ** 2, axis=1)  # Objective function

# # Instantiate SimpleTPE
simple_tpe = TPE()

# # Fit the model
simple_tpe.fit(x_samples, y_samples)

# # Test data
x_test = np.random.uniform(-10, 10, (100, 5))

# # Predict probability improvement
pi = simple_tpe.predict(x_test)

# Visualization for one dimension (for illustration purposes)
plt.figure(figsize=(10, 6))
plt.hist(pi, bins=30, alpha=0.7, label='Probability Improvement')
plt.xlabel("Probability Improvement")
plt.ylabel("Frequency")
plt.title("Probability Improvement Distribution")
plt.legend()
plt.savefig("tpe_analysis_hist.png")

# 选择一个维度进行可视化
dimension = 0  # 选择要可视化的维度
plt.figure(figsize=(10, 6))
plt.scatter(x_test[:, dimension], pi, label='Probability Improvement', color='purple', alpha=0.5)
plt.xlabel(f"x (Dimension {dimension})")
plt.ylabel("Probability Improvement")
plt.title(f"Probability Improvement vs x (Dimension {dimension})")
plt.legend()
plt.savefig("tpe_analysis.png")