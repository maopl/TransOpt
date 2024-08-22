Benchmark Problems
==================
This

.. admonition:: Overview
   :class: info

   - `Register <https://link-to-definition>`_: How to register a new optimization problem to TransOpt
   - `The list of the Test Problems <https://link-to-test-problems>`_: Diverse Test Problems available in `TransOpt <https://link-to-pymoo>`_
   - `Parallelization <https://link-to-parallelization>`_: How to parallelize function evaluations


Register
--------

Synthetic Problems
------------------

The synthetic problems in this section are widely used in the optimization literature for benchmarking optimization algorithms. These problems exhibit diverse characteristics and levels of complexity, making them ideal for testing the robustness and efficiency of different optimization strategies. Below is an overview of the synthetic problems included in this benchmark suite:

- **Sphere:** A simple convex problem that is often used as a baseline. The global minimum is located at the origin, and the objective function value increases quadratically with distance from the origin.

- **Rastrigin:** A non-convex problem characterized by a large number of local minima, making it challenging for optimization algorithms to find the global minimum.

- **Schwefel:** Known for its complex landscape with many local minima, the Schwefel function requires optimization algorithms to balance exploration and exploitation effectively.

- **Ackley:** A multi-modal function with a nearly flat outer region and a large hole at the center, making it difficult for algorithms to escape local minima and converge to the global minimum.

- **Levy:** A multi-modal problem with a complex landscape that tests an algorithm's ability to handle irregularities and identify global optima.

- **Griewank:** A function with many widespread local minima, making it challenging to converge to the global optimum. It is often used to assess the ability of algorithms to avoid getting trapped in local minima.

- **Rosenbrock:** A non-convex problem with a narrow, curved valley that contains the global minimum. This function is commonly used to test the convergence properties of optimization algorithms.

- **Dropwave:** A challenging multi-modal function with steep drops, requiring careful search strategies to avoid local minima.

- **Langermann:** This problem has many local minima and a highly irregular structure, testing an algorithm's ability to explore complex search spaces.

- **Rotated Hyper-Ellipsoid:** A rotated version of the ellipsoid function, which tests an algorithm's capability to optimize problems with rotated and ill-conditioned landscapes.

- **Sum of Different Powers:** A problem where each term in the sum contributes differently to the overall objective, requiring optimization algorithms to handle varying sensitivities across dimensions.

- **Styblinski-Tang:** A function with multiple global minima, commonly used to test an algorithm's ability to avoid suboptimal solutions.

- **Powell:** A problem designed to challenge optimization algorithms with a mixture of convex and non-convex characteristics across different dimensions.

- **Dixon-Price:** This function has a smooth, narrow valley leading to the global minimum, testing an algorithm’s ability to navigate such features.

- **Ellipsoid:** A test problem that features high conditioning and elliptical level sets, requiring algorithms to efficiently search in skewed spaces.

- **Discus:** A variant of the sphere function with a large difference in scale between the first variable and the rest, making it a test of handling unbalanced scales.

- **BentCigar:** A highly anisotropic function where one direction has a much larger scale than the others, challenging algorithms to adjust their search strategies accordingly.

- **SharpRidge:** This function has a sharp ridge along one dimension, testing an algorithm's ability to optimize in narrow, high-gradient regions.

- **Katsuura:** A multi-fractal function that combines periodicity and complexity, testing the capability of algorithms to explore intricate landscapes.

- **Weierstrass:** A problem with a fractal structure, characterized by a large number of local minima and requiring algorithms to handle varying scales of roughness.

- **Different Powers:** A problem where each term contributes differently to the objective, challenging algorithms to manage varying sensitivities and scales.

- **Trid:** A function that has a curved and ridge-like structure, often used to assess the convergence properties of optimization algorithms.

- **LinearSlope:** A simple linear function with a varying slope across dimensions, used to test the basic exploration capabilities of optimization methods.

- **Elliptic:** Similar to the Ellipsoid function but with exponentially increasing scales, testing an algorithm’s ability to search efficiently in poorly conditioned spaces.

- **PERM:** A complex combinatorial problem that combines different power terms, testing an algorithm’s ability to handle permutation-based search spaces.

- **Power Sum:** A problem where each dimension contributes a power sum to the objective, requiring algorithms to handle large variations in sensitivity across variables.

- **Zakharov:** A problem with a complex, non-linear interaction between variables, used to test an algorithm’s ability to navigate multi-variable coupling.

- **Six-Hump Camel:** A low-dimensional, multi-modal problem with several local minima, requiring precise search strategies to find the global optimum.

- **Michalewicz:** A problem known for its challenging steepness and periodicity, making it difficult for algorithms to locate the global minimum.

- **Moving Peak:** A dynamic optimization problem where the objective function changes over time, used to assess an algorithm’s adaptability to changing landscapes.

These problems collectively provide a comprehensive suite for evaluating optimization algorithms across a broad range of difficulties, including convexity, multi-modality, separability, and conditioning.

+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Problem name             | Mathematical formulation                                                                     | Decision space                                   |
+==========================+=============================================================================================+==================================================+
| Sphere                   | :math:`f(\mathbf{x}) = \sum_{i=1}^d x_i^2`                                                  | :math:`x_i \in [-5.12, 5.12]`                    |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Rastrigin                | :math:`f(\mathbf{x}) = 10 d + \sum_{i=1}^d \left[ x_i^2 - 10 \cos(2 \pi x_i) \right]`        | :math:`x_i \in [-32.768, 32.768]`                |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Schwefel                 | :math:`f(\mathbf{x}) = 418.9829 d - \sum_{i=1}^d x_i \sin\left(\sqrt{\left|x_i\right|}\right)`| :math:`x_i \in [-500, 500]`                      |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Ackley                   | :math:`f(\mathbf{x}) = -a \exp \left(-b \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2}\right)`        | :math:`x_i \in [-32.768, 32.768]`                |
|                          | :math:`-\exp \left(\frac{1}{d} \sum_{i=1}^d \cos \left(c x_i\right)\right) + a + \exp(1)`   |                                                  |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Levy                     | :math:`f(\mathbf{x}) = \sin^2\left(\pi w_1\right) + \sum_{i=1}^{d-1}\left(w_i - 1\right)^2` | :math:`x_i \in [-10, 10]`                        |
|                          | :math:`\left[1 + 10 \sin^2\left(\pi w_i + 1\right)\right] + \left(w_d - 1\right)^2`         |                                                  |
|                          | :math:`\left[1 + \sin^2\left(2 \pi w_d\right)\right], w_i = 1 + \frac{x_i - 1}{4}`          |                                                  |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Griewank                 | :math:`f(\mathbf{x}) = \sum_{i=1}^d \frac{x_i^2}{4000} - \prod_{i=1}^d \cos\left(\frac{x_i}{\sqrt{i}}\right) + 1`| :math:`x_i \in [-600, 600]`            |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Rosenbrock               | :math:`f(\mathbf{x}) = \sum_{i=1}^{d-1}\left[100\left(x_{i+1} - x_i^2\right)^2 + \left(x_i - 1\right)^2\right]` | :math:`x_i \in [-5, 10]`              |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Dropwave                 | :math:`f(\mathbf{x}) = -\frac{1 + \cos\left(12 \sqrt{x_1^2 + x_2^2}\right)}{0.5\left(x_1^2 + x_2^2\right) + 2}` | :math:`x_i \in [-5.12, 5.12]`        |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Langermann               | :math:`f(\mathbf{x}) = \sum_{i=1}^m c_i \exp\left(-\frac{1}{\pi} \sum_{j=1}^d \left(x_j - A_{ij}\right)^2\right)` | :math:`x_i \in [0, 10]`         |
|                          | :math:`\cos\left(\pi \sum_{j=1}^d\left(x_j - A_{ij}\right)^2\right)`                        |                                                  |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Rotated Hyper-Ellipsoid  | :math:`f(\mathbf{x}) = \sum_{i=1}^d \sum_{j=1}^i x_j^2`                                      | :math:`x_i \in [-65.536, 65.536]`                |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Sum of Different Powers  | :math:`f(\mathbf{x}) = \sum_{i=1}^d\left|x_i\right|^{i+1}`                                   | :math:`x_i \in [-1, 1]`                          |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Styblinski-Tang          | :math:`f(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^d\left(x_i^4 - 16 x_i^2 + 5 x_i\right)`        | :math:`x_i \in [-5, 5]`                          |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Powell                   | :math:`f(\mathbf{x}) = \sum_{i=1}^{d/4}\left(x_{4i-3} + 10 x_{4i-2}\right)^2`               | :math:`x_i \in [-4, 5]`                          |
|                          | :math:`+ 5\left(x_{4i-1} - x_{4i}\right)^2`                                                 |                                                  |
|                          | :math:`+ \left(x_{4i-2} - 2 x_{4i-1}\right)^4`                                              |                                                  |
|                          | :math:`+ 10\left(x_{4i-3} - x_{4i}\right)^4`                                                |                                                  |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Dixon-Price              | :math:`f(\mathbf{x}) = \left(x_1 - 1\right)^2 + \sum_{i=2}^d i\left(2 x_i^2 - x_{i-1}\right)^2`| :math:`x_i \in [-10, 10]`                    |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Ellipsoid                | :math:`f_2(\mathbf{x}) = \sum_{i=1}^D 10^{6 \frac{i-1}{D-1}} z_i^2 + f_{\mathrm{opt}}`       | :math:`x_i \in [-5, 5]`                          |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Discus                   | :math:`f(\mathbf{x}) = 10^6 x_1^2 + \sum_{i=2}^D x_i^2`                                      | :math:`x_i \in [-5, 5]`                          |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| BentCigar                | :math:`f(\mathbf{x}) = x_1^2 + 10^6 \sum_{i=2}^n x_i^2`                                      | :math:`x_i \in [-5, 5]`                          |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| SharpRidge               | :math:`f(\mathbf{x}) = x_1^2 + 100 \sqrt{\sum_{i=2}^D x_i^2}`                                | :math:`x_i \in [-5, 5]`                          |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Katsuura                 | :math:`f(\mathbf{x}) = \frac{10}{D^2} \prod_{i=1}^D \left(1 + i \sum_{j=1}^{32} \frac{\left|2^j x_i - \left[2^j x_i\right]\right|}{2^j}\right)^{10 / D^{1.2}}`| :math:`x_i \in [-5, 5]`            |
|                          | :math:`- \frac{10}{D^2} + f_{\mathrm{pen}}(\mathbf{x})`                                      |                                                  |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Weierstrass              | :math:`f_{16}(\mathbf{x}) = 10 \left(\frac{1}{D} \sum_{i=1}^D \sum_{k=0}^{11} \frac{1}{2^k} \cos \left(2 \pi 3^k\left(z_i + \frac{1}{2}\right)\right) - f_0\right)^3`| :math:`x_i \in [-5, 5]`    |
|                          | :math:`+ \frac{10}{D} f_{\mathrm{pen}}(\mathbf{x})`                                          |                                                  |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| DifferentPowers          | :math:`f(\mathbf{x}) = \sqrt{\sum_{i=1}^D\left|x_i\right|^{2 + 4 \frac{i-1}{D-1}}}`          | :math:`x_i \in [-5, 5]`                          |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Trid                     | :math:`f(\mathbf{x}) = \sum_{i=1}^d \left(x_i - 1\right)^2 - \sum_{i=2}^d x_i x_{i-1}`       | :math:`x_i \in [-d^2, d^2]`                      |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| LinearSlope              | :math:`f(\mathbf{x}) = \sum_{i=1}^D 5\left|s_i\right| - s_i x_i`                             | :math:`x_i \in [-5, 5]`                          |
|                          | :math:`s_i = \operatorname{sign}\left(x_i^{\mathrm{opt}}\right) 10^{\frac{i-1}{D-1}},`      |                                                  |
|                          | :math:`\text{for } i=1, \ldots, D`                                                          |                                                  |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Elliptic                 | :math:`f(\mathbf{x}) = \sum_{i=1}^D \left(10^6\right)^{\frac{i-1}{D-1}} x_i^2`               | :math:`x_i \in [-5, 5]`                          |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| PERM                     | :math:`f(\mathbf{x}) = \sum_{i=1}^d \left(\sum_{j=1}^d \left(j + \beta\right)\left(x_j^i - \frac{1}{j^i}\right)\right)^2`| :math:`x_i \in [-d, d]`           |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Power Sum                | :math:`f(\mathbf{x}) = \sum_{i=1}^d \left[\left(\sum_{j=1}^d x_j^i\right) - b_i\right]^2`    | :math:`x_i \in [0, d]`                           |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Zakharov                 | :math:`f(\mathbf{x}) = \sum_{i=1}^d x_i^2 + \left(\sum_{i=1}^d 0.5 i x_i\right)^2`           | :math:`x_i \in [-5, 10]`                         |
|                          | :math:`+ \left(\sum_{i=1}^d 0.5 i x_i\right)^4`                                             |                                                  |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Six-Hump Camel           | :math:`f(\mathbf{x}) = \left(4 - 2.1 x_1^2 + \frac{x_1^4}{3}\right) x_1^2 + x_1 x_2`         | :math:`x_1 \in [-3, 3], x_2 \in [-2, 2]`         |
|                          | :math:`+ \left(-4 + 4 x_2^2\right) x_2^2`                                                   |                                                  |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Michalewicz              | :math:`f(\mathbf{x}) = -\sum_{i=1}^d \sin \left(x_i\right) \sin ^{2 m}\left(\frac{i x_i^2}{\pi}\right)`| :math:`x_i \in [0, \pi]`                  |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+
| Moving Peak              | :math:`f(\mathbf{x}) = \sum_{i=1}^D \left(10^6\right)^{\frac{i-1}{D-1}} x_i^2`               | :math:`x_i \in [0, 100]`                         |
+--------------------------+---------------------------------------------------------------------------------------------+--------------------------------------------------+



