Algorithmic objects
===================

.. admonition:: Overview
   :class: info
   
   - :ref:`Register <register-new-algorithm>`: How to register a new algorithmic Object to :ref:`TransOPT <home>`
   - :ref:`Supported Algorithms <alg>`: The list of the synthetic problems available in :ref:`TransOPT <home>`
   - :ref:`Algorithmic Objects<alg-obj>`: The list of the protein inverse folding problems available in :ref:`TransOPT <home>`


.. _register-new-algorithm:

Registering a New Algorithm in TransOPT
---------------------------------------

To register a new algorithm object in TransOPT, follow the steps outlined below:

1. **Import the Model Registry**

   First, you need to import the `model_registry` from the `transopt.agent.registry` module:

   .. code-block:: python

      from transopt.agent.registry import model_registry

2. **Define the Algorithm Object Name**

   Next, use the registry to define the name of your algorithm object. For example:

   .. code-block:: python

      @model_registry.register("MHGP")
      class MHGP(Model):
          pass

   In this example, the algorithm object is named "MHGP".

3. **Choose the Appropriate Base Class**

   Depending on the type of algorithm object you are creating, you must inherit from a specific base class. TransOPT provides several algorithm modules, each corresponding to a different base class:

   - **Surrogate Model**: Inherit from the `Model` class.
   - **Initialization Design**: Inherit from the `Sampler` class.
   - **Acquisition Function**: Inherit from the `AcquisitionBase` class.
   - **Pretrain Module**: Inherit from the `PretrainBase` class.
   - **Normalizer Module**: Inherit from the `NormalizerBase` class.

   For instance, in the example provided, we are creating a surrogate model, so the `MHGP` class inherits from the `Model` base class.

4. **Implement the Required Abstract Methods**

   Once the class is defined, you need to implement several abstract methods that are required by the `Model` base class. These methods include:

   .. code-block:: python

      def meta_fit(
          self,
          source_X : List[np.ndarray],
          source_Y : List[np.ndarray],
          optimize: Union[bool, Sequence[bool]] = True,
      ):
          pass

      def fit(
          self,
          X: np.ndarray,
          Y: np.ndarray,
          optimize: bool = False,
      ):
          pass

      def predict(
          self, X: np.ndarray, return_full: bool = False, with_noise: bool = False
      ) -> Tuple[np.ndarray, np.ndarray]:
          pass

   - **meta_fit**: This method is used to fit meta-data. If your transfer optimization algorithm requires meta-data, this is where you should leverage it.
   - **fit**: This method is used to fit the data for the current task.

By following these steps, you can successfully register a new algorithm object in TransOPT and implement the necessary functionality to integrate it into the framework.



.. _alg:

Supported Algorithms
--------------------



**Multi-Task Bayesian Optimization**:cite:`SwerskySA13`
This method extends multi-task Gaussian processes to transfer knowledge from previous optimizations to new tasks, improving the efficiency of Bayesian optimization. It leverages correlations between tasks to accelerate the optimization process, particularly useful in scenarios like hyperparameter tuning across different datasets.

---

**Practical Transfer Learning for Bayesian Optimization**:cite:`SnoekLA12`
This approach enhances Bayesian optimization by using an ensemble of Gaussian processes from previous tasks. It forms a robust surrogate model that quickly adapts to new tasks without requiring task-specific hyperparameter tuning, significantly reducing optimization time.

---

**Scalable Gaussian Process-Based Transfer Surrogates**:cite:`WistubaSS18`
This framework scales Gaussian processes for hyperparameter optimization by dividing metadata into subsets and training individual models. These models are combined into an ensemble, reducing computational complexity and improving optimization efficiency. 

---

**Few-Shot Bayesian Optimization (FSBO)**:cite:`WistubaG21`
FSBO redefines hyperparameter optimization as a few-shot learning problem using a deep kernel Gaussian process model. It quickly adapts to new tasks, achieving state-of-the-art results through efficient transfer learning. 

---

**Initializing Bayesian Optimization via Meta-Learning**:cite:`FeurerSH15`
This method uses meta-learning to improve the initialization of Sequential Model-based Bayesian Optimization (SMBO). By leveraging prior knowledge from similar datasets, it enhances performance, especially in complex tasks like combined algorithm selection and hyperparameter optimization. 

---

**Learning Hyperparameter Optimization Initializations**:cite:`WistubaSS15a`
This approach transfers knowledge from previous experiments to learn optimal initial hyperparameter configurations. It uses a differentiable estimator to accelerate optimization convergence, outperforming traditional initialization strategies. 

---

**Reinforced Few-Shot Acquisition Function Learning**:cite:`HsiehHL21`
This method improves acquisition functions in Bayesian optimization using a deep Q-network (DQN) trained in a few-shot learning framework. A Bayesian variant of DQN is used to mitigate overfitting, enhancing the exploration-exploitation trade-off.

---

**Meta-Learning Acquisition Functions for Transfer Learning**:cite:`VolppFFDFHD20`
This approach uses meta-learning to design acquisition functions tailored to specific objective functions. It leverages reinforcement learning to train a neural network-based acquisition function, particularly effective in transfer learning scenarios. 

---

**Hyperparameter Search Space Pruning**:cite:`WistubaSS15b`
This technique introduces a pruning strategy to SMBO, discarding regions of the search space unlikely to contain optimal configurations. It enhances optimization efficiency by avoiding unnecessary function evaluations.

---

**Learning Search Spaces for Bayesian Optimization**:cite:`PerroneS19`
This method automatically designs search spaces for Bayesian optimization by learning from historical data. It reduces the search space size, accelerating optimization and improving transfer learning capabilities. 





.. _alg-obj:

List of Algorithmic Objects
---------------------------
The optimization framework includes a variety of state-of-the-art algorithms, each designed with specific features to address different classes of optimization problems. The table below provides a summary of the key algorithms available, categorized by their class, convenience for use, targeted objective(s), and any constraints they impose.

+-----------------------------+----------------------------------------+
| **Component**               | **Method**                             |
+=============================+========================================+
| Problem Specification       | Prune [54]                             |
|                             | Box [33]                               |
+-----------------------------+----------------------------------------+
| Initialization Design       | Random/Sobol sequence                  |
|                             | Latin hypercube sampling               |
|                             | EA [53]                                |
|                             | aLi [55]                               |
+-----------------------------+----------------------------------------+
| Surrogate Model             | GP/Random forest                       |
|                             | MTGP [45]                              |
|                             | MHGP [14]                              |
|                             | PriorGP [50]                           |
|                             | DeepKernelGP [53]                      |
|                             | NeuralProcess [31]                     |
|                             | RGPE [9]                               |
|                             | SGPT [56]                              |
+-----------------------------+----------------------------------------+
| Acquisition Function        | EI/UCB/PI                              |
|                             | TAF [9, 56]                            |
|                             | FSAF [18]                              |
+-----------------------------+----------------------------------------+



+-------------------------+-------------------+---------------------------------------------------------------------------------------------------+
| **Component**           | **Method**        | **Description**                                                                                   |
+=========================+===================+===================================================================================================+
| Problem Specification   | S0 [1]            | Drop area that has no potential to generate promising points.                                     |
|                         | S1 [1]            | Drop area that has no potential to generate promising points.                                     |
|                         | S2 [2]            | Narrow the search space to cover all best points in similar Datasets.                             |
+-------------------------+-------------------+---------------------------------------------------------------------------------------------------+
| Initialization Design   | I0 [3]            | Random/Sobol sequence/ The typical initialization design without the use of any                   |
|                         |                   | information from data.                                                                            |
|                         | I1 [3]            | Use the evolutionary algorithm to find a set of points that can perform better on all             |
|                         |                   | similar datasets.                                                                                 |
|                         | I2 [4]            | Learn the optimal initial points by iteratively minimizing the meta loss defined as               |
|                         |                   | the average minimum loss across similar datasets.                                                 |
|                         | I3 [4]            | Learn the optimal initial points by iteratively minimizing the meta loss defined as               |
|                         |                   | the average minimum loss across similar datasets.                                                 |
+-------------------------+-------------------+---------------------------------------------------------------------------------------------------+
| Surrogate Model         | M0                | The two most commonly used surrogate models in conventional BO.                                   |
|                         | M1 [5]            | Model the data from the current task and similar datasets jointly through a                       |
|                         |                   | coregionalization kernel.                                                                         |
|                         | M2 [6]            | Learn a GP model using the residuals of predictions from models built on similar datasets.        |
|                         | M3 [7]            | Learn better parameters of GP from similar datasets.                                              |
|                         | M4 [3]            | A GP model with a kernel that includes a neural network, trained on similar datasets.             |
|                         | M5 [8]            | A transformer-based deep neural network that provides predictions and uncertainty                 |
|                         |                   | estimates.                                                                                        |
|                         | M6 [9]            | A model that ensembles GPs trained on similar datasets, with weights based on the                 |
|                         |                   | rank accuracy of their predictions on the current task.                                           |
|                         | M7 [10]           | A model that ensembles GPs trained on similar datasets, with weights based on the                 |
|                         |                   | kernel methods.                                                                                   |
|                         | M8 [10]           | A model that ensembles GPs trained on similar datasets, with weights based on the                 |
|                         |                   | kernel methods.                                                                                   |
|                         | M9 [10]           | A model that ensembles GPs trained on similar datasets, with weights based on the                 |
|                         |                   | kernel methods.                                                                                   |
|                         | M10 [10]          | A model that ensembles GPs trained on similar datasets, with weights based on the                 |
|                         |                   | kernel methods.                                                                                   |
+-------------------------+-------------------+---------------------------------------------------------------------------------------------------+
| Acquisition Function    | A0                | Typical acquisition functions only consider the model’s predictions.                              |
|                         | A1 [9, 10]        | Transfer acquisition functions leverage individual GP models trained on source tasks              |
|                         |                   | to improve the evaluation of new points.                                                          |
|                         | A2 [11]           | Train a neural network on similar datasets using reinforcement learning methods,                  |
|                         |                   | then use it as the acquisition function.                                                          |
|                         | A3 [11]           | Train a neural network on similar datasets using reinforcement learning methods,                  |
|                         |                   | then use it as the acquisition function.                                                          |
|                         | A4 [11]           | Train a neural network on similar datasets using reinforcement learning methods,                  |
|                         |                   | then use it as the acquisition function.                                                          |
|                         | A5 [11]           | Train a neural network on similar datasets using reinforcement learning methods,                  |
|                         |                   | then use it as the acquisition function.                                                          |
|                         | A6 [11]           | Train a neural network on similar datasets using reinforcement learning methods,                  |
|                         |                   | then use it as the acquisition function.                                                          |
|                         | A7 [11]           | Train a neural network on similar datasets using reinforcement learning methods,                  |
|                         |                   | then use it as the acquisition function.                                                          |
|                         | A8 [11]           | Train a neural network on similar datasets using reinforcement learning methods,                  |
|                         |                   | then use it as the acquisition function.                                                          |
+-------------------------+-------------------+---------------------------------------------------------------------------------------------------+




References
----------

.. bibliography:: TOS.bib
   :style: plain