Algorithmic objects
===================

.. admonition:: Overview
   :class: info

Register new Algorithmic Object
   - :ref:`Register <registering-new-algorithm>`: How to register a new algorithmic Object to :ref:`TransOpt <home>`
   - :ref:`Supported Algorithms <alg>`: The list of the synthetic problems available in :ref:`TransOpt <home>`
   - :ref:`Algorithmic Objects<alg-obj>`: The list of the protein inverse folding problems available in :ref:`TransOpt <home>`



.. _registering-new-algorithm:

Registering a new Algorithmic Object
------------------------------------



.. _alg:

Supported Algorithms
--------------------

**Multi-Task Bayesian Optimization**  
This method extends multi-task Gaussian processes to transfer knowledge from previous optimizations to new tasks, improving the efficiency of Bayesian optimization. It leverages correlations between tasks to accelerate the optimization process, particularly useful in scenarios like hyperparameter tuning across different datasets. :cite:`SwerskySA13`

---

**Practical Transfer Learning for Bayesian Optimization**  
This approach enhances Bayesian optimization by using an ensemble of Gaussian processes from previous tasks. It forms a robust surrogate model that quickly adapts to new tasks without requiring task-specific hyperparameter tuning, significantly reducing optimization time. :cite:`SnoekLA12`

---

**Scalable Gaussian Process-Based Transfer Surrogates**  
This framework scales Gaussian processes for hyperparameter optimization by dividing metadata into subsets and training individual models. These models are combined into an ensemble, reducing computational complexity and improving optimization efficiency. :cite:`WistubaSS18`

---

**Few-Shot Bayesian Optimization (FSBO)**  
FSBO redefines hyperparameter optimization as a few-shot learning problem using a deep kernel Gaussian process model. It quickly adapts to new tasks, achieving state-of-the-art results through efficient transfer learning. :cite:`WistubaG21`

---

**Initializing Bayesian Optimization via Meta-Learning**  
This method uses meta-learning to improve the initialization of Sequential Model-based Bayesian Optimization (SMBO). By leveraging prior knowledge from similar datasets, it enhances performance, especially in complex tasks like combined algorithm selection and hyperparameter optimization (CASH). :cite:`FeurerSH15`

---

**Learning Hyperparameter Optimization Initializations**  
This approach transfers knowledge from previous experiments to learn optimal initial hyperparameter configurations. It uses a differentiable estimator to accelerate optimization convergence, outperforming traditional initialization strategies. :cite:`WistubaSS15a`

---

**Reinforced Few-Shot Acquisition Function Learning**  
This method improves acquisition functions in Bayesian optimization using a deep Q-network (DQN) trained in a few-shot learning framework. A Bayesian variant of DQN is used to mitigate overfitting, enhancing the exploration-exploitation trade-off. :cite:`HsiehHL21`

---

**Meta-Learning Acquisition Functions for Transfer Learning**  
This approach uses meta-learning to design acquisition functions tailored to specific objective functions. It leverages reinforcement learning to train a neural network-based acquisition function, particularly effective in transfer learning scenarios. :cite:`VolppFFDFHD20`

---

**Hyperparameter Search Space Pruning**  
This technique introduces a pruning strategy to SMBO, discarding regions of the search space unlikely to contain optimal configurations. It enhances optimization efficiency by avoiding unnecessary function evaluations. :cite:`WistubaSS15b`

---

**Learning Search Spaces for Bayesian Optimization**  
This method automatically designs search spaces for Bayesian optimization by learning from historical data. It reduces the search space size, accelerating optimization and improving transfer learning capabilities. :cite:`PerroneS19`





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


.. bibliography:: TOS.bib
   :style: plain