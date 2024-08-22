Algorithmic objects
===================

.. admonition:: Overview
   :class: info

   - `Register <https://link-to-definition>`_: How to register a new algorithmic Object to TransOpt.
   - `List of Algorithmic objects <https://link-to-parallelization>`_: Transfer learning for BO, .


Register
--------


List of Algorithmic Objects
---------------------------
The optimization framework includes a variety of state-of-the-art algorithms, each designed with specific features to address different classes of optimization problems. The table below provides a summary of the key algorithms available, categorized by their class, convenience for use, targeted objective(s), and any constraints they impose.

+------------------------+-----------+-----------------+------------------+-----------------------------------------------------+
| **Algorithm**          | **Class** | **Convenience** | **Objective(s)** | **Constraints**                                     |
+========================+===========+=================+======+===========+=====================================================+
| Hyper BO               | GA        | single          | x                | A modular implementation of a genetic algorithm.    |
|                        |           |                 |                  | It can be easily customized with different          |
|                        |           |                 |                  | evolutionary operators and applies to a broad       |
|                        |           |                 |                  | category of problems.                               |
+------------------------+-----------+-----------------+------------------+-----------------------------------------------------+
| Multi-task BO          | DE        | single          | x                | Different variants of differential evolution which  |
|                        |           |                 |                  | is a well-known concept for in continuous           |
|                        |           |                 |                  | optimization especially for global optimization.    |
+------------------------+-----------+-----------------+------------------+-----------------------------------------------------+

