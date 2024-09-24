.. TransOPT documentation master file, created by
   sphinx-quickstart on Mon Aug 19 16:00:09 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _home:


TransOPT: Transfer Optimization System for Automated Configuration
==================================================================
TransOPT is an open-source software platform designed to facilitate the design, benchmarking, and application of transfer learning for Bayesian optimization (TLBO) algorithms through a modular, data-centric framework.

.. raw:: html
   :file: home/guide.html

Features
********************************************************************************
TransOPT offers diverse features covering various aspects of transfer optimization.

.. raw:: html
   :file: home/feature.html



Available Algorithmic Objects
********************************************************************************

.. csv-table::
   :header: "Algorithmic Objects", "Type", "Source Algorithm", "Description"
   :widths: 60, 10, 60, 100
   :file: usage/algorithms.csv



Contents
********************************************************************************

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   features
   usage/algorithms
   usage/problems
   usage/results
   usage/visualization
   usage/cli
   development/architecture
   development/api_reference
   faq




Contact
********************************************************************************
| **Peili Mao**  
| *University of Electronic Science and Technology of China*  
| *Department of Computer Science*  
| **E-mail**:  
| peili.z.mao@gmail.com



Cite
********************************************************************************

If you have utilized our framework for research purposes, we kindly invite you to cite our publication as follows:

BibTex:

.. code-block:: bibtex

    @ARTICLE{TransOPT,
      title = {{TransOPT}: Transfer Optimization System For Automated Configuration},
      author = {Author Name and Collaborator Name},
      url = {https://github.com/maopl/TransOPT},
      year = {2024}
    }



