Quick Start
======================

TransOpt is a sophisticated system designed to facilitate transfer optimization services and experiments. It is composed of two parts: the agent and the web user interface. The agent is responsible for running the optimization algorithms and the web user interface is responsible for displaying the results and managing the experiments.

Start the backend agent:

::

  $ python transopt/agent/app.py



Web User Interface Mode
-----------------------
When TransOpt has been started successfully, go to the directory of webui and start the webui on your local machine. Enable the user interface mode with the following command:

::

  $ cd webui && npm start


Command Line Mode
-----------------
Except using TransOpt in the web UI mode, we also provide command line (CMD) mode for these users who don’t have a display screen for some reasons(like on a remote server).

::

  $ python transopt/agent/run_cli.py --seed 0 