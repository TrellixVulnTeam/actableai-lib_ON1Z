.. ActableAI lib documentation master file, created by
   sphinx-quickstart on Thu Mar 31 15:24:35 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: https://docs.actable.ai/_images/logo.png
   :width: 600

Actable AI ML lib
=================

What is Actable-AI lib ?
------------------------

Actable-AI lib is the Machine Learning library used for `app.actable.ai <http://app.actable.ai>`_

This Machine Learning library is Open Source and licensed under Apache v2 so free to use
and modify.

Actable-AI implements Machine Learning models usable out of the box for multiple tasks.
You can discover each task usage with our notebooks:

   - `Classification <https://github.com/Actable-AI/actableai-lib/blob/master/examples/classification.ipynb>`_
   - `Regression <https://github.com/Actable-AI/actableai-lib/blob/master/examples/regression.ipynb>`_
   - `Association Rules <https://github.com/Actable-AI/actableai-lib/blob/master/examples/association_rules.ipynb>`_
   - `Forecast <https://github.com/Actable-AI/actableai-lib/blob/master/examples/forecast.ipynb>`_
   - `Clustering <https://github.com/Actable-AI/actableai-lib/blob/master/examples/clustering.ipynb>`_
   - `Correlation <https://github.com/Actable-AI/actableai-lib/blob/master/examples/correlation.ipynb>`_
   - `More <https://github.com/Actable-AI/actableai-lib/blob/master/examples>`_

Installations
-------------

To install ActableAI simply run:

.. code-block:: console

   pip install actableai-lib[gpu]

To install it from source:

.. code-block:: console

   git clone git@github.com:Actable-AI/actableai-lib.git --recursive
   cd actableai-lib
   pip install .[gpu]

To contribute as a developer (see *Contributions* section):

.. code-block:: console

   git clone git@github.com:Actable-AI/actableai-lib.git --recursive
   cd actableai-lib
   ./scripts/setup_hooks.sh
   pip install -e .[gpu,dev]

Note: You can replace [gpu] with [cpu] to have the CPU version of Actable-AI.

Main API
----------

ActableAI maintains tasks that are the core of our Analytics.
All the tasks and their usage are listed in this module.

.. toctree::
   :maxdepth: 2

   actableai.tasks


Contributions
-------------

.. toctree::
   :maxdepth: 2

   contributing

Index
-----

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
