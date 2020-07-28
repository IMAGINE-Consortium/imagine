*****************************
Installation and dependencies
*****************************

Here you can find basic instructions for the installation of IMAGINE.
There are two main installation routes:

  1. one can pull and run a :ref:`DockerInstallation` which allows
     you to setup and run IMAGINE by typing only two lines.
     IMAGINE will run in a container, i.e. separate from your system.
  2. one can :ref:`download and install <StandardInstallation>` IMAGINE and all
     the dependencies alongside your system.

The first option is particularly useful when one is a newcomer, interested
experimenting or when one is deploying IMAGINE in a cloud service or multiple
machines.

The second option is better if one wants to use ones pre-installed tools and
packages, or if one is interested in running on a computing cluster (running
docker images in some typical cluster settings may be difficult or impossible).

`Let us know <https://github.com/IMAGINE-Consortium/imagine/issues/new>`_
if you face major difficulties.

.. _DockerInstallation:

Docker installation
-------------------

This is a very convenient and fast way of deploying IMAGINE. You must first
pull the image of `one of IMAGINE's versions from GitHub <https://github.com/IMAGINE-Consortium/imagine/packages/327300/versions>`_, for example, the latest (*development*) version can be pulled
using:

.. code-block:: console

    sudo docker pull docker.pkg.github.com/imagine-consortium/imagine/imagine:latest

If you would like to start working (or testing IMAGINE) immediately, a
jupyter-lab session can be launched using:

.. code-block:: console

    sudo docker run -i -t -p 8888:8888 imagine:latest /bin/bash -c "source ~/jupyterlab.bash"

After running this, you may copy and paste the link with a token to a browser,
which will allow you to access the jupyter-lab session. From there you may,
for instance, navigate to the `imagine/tutorials` directory.


.. _StandardInstallation:

Standard installation
---------------------

Download
^^^^^^^^

A copy of IMAGINE source can be downloaded from its main
`GitHub repository <https://github.com/IMAGINE-Consortium/imagine/>`_.
If one does not intend to contribute to the development, one should download
and unpack the
`latest release <https://github.com/IMAGINE-Consortium/imagine/releases/latest>`_:

.. code-block:: console

    wget https://github.com/IMAGINE-Consortium/imagine/archive/v2.0.0-alpha.1.tar.gz
    tar -xvvzf v2.0.0-alpha.1.tar.gz


Alternatively, if one is interested in getting involved with the development,
we recommend cloning the git repository

.. code-block:: console

    git clone git@github.com:IMAGINE-Consortium/imagine.git



Setting up the environment with conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

IMAGINE depends on a number of different python packages. The easiest way of
setting up your environment is using the *conda* package manager. This allows
one to setup a dedicated, contained, python environment in the user area.

Conda is the package manager of the `Anaconda <https://www.anaconda.com/>`_
Python distribution, which by default comes with a large number of packages
frequently used in data science and scientific computing, as well as a GUI
installer and other tools.

A lighter, recommended, alternative is the
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ distribution,
which allows one to use the conda commands to install only what is actually
needed.

Once one has installed (mini)conda, one can download and install the IMAGINE
environment in the following way:

.. code-block:: console

    conda env create --file=imagine_conda_env.yml
    conda activate imagine
    python -m ipykernel install --user --name imagine --display-name "Python (imagine)"

The (optional) last line creates a Jupyter kernel linked to the new conda
environment (which is required, for example, for executing the tutorial
Jupyter notebooks).

Whenever one wants to run an IMAGINE script, one has to first activate the
associated environment with the command `conda activate imagine`.
To leave this environment one can simply run `conda deactivate`


Hammurabi X
^^^^^^^^^^^

A key dependency of IMAGINE is the
`Hammurabi X <https://github.com/hammurabi-dev/hammurabiX>`_ code,
a `HEALPix <https://healpix.jpl.nasa.gov/>`_-based
numeric simulator for Galactic polarized emission
(`arXiv:1907.00207 <https://arxiv.org/abs/1907.00207>`_).

Before proceeding with the IMAGINE installation, it is necessary to install
Hammurabi X following the instructions on its project
`wiki <https://github.com/hammurabi-dev/hammurabiX/wiki>`_.
Then, one needs to install the `hampyx` python wrapper:

.. code-block:: console

    conda activate imagine # if using conda
    cd PATH_TO_HAMMURABI
    pip install -e .


Installing
^^^^^^^^^^

After downloading, setting up the environment and installing Hammurabi X,
IMAGINE can finally be installed through:

.. code-block:: console

    conda activate imagine # if using conda
    cd IMAGINE_PATH
    pip install .

If one does not have admistrator/root privileges/permissions, one may instead
want to use

.. code-block:: console

    pip install --user .

Also, if you are working on further developing or modifying IMAGINE for your own needs, you may wish to use the `-e` flag, to keep links to the source directory instead of copying the files,

.. code-block:: console

    pip install -e .


