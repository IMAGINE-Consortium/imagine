*****************************
Installation and dependencies
*****************************

Here you can find basic instructions for the installation of IMAGINE.

`Let us know <https://github.com/IMAGINE-Consortium/imagine/issues/new>`_
if you face major difficulties.

Download
--------

A copy of IMAGINE source can be downloaded from its main
`GitHub repository <https://github.com/IMAGINE-Consortium/imagine/>`_.
If one does not intend to contribute to the development, one should download
and unpack the
`latest release <https://github.com/IMAGINE-Consortium/imagine/releases/latest>`_:

.. code-block:: console
    wget https://github.com/IMAGINE-Consortium/imagine/archive/v2.0.0-alpha.tar.gz
    tar -xvvzf v2.0.0-alpha.tar.gz
    
    
Alternatively, if one is interested in getting involved with the development,
we recommend cloning the git repository

.. code-block:: console

    git clone git@github.com:IMAGINE-Consortium/imagine.git



Setting up the environment with conda
-------------------------------------

IMAGINE depends on a number of different python packages. The easiest way of
setting up your environment is using the *conda* package manager. This allows
one to setup a dedicated, contained, python environment in the user area.

Conda is the package manager of the `Anaconda <https://www.anaconda.com/>`_
Python distribution, which by default comes with a large number of packages frequently used in data science and scientific computing, as well as a GUI
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
-----------

A key dependency of IMAGINE is the
`Hammurabi code <https://bitbucket.org/hammurabicode/hamx/>`_,
a `HEALPix <https://healpix.jpl.nasa.gov/>`_-based
numeric simulator for Galactic polarized emission
(`arXiv:1907.00207 <https://arxiv.org/abs/1907.00207>`_).

Before proceeding with the IMAGINE installation, it is necessary to install
Hammurabi X following the instructions on its project
`wiki <https://bitbucket.org/hammurabicode/hamx/wiki/>`_.
Then, one needs to install the `hampyx` python wrapper:

.. code-block:: console

    conda activate imagine # if using conda
    cd PATH_TO_HAMMURABI
    pip install -e .


Installing
----------

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


Docker
------

A docker image is a convenient, light-weight and fast way of deploying IMAGINE.
One can either build the image directly with the Dockerfile provided in the
`source repository <https://github.com/IMAGINE-Consortium/imagine/tree/master/docker>`_ or pull our
`pre-built docker image <still_unavailable>`_.
