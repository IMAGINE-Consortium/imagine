*****************************
Installation and dependencies
*****************************

Here you can find basic instructions for the installation of IMAGINE.
A copy of IMAGINE source can be downloaded from its main
`GitHub repository <https://github.com/IMAGINE-Consortium/imagine/>`_.
Alternatively, one can opt for using a `Docker`_ image (see below).

`Let us know <https://github.com/IMAGINE-Consortium/imagine/issues/new>`_ if you face major difficulties.


Setting up the environment
==========================

Before installing IMAGINE, one needs to prepare one has to prepare the
environment installing the `Hammurabi`_ code and the following packages:

 * `Python3 <https://python.org>`_
 * `NumPy <https://numpy.org/>`_
 * `mpi4py <https://mpi4py.readthedocs.io/>`_
 * `PyMultiNest <https://johannesbuchner.github.io/PyMultiNest/>`_
 * `Dynesty <https://dynesty.readthedocs.io/en/latest/>`_
 * `healpy <https://healpy.readthedocs.io/>`_
 * `h5py <https://docs.h5py.org/>`_

The procedure is *significantly* simplified setting up an `Conda`_ environment.


Hammurabi
---------

A key dependency of IMAGINE is the
`Hammurabi code <https://bitbucket.org/hammurabicode/hamx/>`_,
a `HEALPix <https://healpix.jpl.nasa.gov/>`_-based
numeric simulator for Galactic polarized emission
(`arXiv:1907.00207 <https://arxiv.org/abs/1907.00207>`_).
Instructions for its installation can be found on its project
`wiki <https://bitbucket.org/hammurabicode/hamx/wiki/>`_.



Conda
-----

Users of `Anaconda <https://www.anaconda.com/>`_ or
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
users can automatically setup a dedicated environment with all the
dependencies (except for hammurabi) using the YAML file provided:

.. code-block:: console

    conda env create --file=imagine_conda_env.yml
    conda activate imagine
    python -m ipykernel install --user --name imagine --display-name "Python (imagine)"

The (optional) last line creates a Jupyter kernel linked to the new conda
environment (which is required, for example, for executing the tutorial
notebooks).


Installing
==========

After downloading IMAGINE and extracting/cloning it to IMAGINE_PATH, simply
do

.. code-block:: console

    cd IMAGINE_PATH
    pip install .

If you do not have admistrator/root privileges/permissions, you may instead want to use

.. code-block:: console

    pip install --user .

Also, if you are working on further developing or modifying IMAGINE for your own needs, you may wish to use the `-e` flag, to keep links to the source directory instead of copying the files,

.. code-block:: console

    pip install -e .



Docker
======

A docker image is a convenient, light-weight and fast way of deploying IMAGINE.
One can either build the image directly with the Dockerfile provided in the
`source repository <https://github.com/IMAGINE-Consortium/imagine/tree/master/docker>`_ or pull our
`pre-built docker image <https://cloud.docker.com/u/ricphy/repository/docker/ricphy/imagine>`_.
