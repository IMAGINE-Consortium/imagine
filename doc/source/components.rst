

.. _IMAGINE Components:

==================
IMAGINE Components
==================

In the following sections we describe each of the basic components of IMAGINE.
We also demonstrate how to write wrappers that allow the inclusion of external
(pre-existing) code, and provide code templates for this.

.. contents:: Contents
   :local:

.. _Fields:

------
Fields
------

In IMAGINE terminology, a **field** refers to any  *computational model* of a
spatially varying physical quantity, such as the Galactic Magnetic Field (GMF),
the thermal electron distribution, or the Cosmic Ray (CR) distribution.
Generally, a field object will have a set of parameters — e.g. a GMF field
object may have a pitch angle, scale radius, amplitude, etc.
*Field* objects are used as inputs by the `Simulators`_, which *simulate*
those physical models, i.e. they allow constructing construct *observables*
based on a set models.

During the sampling, the `Pipeline`_ does not handle fields directly  but
instead relies on `Field Factory`_ objects.
While the *field objects* will do the actual computation of the physical field,
given a set of physical parameters and a coordinate grid,
the *field factory objects* take care of book-keeping tasks: they hold the
parameter ranges and default values, they translate the dimensionless parameters
used by the sampler (always in the interval :math:`[0,1]`) to physical
parameters, they store `Priors`_ associated with each parameter of that *field*.

To convert ones own model into a IMAGINE-compatible field, one must create
subclass one of the base classes available the
:py:mod:`imagine.fields.basic_fields`, most likely using one of the available
templates to write a *wrapper* to the original code which are discussed in the
sections below.
If basic field type one is interested is *not* available as a basic field, one
can create it directly subclassing
:py:class:`imagine.fields.field.GeneralField` —
and if you think this could benefit the wider community, please consider
submitting a
`pull request <https://github.com/IMAGINE-Consortium/imagine/pulls>`_ or
openning an `issue <https://github.com/IMAGINE-Consortium/imagine/issues>`_
requesting the inclusion of the new field type!

It is assumed that **Field** objects can be expressed as a parametrised *mapping of
a coordinate grid into a physical field*. The grid is represented by a IMAGINE
:ref:`Grid` object, discussed in detail in the next section.
If the field of random or **stochastic** nature (e.g. the density field of
a turbulent medium), IMAGINE will compute a *fnite ensemble* of different
realisations which are later used in the inference to deteremine the likelihood
of the actual observation given accounting for the model's expected variability.

To test a Field class, :py:class:`FieldFoo`, one can
instantiate the field object::

    bar = FieldFoo(grid=example_grid, parameters={'answer': 42*u.cm}, ensemble_size=2)

where :py:obj:`example_grid` is a previously instantiated grid object. The argument
:py:data:`parameters` receives a dictionary of all the parameters used by
:py:obj:`FieldFoo`, these are usually expressed as dimensional quantities
(using :py:mod:`astropy.units`).
Finally, the argument :py:data:`ensemble_size`, as the name suggests allows
requestion a number of different realisations of the field (for non-stochastic
fields, all these will be references to the same data).

To further illustrate, assuming we are dealing with a scalar,
the (spherical) radial dependence of the above defined :py:obj:`bar`
can be easily plotted using::

    import matplotlib.pyplot as plt
    plt.plot(bar.grid.r_spherical.ravel(),
             bar.data[ensemble_index].ravel())


The design of any field is done writing a *subclass* of one of the classes in
:py:mod:`imagine.fields.basic_fields` or
:py:class:`imagine.GeneralField <imagine.fields.field.GeneralField>`
that overrides the method
:py:meth:`get_field(seed) <imagine.fields.field.GeneralField.get_field>`,
using it to computate the field on each spatial point.
For this, the coordinate grid on which the field should
be evaluated can be accessed from
:py:data:`self.grid <imagine.fields.field.GeneralField.grid>`
and the parameters from
:py:data:`self.parameters <imagine.fields.field.GeneralField.parameters>`.
The same parameters must be listed as *keys* in the
:py:data:`field_checklist <imagine.fields.field.GeneralField.field_checklist>`
dictionary (the values in the dictionary are used to supply extra information
to specific simulators, but can be left as `None` in the general case).


.. _Grid:

^^^^
Grid
^^^^

Field objects require (with the exception of `Dummy`_ fields) a coordinate grid
to operate.
In IMAGINE this is expressed as an instance of the
:py:class:`imagine.fields.grid.BaseGrid` class, which represents coordinates as
a set of three 3-dimensional arrays.
The grid object supports cartesian, cylindrical and spherical coordinate systems,
handling any conversions between these automatically through the properties.

The convention is that :math:`0` of the coordinates corresponds to the
Galaxy (or galaxy) centre, with the :math:`z` coordinate giving the distance to
the midplane.

To construct a grid with uniformly-distributed coordinates one can use the
:py:class:`imagine.fields.grid.UniformGrid` that accompanies IMAGINE.
For example, one can create a grid where the cylindrical coordinates are equally
spaced using::

    import imagine as img
    cylindrical_grid = img.UniformGrid(box=[[0.25*u.kpc, 15*u.kpc],
                                            [-180*u.deg, np.pi*u.rad],
                                            [-15*u.kpc, 15*u.kpc]],
                          resolution = [9,12,9],
                          grid_type = 'cylindrical')

The :py:data:`box` argument contains the lower and upper limits of the
coordinates (respectively :math:`r`, :math:`\phi` and :math:`z`),
:py:data:`resolution` specifies the number of points for each dimension, and
:py:data:`grid_type` chooses this to be cylindrical coordinates.

The coordinate grid can be accessed through the properties
:py:data:`cylindrical_grid.x`,
:py:data:`cylindrical_grid.y`,
:py:data:`cylindrical_grid.z`,
:py:data:`cylindrical_grid.r_cylindrical`,
:py:data:`cylindrical_grid.r_spherical`,
:py:data:`cylindrical_grid.theta` (polar angle), and
:py:data:`cylindrical_grid.phi` (azimuthal angle),
with conversions handled automatically.
Thus, if one wants to access, for instance, the corresponding :math:`x`
cartesian coordinate values, this can be done simply using::

    cylindrical_grid.x[:,:,:]


To create a personalised (non-uniform) grid, one needs to subclass
:py:class:`imagine.fields.grid.BaseGrid` and override the method :py:meth:`generate_coordinates <imagine.fields.grid.BaseGrid.generate_coordinates>`. The :py:class:`imagine.fields.grid.UniformGrid`
class should itself provide a good example/template of how to do this.


^^^^^^^^^^^^^^^^^
Thermal electrons
^^^^^^^^^^^^^^^^^

A new model for the distribution of thermal electrons can be introduced
subclassing :py:class:`imagine.fields.basic_fields.ThermalElectronDensityField`
according to the template below.

.. literalinclude:: ../../imagine/templates/thermal_electrons_template.py

Note that the return value of the method :py:meth:`get_field()  <imagine.fields.basic_fields.ThermalElectronDensityField>` must be of type
:py:class:`astropy.units.Quantity`, with shape consistent with the coordinate
grid, and units of :math:`\rm cm^{-3}`.

The template assumes that one already possesses a model for distribution of
thermal :math:`e^-` in a module :py:mod:`MY_GALAXY_MODEL`. Such model needs to
be able to map an arbitrary coordinate grid into densities.

Of course, one can also write ones model (if it is simple enough) into the
derived subclass definition. On example of a class derived from
:py:class:`imagine.fields.basic_fields.ThermalElectronDensityField` can be seen
bellow::

    from imagine import ThermalElectronDensityField

    class ExponentialThermalElectrons(ThermalElectronDensityField):
        """Example: thermal electron density of an (double) exponential disc"""

        field_name = 'exponential_disc_thermal_electrons'

        @property
        def field_checklist(self):
            return {'central_density' : None, 'scale_radius' : None, 'scale_height' : None}

        def get_field(self):
            R = self.grid.r_cylindrical
            z = self.grid.z
            Re = self.parameters['scale_radius']
            he = self.parameters['scale_height']
            n0 = self.parameters['central_density']

            return n0*np.exp(-R/Re)*np.exp(-np.abs(z/he))


^^^^^^^^^^^^^^^
Magnetic Fields
^^^^^^^^^^^^^^^

One can add a new model for magnetic fields subclassing
:py:class:`imagine.fields.basic_fields.MagneticField` as illustrated in the
template below.

.. literalinclude:: ../../imagine/templates/magnetic_field_template.py

It was assumed the existence of a hypothetical module :py:mod:`MY_GALAXY_MODEL`
which, given a set of parameters and three 3-arrays containing coordinate values,
computes the magnetic field vector at each point.

The method :py:meth:`get_field() <imagine.fields.basic_fields.MagneticField.get_field>` must return an :py:class:`astropy.units.Quantity`,
with shape `(Nx,Ny,Nz,3)` where `Ni` is the corresponding grid resolution and
the last axis corresponds to the component (with x, y and z associated with
indices 0, 1 and 2, respectively). The Quantity returned by the method must
correpond to a magnetic field (i.e. units must be :math:`\mu\rm G`, :math:`\rm G`,
:math:`\rm nT`, or similar).

A simple example, comprising a constant magnetic field can be seen below::

    from imagine import MagneticField

    class ConstantMagneticField(MagneticField):
        """Example: constant magnetic field"""
        field_name = 'constantB'

        @property
        def field_checklist(self):
            return {'Bx': None, 'By': None, 'Bz': None}

        def get_field(self):
            # Creates an empty array to store the result
            B = np.empty(self.data_shape) * self.parameters['Bx'].unit
            # For a magnetic field, the output must be of shape:
            # (Nx,Ny,Nz,Nc) where Nc is the index of the component.
            # Computes Bx
            B[:,:,:,0] = self.parameters['Bx']*np.ones(self.grid.shape)
            # Computes By
            B[:,:,:,1] = self.parameters['By']*np.ones(self.grid.shape)
            # Computes Bz
            B[:,:,:,2] = self.parameters['Bz']*np.ones(self.grid.shape)
            return B



^^^^^^^^^^^^^^^^^^^^
Cosmic ray electrons
^^^^^^^^^^^^^^^^^^^^

*Under development*

.. .. literalinclude:: ../../imagine/templates/cre_density_template.py


^^^^^
Dummy
^^^^^

There are situations when one may want to sample parameters which are not
used to evaluate a field on a grid before being sent to a Simulator object.
One possible use for this is representing a global property of the physical
system which affects the observations (for instance, some global property of
the ISM or, if modelling an external galaxy, the position of the galaxy).

Another common use of dummy fields is when a field is generated at runtime
*by the simulator*. One example is the range built-in fields available in
Hammurabi: instead of requesting IMAGINE to produce these fields and hand it to
Hammurabi to compute the associated synchrotron observables, one can use dummy
fields to request Hammurabi to generate these fields internally for a given
choice of parameters.

Using dummy fields to bypass the design of a full IMAGINE Field may simplify
implementation of a Simulator wrapper and (sometimes) may be offer good
performance. However, this practice of generating the actual field within the
Simulator *breaks the modularity of IMAGINE*, and it becomes impossible to
check the validity of the results plugging the same field on a different
Simulator. Thus, use this with care!


A dummy field can be implementing subclassing
:py:class:`imagine.fields.basic_fields.DummyField` as shown bellow.

.. literalinclude:: ../../imagine/templates/dummy_field_template.py


.. _Field Factory:

^^^^^^^^^^^^^^^^^^^^
Field Factory
^^^^^^^^^^^^^^^^^^^^

Field Factories, represented in IMAGINE by a subclass of
:py:class:`imagine.fields.field_factory.GeneralFieldFactory`
are an additional layer of infrastructure used by the samplers
to provide the connection between the sampling of points in the likelihood space
and the field object that will be given to the simulator.

A *Field Factory* object has a list of all of the field's parameters and a
list of the subset of those that are to be varied in the sampler
— the latter are called the **active parameters**.
The Field Factory also holds the allowed value ranges for each parameter,
the default values (which are used for inactive parameters) and the prior
distribution associated with each parameter.

At each step the
`Pipeline`_ it asks the field factory for the next point in parameter space,
and the factory gives it one in the form of a Field object that can be handed
to the simulator, which in turn provides simulated observables for comparison
with the measured observables in the likelihood module.

Given a Field `YourFieldClass` (which must be an instance of class derived from
:py:class:`imagine.fields.field.GeneralField`) the following template can be
used construct a field factory:

.. literalinclude:: ../../imagine/templates/field_factory_template.py

The object `Prior A` must be an instance of
:py:class:`imagine.priors.prior.GeneralPrior` (see section `Priors`_ for
details). A flat prior (i.e. a uniform, where all parameter values are
equally likely) can be set using the :py:class:`imagine.priors.basic_priors.FlatPrior` class.

One can initialize the Field Factory supplying the grid on which the
corresponding Field will be evaluated::

      myFactory = YourField_Factory(grid=cartesian_grid)

The factory object :py:obj:`myFactory` can now be handled to the `Pipeline`_,
which will generate new fields through the method :py:meth:`generate <imagine.fields.field_factory.GeneralFieldFactory.generate>`.


.. _Datasets:

--------
Datasets
--------

:py:class:`imagine.observables.dataset.Dataset` objects are helpers
used for the inclusion of observational data in IMAGINE.
They convert the measured data and uncertainties to a standard format which
can be later handed to an
:ref:`observable dictionary <ObservableDictionaries>`.
There are two main types of datasets: `Tabular datasets`_ and
`HEALPix datasets`_.



.. _Tabular datasets:

As the name indicates, in **tabular datasets** the observational data was
originally
in tabular format, i.e. a table where each row corresponds to a different
*position in the sky* and columns contain (at least) the sky coordinates,
the measurement and the associated error. A final requirement is that the
dataset is stored in a *dictionary-like* object i.e. the columns can be
selected by column name.

To construct a tabular dataset, one needs to instantiate
:py:class:`imagine.observables.dataset.TabularDataset`.

.. To exemplify this, we

::

    from astroquery.vizier import Vizier
    from imagine.observables.dataset import TabularDataset

    class FaradayRotationMao2010(TabularDataset):
        def __init__(self):
            # Fetches the catalogue
            catalog = Vizier.get_catalogs('J/ApJ/714/1170')[0]
            # Reads it to the TabularDataset (the catalogue obj actually contains units)
            super().__init__(catalog, name='fd', units=catalog['RM'].unit,
                             data_column='RM', error_column='e_RM',
                             lat_column='GLAT', lon_column='GLON')

.. _HEALPix datasets:

**HEALPix datasets** will generally comprise maps of the full-sky, where
`HEALPix <https://healpix.sourceforge.io/>`_ pixelation is employed.
:py:class:`imagine.observables.dataset.HEALPixDataset`
:py:class:`imagine.observables.dataset.SynchrotronHEALPixDataset`
:py:class:`imagine.observables.dataset.FaradayDepthHEALPixDataset`
:py:class:`imagine.observables.dataset.DispersionMeasureHEALPixDataset`


::

    import requests, io
    import numpy as np
    from astropy.io import fits
    from astropy import units as u
    from imagine.observables.dataset import FaradayDepthHEALPixDataset

    class FaradayDepthOppermann2012(FaradayDepthHEALPixDataset):
        def __init__(self, skip=None):
            # Fetches and reads the
            download = requests.get('https://wwwmpa.mpa-garching.mpg.de/ift/faraday/2012/faraday.fits')
            raw_dataset = fits.open(io.BytesIO(download.content))
            # Adjusts the data to the right format
            fd_raw = raw_dataset[3].data.astype(np.float)
            sigma_fd_raw = raw_dataset[4].data.astype(np.float)
            # Includes units in the data
            fd_raw *= u.rad/u.m/u.m
            sigma_fd_raw *= u.rad/u.m/u.m

            # If requested, makes it small, to save memory in this example
            if skip is not None:
                fd_raw = fd_raw[::skip]
                sigma_fd_raw = sigma_fd_raw[::skip]
            # Loads into the Dataset
            super().__init__(data=fd_raw, error=sigma_fd_raw)



.. _Observables:

-----------------------------------------
Measurements, Simulations and Covariances
-----------------------------------------

.. _ObservableDictionaries:

--




.. _Simulators:

----------
Simulators
----------

.. literalinclude:: ../../imagine/templates/simulator_template.py






.. _Likelihood:

-----------
Likelihoods
-----------

Likelihoods define how to quantitatively compare the simulated and measured
observables.  They are represented within IMAGINE by an instance of class
derived from :py:class:`imagine.likelihoods.likelihood.Likelihood`.
There are two pre-implemented subclasses within IMAGINE:

 * :py:class:`imagine.likelihoods.simple_likelihood.SimpleLikelihood`:
   this is the traditional method, which is like a :math:`\chi^2` based on the
   covariance matrix of the measurements (i.e., noise).
 * :py:class:`imagine.likelihoods.simple_likelihood.EnsembleLikelihood`:
    combines covariance matrices from measurements with the expected galactic
    variance from models that include a stochastic component.

Likelihoods need to be initialized before running the pipeline, and require measurements (at the front end).  In most cases, data sets will not have covariance matrices but only noise values, in which case the covariance matrix is only the diagonal.

::

  likelihood = EnsembleLikelihood(data, covariancematrix)

The optional input argument covariancematrix does not have to contain covariance matrices corresponding to all entries in input data.  The Likelihood automatically defines the proper way for the various cases.

If the EnsembleLikelihood is used, then the sampler will be run multiple times at each point in likelihood space to create an ensemble of simulated observables.  The covariance of these observables can then be included in the likelihood quantitatively so that the comparison of the measured observables




.. _Priors:

------
Priors
------

A powerful aspect of a fully Baysean analysis approach is the possibility of
explicitly stating any prior expectations about the parameter values based on
previous knowledge. A prior is represented by an instance of
:py:class:`imagine.priors.prior.GeneralPrior` or one of its subclasses.

To use a prior, one has to initialize it and include it in the associated
:ref:`Field Factory`. The simplest choice is a :py:class:`FlatPrior <imagine.priors.basic_priors.FlatPrior>`
(i.e. any parameter within the range are equally likely before the looking at
the observational data), which can be initialized in the following way::

  import imagine as img
  import astropy.units as u
  myFlatPrior = img.FlatPrior(interval=[-2*u.pc,10*u.pc])

where the range for this parameter was chosen to be between :math:`-2` and
:math:`10\rm pc`.

After the flat prior, a common choice is that the parameter values are
characterized by a Gaussian distribution around some central value.
This can be achieved using the
:py:class:`imagine.priors.basic_priors.GaussianPrior` class. As an example,
let us suppose one has a parameter which characterizes the strength of a
component of the magnetic field, and that ones prior expectation is that this
should be gaussian distributed with mean :math:`1\mu\rm G` and standard
deviation :math:`5\mu\rm G`. Moreover, let us assumed that the model only works
within the range :math:`[-30\mu \rm G, 30\mu G]`. A prior consistent with these
requirements can be achieved using::

  import imagine as img
  import astropy.units as u
  myGaussianPrior = img.GaussianPrior(mu=1*u.microgauss, sigma=5*u.microgauss,
                                      interval=[-30*u.microgauss,30*u.microgauss])






.. _Pipeline:

--------
Pipeline
--------




.. _Disclaimer:

-----------
Disclaimer
-----------

Nothing is written in stone and the base classes may be updated with time
(so, always remember to report the `code release <https://github.com/IMAGINE-Consortium/imagine/releases>`_ when you make use of IMAGINE).
Suggestions and improvements are welcome as GitHub `issues <https://github.com/IMAGINE-Consortium/imagine/issues/new>`_ or pull requests
