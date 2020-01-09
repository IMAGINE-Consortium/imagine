===========
Conventions
===========

To preserve the modularity and flexibility of IMAGINE, one should be strict
about using the provided base classes and also adhere to a small number of
conventions listed in this page.

Nothing is written in stone and these may be updated with time (so, always
remember to report the code release when you make use of IMAGINE).
Suggestions and improvements are welcome as GitHub `issues <https://github.com/IMAGINE-Consortium/imagine/issues/new>`_ or pull requests

-----------
Field types
-----------

:doc:`Field <imagine.fields>` and :doc:`Simulator <imagine.simulators>` classes
should be written so that different Simulators could be interchanged for a
particular set of fields (e.g. one should be able to any Simulator
that computes Faraday Rotation Measure and apply to the same magnetic and
thermal electron density fields). Likewise, the Simulator should be agnostic
of the model which actually computed the Field (e.g. it should not formally
matter whether the magnetic field was computed by a simple function which
returns a constant field or, say, the GalMag model).



=============================  ============  ===================  ========================================
 Field type                     Shape        Units                 Description
=============================  ============  ===================  ========================================
'magnetic_field'               (Nx,Ny,Nz,N)  :math:`\mu\rm G`     :math:`\vec{B}`-field
'cosmic_ray_electron_density'  (Nx,Ny,Nz,NE)    :math:`\rm cm^{-3}`  Number density, :math:`n_{\rm cr}`.
'thermal_electron_density'     (Nx,Ny,Nz)    :math:`\rm cm^{-3}`  Number density, :math:`n_{\rm e}`
'dummy'                        None          None                 Only parameter values.
=============================  ============  ===================  ========================================

